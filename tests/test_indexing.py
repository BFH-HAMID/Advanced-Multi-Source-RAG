"""Tests for embeddings, the three indexes and the index builder."""

from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.errors import IndexingError
from src.indexing.build_all import IndexBuilder, chunks_fingerprint
from src.indexing.embeddings import (
    CachedEmbedding,
    HashingEmbedding,
    get_embed_model,
    get_embedding_dimension,
)
from src.indexing.graph_index import (
    GraphIndex,
    HeuristicTripleExtractor,
    normalise_entity,
    normalise_relation,
)
from src.indexing.sentence_window import SentenceWindowIndex
from src.indexing.vector_index import VectorIndex
from src.models import Chunk, SourceType


def make_chunk(chunk_id: str, text: str, page: int = 1, source: str = "doc.pdf") -> Chunk:
    """Helper to build a chunk."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-1",
        text=text,
        chunk_index=0,
        source=source,
        source_type=SourceType.PDF,
        page=page,
        token_count=len(text.split()),
        metadata={"keywords": ["leave", "policy"]},
    )


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_hashing_embedding_is_deterministic_and_normalised() -> None:
    model = HashingEmbedding(embed_dim=128)
    first = model.get_text_embedding("annual leave policy")
    second = model.get_text_embedding("annual leave policy")
    assert first == second
    assert len(first) == 128
    norm = sum(value * value for value in first) ** 0.5
    assert 0.99 <= norm <= 1.01


def test_hashing_embedding_supports_bangla() -> None:
    model = HashingEmbedding(embed_dim=128)
    vector = model.get_text_embedding("বার্ষিক ছুটি কত দিন")
    assert len(vector) == 128
    assert any(abs(value) > 0 for value in vector)


def test_embedding_cache_reuses_vectors(settings: Settings) -> None:
    model = CachedEmbedding(
        inner=HashingEmbedding(embed_dim=64),
        cache_path=str(settings.index_dir / "cache.db"),
    )
    first = model.get_text_embedding("cached passage")
    second = model.get_text_embedding("cached passage")
    assert first == second
    assert model.stats["hits"] >= 1
    assert model.stats["misses"] >= 1


def test_get_embed_model_offline(settings: Settings) -> None:
    model = get_embed_model(settings, backend="hashing", use_cache=False)
    assert isinstance(model, HashingEmbedding)
    assert get_embedding_dimension(model) == settings.embedding_dim


# --------------------------------------------------------------------------- #
# Vector index
# --------------------------------------------------------------------------- #
def test_vector_index_build_and_search(settings: Settings) -> None:
    index = VectorIndex(settings, embed_model=get_embed_model(settings, use_cache=False))
    chunks = [
        make_chunk("c1", "Annual leave is 20 days per calendar year."),
        make_chunk("c2", "The security team handles incident response.", page=2),
    ]
    assert index.build(chunks) == 2
    assert index.exists()

    results = index.search("annual leave", top_k=2)
    assert results
    assert "annual leave" in results[0].text.lower()
    assert results[0].source == "doc.pdf"


def test_vector_index_persists_across_instances(settings: Settings) -> None:
    embed_model = get_embed_model(settings, use_cache=False)
    index = VectorIndex(settings, embed_model=embed_model)
    index.build([make_chunk("c1", "Learning budget is 50,000 BDT.")])
    reopened = VectorIndex(settings, embed_model=embed_model)
    assert reopened.exists()
    assert reopened.search("learning budget", top_k=1)


def test_vector_index_requires_chunks(settings: Settings) -> None:
    with pytest.raises(IndexingError):
        VectorIndex(settings).build([])


def test_vector_index_missing_raises(settings: Settings) -> None:
    with pytest.raises(IndexingError, match="no vector index"):
        VectorIndex(settings).search("anything")


# --------------------------------------------------------------------------- #
# Sentence-window index
# --------------------------------------------------------------------------- #
def test_sentence_window_nodes_carry_windows(settings: Settings) -> None:
    index = SentenceWindowIndex(settings, embed_model=get_embed_model(settings, use_cache=False))
    text = "Sentence one about leave. Sentence two about remote work. Sentence three about budget."
    nodes = index._make_nodes([make_chunk("c1", text)])
    assert len(nodes) >= 3
    assert all("window" in node.metadata for node in nodes)
    assert any("window_size" in node.metadata for node in nodes)


def test_sentence_window_search_returns_window_text(settings: Settings) -> None:
    index = SentenceWindowIndex(settings, embed_model=get_embed_model(settings, use_cache=False))
    index.build(
        [
            make_chunk(
                "c1",
                "Annual leave is 20 days. Sick leave is 10 days. Remote work is 3 days per week.",
            )
        ]
    )
    contexts = index.search("how many sick leave days", top_k=1)
    assert contexts
    assert "sick leave" in contexts[0].text.lower()
    assert contexts[0].metadata["window_size"] == settings.sentence_window_size
    assert contexts[0].metadata["matched_sentence"]


# --------------------------------------------------------------------------- #
# Graph index
# --------------------------------------------------------------------------- #
def test_heuristic_extractor_finds_triples() -> None:
    triples = HeuristicTripleExtractor().extract(
        "Acme Bangladesh is headquartered in Dhaka. Acme Bangladesh owns Acme Labs."
    )
    assert triples
    subjects = {subject for subject, _, _ in triples}
    assert any("Acme" in subject for subject in subjects)


def test_graph_index_build_persist_load(settings: Settings) -> None:
    graph = GraphIndex(settings=settings, use_llm=False)
    stats = graph.build([make_chunk("c1", "Acme Bangladesh is headquartered in Dhaka.")])
    assert stats.entities >= 2
    assert stats.triples >= 1
    assert graph.exists()

    reloaded = GraphIndex(settings=settings, use_llm=False)
    reloaded.load()
    assert reloaded.graph.number_of_nodes() == graph.graph.number_of_nodes()
    assert reloaded.chunk_store.keys() == graph.chunk_store.keys()
    payload = json.loads(graph.graph_path.read_text(encoding="utf-8"))
    assert payload["version"] >= 1
    assert payload["nodes"]


def test_graph_retrieval_by_entity(settings: Settings) -> None:
    graph = GraphIndex(settings=settings, use_llm=False)
    graph.build(
        [
            make_chunk("c1", "Acme Bangladesh owns Acme Labs. Acme Labs is located in Dhaka."),
            make_chunk("c2", "The learning budget is BDT 50,000 per employee.", page=2),
        ]
    )
    contexts = graph.search("Where is Acme Labs?", top_k=2)
    assert contexts
    assert "Acme" in contexts[0].text


def test_graph_find_entities_scores_matches(settings: Settings) -> None:
    graph = GraphIndex(settings=settings, use_llm=False)
    graph.build([make_chunk("c1", "Nusrat Jahan manages People Operations in Dhaka.")])
    matches = graph.find_entities("Who is Nusrat Jahan?")
    assert matches
    assert matches[0][1] >= 0.6


def test_graph_stats_and_triples(settings: Settings) -> None:
    graph = GraphIndex(settings=settings, use_llm=False)
    graph.build([make_chunk("c1", "Acme Labs is located in Chattogram.")])
    stats = graph.stats()
    assert stats["entities"] >= 2
    assert isinstance(stats["top_entities"], list)
    assert graph.export_triples()


def test_graph_normalisation_helpers() -> None:
    assert normalise_entity("  Acme Corp. ") == "acme corp"
    assert normalise_relation("Located In") == "located_in"
    assert normalise_relation("") == "related_to"


def test_graph_requires_chunks(settings: Settings) -> None:
    with pytest.raises(IndexingError):
        GraphIndex(settings=settings, use_llm=False).build([])


# --------------------------------------------------------------------------- #
# Index builder
# --------------------------------------------------------------------------- #
def test_build_all_creates_every_index(settings: Settings, populated_db) -> None:
    builder = IndexBuilder(settings=settings, db=populated_db)
    report = builder.build_all()

    assert report.vector_nodes > 0
    assert report.window_nodes > 0
    assert report.graph_entities > 0
    assert report.errors == []
    assert all(builder.indexes_exist().values())
    assert settings.manifest_path.exists()


def test_manifest_tracks_fingerprint_and_staleness(settings: Settings, populated_db) -> None:
    builder = IndexBuilder(settings=settings, db=populated_db)
    builder.build_all()
    assert builder.is_stale() is False

    from src.models import RawDocument

    new_document = RawDocument(
        source="new.txt",
        source_type=SourceType.TEXT,
        text="A brand new document about travel policy.",
    )
    populated_db.insert_documents([new_document])
    populated_db.insert_chunks(
        [
            Chunk(
                chunk_id="new-chunk",
                doc_id=new_document.doc_id,
                text=new_document.text,
                chunk_index=0,
                source=new_document.source,
                source_type=SourceType.TEXT,
            )
        ]
    )
    assert builder.is_stale() is True


def test_chunks_fingerprint_is_order_independent() -> None:
    first = make_chunk("a", "text one")
    second = make_chunk("b", "text two")
    assert chunks_fingerprint([first, second]) == chunks_fingerprint([second, first])


def test_build_all_requires_chunks(settings: Settings) -> None:
    from src.storage.document_db import DocumentDB

    builder = IndexBuilder(settings=settings, db=DocumentDB(settings=settings))
    with pytest.raises(IndexingError, match="no chunks"):
        builder.build_all()
