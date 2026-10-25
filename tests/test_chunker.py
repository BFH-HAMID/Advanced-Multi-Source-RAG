"""Tests for sentence-aware, token-bounded chunking."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.errors import IngestionError
from src.models import RawDocument, SourceType
from src.processing.chunker import TokenChunker, chunk_text
from src.text_utils import count_tokens, split_sentences

LONG_TEXT = " ".join(
    f"Sentence number {index} explains policy detail {index} for employees in detail." for index in range(40)
)


def test_chunks_respect_token_budget(settings: Settings) -> None:
    chunker = TokenChunker(settings=settings)
    chunks = chunker.split_text(LONG_TEXT)
    assert len(chunks) > 1
    for chunk in chunks:
        assert count_tokens(chunk) <= settings.chunk_size_tokens


def test_chunks_have_overlap(settings: Settings) -> None:
    chunker = TokenChunker(chunk_size_tokens=60, overlap_tokens=20, settings=settings)
    chunks = chunker.split_text(LONG_TEXT)
    assert len(chunks) >= 3
    tail = set(chunks[0].split()[-6:])
    head = set(chunks[1].split()[:12])
    assert tail & head, "consecutive chunks must share overlapping text"


def test_chunk_ids_are_deterministic(settings: Settings) -> None:
    chunker = TokenChunker(settings=settings)
    first = chunker.chunk_text(LONG_TEXT, source="policy.txt", doc_id="doc-1")
    second = chunker.chunk_text(LONG_TEXT, source="policy.txt", doc_id="doc-1")
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert [chunk.chunk_index for chunk in first] == list(range(len(first)))


def test_no_empty_chunks_and_metadata(settings: Settings) -> None:
    chunker = TokenChunker(settings=settings)
    chunks = chunker.chunk_text(
        LONG_TEXT,
        source="policy.pdf",
        source_type=SourceType.PDF,
        page=3,
        doc_id="doc-2",
        title="Policy",
        metadata={"loader": "pdf"},
    )
    assert all(chunk.text.strip() for chunk in chunks)
    assert chunks[0].page == 3
    assert chunks[0].source_type is SourceType.PDF
    assert chunks[0].metadata["chunk_total"] == len(chunks)
    assert chunks[0].metadata["loader"] == "pdf"
    assert chunks[0].citation_key == "policy.pdf, p.3"


def test_bangla_sentence_splitting() -> None:
    text = "প্রতি বছর ২০ দিন ছুটি পাবেন। অসুস্থতার ছুটি ১০ দিন। রিমোট কাজ সপ্তাহে ৩ দিন।"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].startswith("প্রতি বছর")


def test_bangla_chunking_keeps_sentences_intact(settings: Settings) -> None:
    text = " ".join(f"বাক্য নম্বর {index} একটি নীতিমালা ব্যাখ্যা করে।" for index in range(30))
    chunks = TokenChunker(chunk_size_tokens=50, overlap_tokens=10, settings=settings).split_text(text)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith("।")


def test_overlong_sentence_is_split(settings: Settings) -> None:
    sentence = " ".join(["word"] * 400) + "."
    chunks = TokenChunker(chunk_size_tokens=50, overlap_tokens=5, settings=settings).split_text(sentence)
    assert len(chunks) > 1
    assert all(count_tokens(chunk) <= 50 for chunk in chunks)


def test_document_chunking_preserves_provenance(settings: Settings) -> None:
    document = RawDocument(
        source="handbook.pdf",
        source_type=SourceType.PDF,
        text=LONG_TEXT,
        page=2,
        title="Handbook",
        metadata={"loader": "pdf"},
    )
    chunker = TokenChunker(settings=settings)
    chunks = chunker.chunk_documents([document])
    assert chunks
    assert {chunk.doc_id for chunk in chunks} == {document.doc_id}
    assert {chunk.source for chunk in chunks} == {"handbook.pdf"}
    assert {chunk.metadata["title"] for chunk in chunks} == {"Handbook"}


def test_invalid_overlap_rejected(settings: Settings) -> None:
    with pytest.raises(IngestionError):
        TokenChunker(chunk_size_tokens=50, overlap_tokens=60, settings=settings)


def test_chunk_text_helper_returns_strings(settings: Settings) -> None:
    chunks = chunk_text(LONG_TEXT, settings=settings)
    assert chunks and all(isinstance(chunk, str) for chunk in chunks)
