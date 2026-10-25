"""Tests for the multi-retriever merge/dedupe logic and the concrete retrievers."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from src.config import Settings
from src.errors import RetrievalError
from src.models import RetrievedContext
from src.retrieval.base import BaseRetriever
from src.retrieval.multi_retriever import MultiRetriever, merge_results
from src.retrieval.vector_retriever import VectorRetriever


def make_context(node_id: str, text: str, score: float, retriever: str, page: int | None = None) -> RetrievedContext:
    """Helper to build a retrieved context."""
    return RetrievedContext(
        node_id=node_id,
        text=text,
        score=score,
        source="doc.pdf",
        page=page,
        retriever=retriever,
        retrievers=[retriever],
    )


# --------------------------------------------------------------------------- #
# merge_results
# --------------------------------------------------------------------------- #
def test_merge_deduplicates_by_node_id() -> None:
    groups = {
        "vector": [make_context("n1", "alpha text", 0.9, "vector")],
        "graph": [make_context("n1", "alpha text", 0.4, "graph")],
    }
    merged = merge_results(groups)
    assert len(merged) == 1
    assert merged[0].retrievers == ["vector", "graph"]
    assert merged[0].metadata["retrieval_score"] is not None


def test_merge_deduplicates_by_text_when_ids_differ() -> None:
    groups = {
        "vector": [make_context("v1", "Same   passage here", 0.8, "vector")],
        "sentence_window": [
            make_context("w1", "Same passage here", 0.6, "sentence_window", page=2)
        ],
    }
    merged = merge_results(groups)
    assert len(merged) == 1
    assert set(merged[0].retrievers) == {"vector", "sentence_window"}


def test_merge_orders_by_fused_score() -> None:
    groups = {
        "vector": [
            make_context("a", "first", 0.9, "vector"),
            make_context("b", "second", 0.5, "vector"),
        ],
        "graph": [make_context("b", "second", 0.8, "graph")],
    }
    merged = merge_results(groups)
    # "b" appears in both retrievers, so RRF ranks it first.
    assert merged[0].node_id == "b"
    assert merged[0].score > merged[1].score


def test_merge_respects_weights() -> None:
    groups = {
        "vector": [make_context("v", "vector hit", 1.0, "vector")],
        "graph": [make_context("g", "graph hit", 1.0, "graph")],
    }
    merged = merge_results(groups, weights={"vector": 5.0, "graph": 0.1})
    assert merged[0].node_id == "v"


def test_merge_truncates_to_top_k() -> None:
    groups = {
        "vector": [make_context(f"n{i}", f"text {i}", 1.0 - i / 100, "vector") for i in range(20)]
    }
    assert len(merge_results(groups, top_k=5)) == 5


def test_merge_without_rrf_uses_normalised_scores() -> None:
    groups = {
        "vector": [make_context("a", "a text", 1.0, "vector"), make_context("b", "b text", 0.5, "vector")],
        "graph": [make_context("c", "c text", 0.1, "graph")],
    }
    merged = merge_results(groups, use_rrf=False, weights={"vector": 1.0, "graph": 0.2})
    assert merged[0].node_id == "a"


def test_merge_handles_empty_groups() -> None:
    assert merge_results({}) == []
    assert merge_results({"vector": []}) == []


def test_merge_keeps_longest_text() -> None:
    groups = {
        "vector": [make_context("n1", "short sentence only", 0.5, "vector")],
        "sentence_window": [
            make_context("n1", "a much longer passage with the surrounding window", 0.5, "sentence_window")
        ],
    }
    merged = merge_results(groups)
    assert len(merged) == 1
    assert "surrounding window" in merged[0].text


# --------------------------------------------------------------------------- #
# MultiRetriever
# --------------------------------------------------------------------------- #
class StubRetriever(BaseRetriever):
    """Retriever returning fixed contexts (or raising)."""

    def __init__(self, name: str, contexts: Sequence[RetrievedContext], error: Exception | None = None) -> None:
        super().__init__()
        self.name = name
        self._contexts = list(contexts)
        self._error = error
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        self.calls += 1
        if self._error:
            raise self._error
        return self._contexts[:top_k]


def test_multi_retriever_runs_all_retrievers(settings: Settings) -> None:
    first = StubRetriever("vector", [make_context("a", "alpha", 0.9, "vector")])
    second = StubRetriever("graph", [make_context("b", "beta", 0.8, "graph")])
    multi = MultiRetriever([first, second], settings=settings)
    result = multi.retrieve_detailed("query", top_k=5)

    assert first.calls == 1 and second.calls == 1
    assert result.per_retriever_counts == {"vector": 1, "graph": 1}
    assert {context.node_id for context in result.contexts} == {"a", "b"}


def test_multi_retriever_survives_failing_retriever(settings: Settings) -> None:
    good = StubRetriever("vector", [make_context("a", "alpha", 0.9, "vector")])
    bad = StubRetriever("graph", [], error=RetrievalError("boom"))
    result = MultiRetriever([good, bad], settings=settings).retrieve_detailed("query")

    assert [context.node_id for context in result.contexts] == ["a"]
    assert any("boom" in error for error in result.errors)


def test_multi_retriever_empty_query(settings: Settings) -> None:
    retriever = StubRetriever("vector", [make_context("a", "alpha", 0.9, "vector")])
    result = MultiRetriever([retriever], settings=settings).retrieve_detailed("   ")
    assert result.contexts == []
    assert retriever.calls == 0


def test_multi_retriever_no_retrievers(settings: Settings) -> None:
    result = MultiRetriever([], settings=settings).retrieve_detailed("query")
    assert result.errors == ["no retrievers configured"]


# --------------------------------------------------------------------------- #
# Vector retriever against a real FAISS index
# --------------------------------------------------------------------------- #
def test_vector_retriever_returns_relevant_chunk(settings: Settings, built_indexes) -> None:
    retriever = VectorRetriever(settings, index=built_indexes.vector_index)
    assert retriever.is_available()
    contexts = retriever.retrieve("annual leave days", top_k=3)
    assert contexts
    assert any("annual leave" in context.text.lower() for context in contexts)


def test_vector_retriever_missing_index_raises(settings: Settings) -> None:
    from src.indexing.vector_index import VectorIndex

    retriever = VectorRetriever(settings, index=VectorIndex(settings))
    assert retriever.is_available() is False
    with pytest.raises(RetrievalError):
        retriever.retrieve("anything")
