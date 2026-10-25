"""Tests for the cross-encoder / lexical reranker."""

from __future__ import annotations

from src.config import Settings
from src.models import RetrievedContext
from src.rerank.reranker import LexicalReranker, Reranker


def context(node_id: str, text: str, score: float = 0.5, page: int = 1) -> RetrievedContext:
    """Helper to build a retrieved context."""
    return RetrievedContext(
        node_id=node_id,
        text=text,
        score=score,
        source="policy.pdf",
        page=page,
        retriever="vector",
        retrievers=["vector"],
    )


def test_lexical_reranker_prefers_relevant_passage() -> None:
    contexts = [
        context("a", "The office cafeteria serves lunch between noon and two."),
        context("b", "Every employee is entitled to 20 days of annual leave per year."),
        context("c", "Laptops must be encrypted and passwords rotated every 180 days."),
    ]
    scores = LexicalReranker().score("how many annual leave days?", [c.text for c in contexts])
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_reranker_returns_top_n_sorted(settings: Settings) -> None:
    contexts = [
        context("a", "Office cafeteria information."),
        context("b", "Annual leave is 20 days per year."),
        context("c", "Annual leave requests need 14 days notice."),
    ]
    reranker = Reranker(backend="lexical", settings=settings, top_n=2)
    ranked = reranker.rerank("annual leave", contexts)
    assert len(ranked) == 2
    assert ranked[0].rerank_score >= ranked[1].rerank_score
    assert all("leave" in item.text for item in ranked)


def test_reranker_supports_bangla(settings: Settings) -> None:
    contexts = [
        context("a", "অফিসে দুপুরের খাবারের ব্যবস্থা আছে।"),
        context("b", "প্রতি কর্মীর জন্য বছরে ৫০,০০০ টাকার শিক্ষা বাজেট রয়েছে।"),
    ]
    ranked = Reranker(backend="lexical", settings=settings, top_n=1).rerank("শিক্ষা বাজেট কত?", contexts)
    assert ranked[0].node_id == "b"


def test_reranker_deduplicates_near_identical_passages(settings: Settings) -> None:
    base = "Annual leave is 20 days per year. Requests need 14 days notice. Sick leave is 10 days."
    contexts = [
        context("a", base),
        context("b", base + " Extra trailing sentence."),
        context("c", "The security team handles incident response."),
    ]
    ranked = Reranker(backend="lexical", settings=settings, top_n=3).rerank("annual leave", contexts)
    assert len(ranked) == 2
    assert {item.node_id for item in ranked} == {"a", "c"}


def test_reranker_none_backend_keeps_order(settings: Settings) -> None:
    contexts = [context("a", "first passage", score=0.9), context("b", "second passage", score=0.1)]
    ranked = Reranker(backend="none", settings=settings, top_n=5).rerank("query", contexts)
    assert [item.node_id for item in ranked] == ["a", "b"]


def test_reranker_handles_empty_input(settings: Settings) -> None:
    assert Reranker(backend="lexical", settings=settings).rerank("query", []) == []


def test_reranker_default_relevance_gate(settings: Settings) -> None:
    reranker = Reranker(backend="lexical", settings=settings)
    assert reranker.default_min_score > 0
    assert reranker.backend == "lexical"


def test_reranker_details(settings: Settings) -> None:
    contexts = [context("a", "Annual leave is 20 days.")]
    result = Reranker(backend="lexical", settings=settings).rerank_with_details("annual leave", contexts)
    assert result.candidates == 1
    assert result.contexts
    assert result.backend
