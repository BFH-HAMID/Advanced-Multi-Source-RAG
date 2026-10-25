"""Tests for prompt construction, LLM plumbing, citations and the pipeline."""

from __future__ import annotations

import json

import pytest

from src.config import Settings
from src.generation.formatter import (
    extract_citations,
    format_response,
    group_contexts_by_source,
    render_markdown,
    resolve_citations,
)
from src.generation.llm import (
    ExtractiveBackend,
    LLMClient,
    OpenAIBackend,
    get_llm_client,
    parse_json_object,
)
from src.generation.query_builder import (
    NOT_FOUND_EN,
    build_prompt,
    extractive_answer,
    is_refusal,
)
from src.models import RetrievedContext


def context(node_id: str, text: str, source: str = "handbook.pdf", page: int | None = 4) -> RetrievedContext:
    """Helper to build a retrieved context."""
    return RetrievedContext(
        node_id=node_id,
        text=text,
        score=0.9,
        source=source,
        page=page,
        retriever="vector",
        retrievers=["vector", "graph"],
        rerank_score=0.8,
    )


# --------------------------------------------------------------------------- #
# Query construction
# --------------------------------------------------------------------------- #
def test_build_prompt_contains_context_and_citation_labels(settings: Settings) -> None:
    contexts = [context("c1", "Annual leave is 20 days."), context("c2", "Sick leave is 10 days.", page=5)]
    prompt = build_prompt("How much leave do I get?", contexts, settings=settings)

    assert "[handbook.pdf, p.4]" in prompt.user
    assert "[handbook.pdf, p.5]" in prompt.user
    assert "How much leave do I get?" in prompt.user
    assert NOT_FOUND_EN in prompt.system
    assert "ONLY" in prompt.system.upper()
    assert prompt.context_tokens > 0
    assert prompt.total_tokens > 0


def test_build_prompt_respects_token_budget(settings: Settings) -> None:
    contexts = [context(f"c{i}", "A long policy sentence. " * 60, page=i + 1) for i in range(6)]
    prompt = build_prompt("leave policy", contexts, max_context_tokens=150, settings=settings)
    assert len(prompt.contexts) < len(contexts)
    assert prompt.dropped_contexts > 0
    assert prompt.context_tokens <= 150 * 2


def test_build_prompt_without_context(settings: Settings) -> None:
    prompt = build_prompt("nothing here", [], settings=settings)
    assert "(no context available)" in prompt.user
    assert prompt.contexts == []


def test_build_prompt_adds_bangla_hint(settings: Settings) -> None:
    prompt = build_prompt("বার্ষিক ছুটি কত দিন?", [context("c1", "২০ দিন।")], settings=settings)
    assert "Bangla" in prompt.system


def test_is_refusal_detects_messages() -> None:
    assert is_refusal(NOT_FOUND_EN)
    assert is_refusal("প্রদত্ত নথিতে এই প্রশ্নের উত্তর খুঁজে পাওয়া যায়নি।")
    assert not is_refusal("Annual leave is 20 days [handbook.pdf, p.4].")


def test_extractive_answer_cites_sources() -> None:
    contexts = [
        context("c1", "Annual leave is 20 days per calendar year. Sick leave is 10 days.", page=2),
        context("c2", "Remote work is allowed 3 days per week.", page=3),
    ]
    answer = extractive_answer("How many annual leave days?", contexts)
    assert "20 days" in answer
    assert "[handbook.pdf, p.2]" in answer


def test_extractive_answer_refuses_unrelated_question() -> None:
    contexts = [context("c1", "Annual leave is 20 days per calendar year.")]
    assert extractive_answer("What is the capital of France?", contexts) == NOT_FOUND_EN


# --------------------------------------------------------------------------- #
# LLM plumbing
# --------------------------------------------------------------------------- #
def test_parse_json_object_handles_fences_and_prose() -> None:
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"triples": [{"subject": "A"}]}') == {"triples": [{"subject": "A"}]}
    assert parse_json_object("{'a': 1,}") == {"a": 1}
    assert parse_json_object("no json here") == {}


def test_extractive_backend_uses_contexts() -> None:
    backend = ExtractiveBackend()
    contexts = [context("c1", "Learning budget is BDT 50,000 per employee.")]
    answer = backend.complete("prompt", query="What is the learning budget?", contexts=contexts)
    assert "50,000" in answer


def test_llm_client_falls_back_without_credentials(settings: Settings, clean_env) -> None:
    settings = settings.model_copy(update={"llm_provider": "openai", "openai_api_key": "", "llm_max_retries": 0})
    client = get_llm_client(settings)
    assert client.provider == "openai"
    contexts = [context("c1", "Annual leave is 20 days.")]
    answer = client.complete("prompt", query="annual leave?", contexts=contexts)
    assert "20 days" in answer
    assert client.used_fallback is True
    assert client.provider == "extractive"


def test_llm_client_reports_errors_when_fallback_disabled(settings: Settings) -> None:
    settings = settings.model_copy(update={"allow_model_fallbacks": False})
    backend = OpenAIBackend(model="gpt-4o-mini", api_key="")
    client = LLMClient(backend=backend, fallback=None, max_retries=0, allow_fallback=False)
    from src.errors import GenerationError

    with pytest.raises(GenerationError):
        client.complete("prompt")


def test_llm_client_streams_single_chunk_for_extractive(settings: Settings) -> None:
    client = get_llm_client(settings)
    contexts = [context("c1", "Annual leave is 20 days.")]
    chunks = list(client.stream_complete("prompt", query="annual leave?", contexts=contexts))
    assert chunks and "20 days" in "".join(chunks)


def test_ollama_and_anthropic_backends_registered(settings: Settings) -> None:
    from src.generation.llm import build_backend

    assert build_backend(settings.model_copy(update={"llm_provider": "ollama"})).provider == "ollama"
    assert build_backend(settings.model_copy(update={"llm_provider": "anthropic"})).provider == "anthropic"


# --------------------------------------------------------------------------- #
# Citation handling
# --------------------------------------------------------------------------- #
def test_extract_citations_variants() -> None:
    text = "Leave is 20 days [handbook.pdf, p.4]. Remote work [handbook.pdf, page 7]. Url source [https://x.example/a]."
    citations = extract_citations(text)
    assert ("handbook.pdf", 4) in citations
    assert ("handbook.pdf", 7) in citations
    assert any(source.startswith("https://") for source, _ in citations)


def test_resolve_citations_matches_contexts() -> None:
    contexts = [context("c1", "Annual leave is 20 days.", page=4), context("c2", "Sick leave.", page=5)]
    citations = resolve_citations("You get 20 days [handbook.pdf, p.4].", contexts)
    assert len(citations) == 1
    assert citations[0].key == "handbook.pdf, p.4"
    assert citations[0].snippet


def test_resolve_citations_falls_back_to_contexts() -> None:
    contexts = [context("c1", "Annual leave is 20 days.", page=4)]
    citations = resolve_citations("You get 20 days.", contexts)
    assert citations and citations[0].key == "handbook.pdf, p.4"


def test_format_response_and_markdown() -> None:
    contexts = [context("c1", "Annual leave is 20 days.", page=4)]
    answer = format_response("leave?", "20 days [handbook.pdf, p.4].", contexts, {"total_ms": 12.0})
    assert answer.grounded is True
    assert answer.citation_keys == ["handbook.pdf, p.4"]
    markdown = render_markdown(answer)
    assert "**Citations**" in markdown
    assert "**Sources**" in markdown


def test_group_contexts_by_source() -> None:
    grouped = group_contexts_by_source([context("c1", "a"), context("c2", "b", source="other.csv", page=None)])
    assert set(grouped) == {"handbook.pdf", "other.csv"}


# --------------------------------------------------------------------------- #
# End-to-end pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_answers_from_index(pipeline) -> None:
    answer = pipeline.answer("How many days of annual leave do employees get?")
    assert "20" in answer.answer
    assert answer.grounded is True
    assert answer.citations
    assert "leave_policy.txt" in " ".join(answer.citation_keys)
    assert answer.timings_ms["total_ms"] >= 0


def test_pipeline_refuses_unanswerable_question(pipeline) -> None:
    answer = pipeline.answer("What is the capital of France?")
    assert answer.grounded is False
    assert is_refusal(answer.answer)
    assert answer.citations == []


def test_pipeline_handles_bangla_question(pipeline) -> None:
    answer = pipeline.answer("বার্ষিক ছুটি কত দিন?")
    assert "২০" in answer.answer or "20" in answer.answer
    assert answer.citations
    assert "hr_bangla.txt" in " ".join(answer.citation_keys)


def test_pipeline_streaming(pipeline) -> None:
    prepared = pipeline.prepare("How many days of annual leave do employees get?")
    assert prepared.has_context
    chunks = list(pipeline.stream(prepared))
    text = "".join(chunks)
    assert text.strip()
    answer = pipeline.finalize(prepared, text, generation_ms=5.0)
    assert answer.answer == text.strip()
    assert answer.citations


def test_pipeline_reports_diagnostics(pipeline) -> None:
    description = pipeline.describe()
    assert description["retrievers"] == ["vector", "sentence_window", "graph"]
    assert description["rerank_top_n"] == 3
    answer = pipeline.answer("remote work policy")
    assert answer.retriever_stats
    assert "retrieval_ms" in answer.timings_ms
    assert json.dumps(answer.to_dict())  # serialisable
