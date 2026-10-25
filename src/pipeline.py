"""End-to-end RAG orchestrator (steps 6 → 10).

``RAGPipeline.answer()`` runs:

1. **multi-retrieval** – vector + sentence-window + graph in parallel;
2. **re-ranking** – cross-encoder (or fallback) down to ``RERANK_TOP_N``;
3. **query construction** – strict, citation-oriented prompt;
4. **LLM generation** – Claude / OpenAI / Ollama, grounded in context;
5. **formatting** – answer + resolved citations + sources + diagnostics.

``RAGPipeline.prepare()`` + ``RAGPipeline.stream()`` split the flow so the
Streamlit UI can show sources first and stream the answer afterwards.

Example:
    >>> pipeline = build_pipeline()                 # doctest: +SKIP
    >>> result = pipeline.answer("What is the leave policy?")  # doctest: +SKIP
    >>> result.answer                                # doctest: +SKIP
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from src.config import Settings, get_settings
from src.errors import RetrievalError
from src.generation.formatter import format_response
from src.generation.llm import LLMClient, get_llm_client
from src.generation.query_builder import (
    NOT_FOUND_BN,
    NOT_FOUND_EN,
    PromptBundle,
    build_prompt,
)
from src.models import RAGAnswer
from src.rerank.reranker import Reranker
from src.retrieval import MultiRetrievalResult, MultiRetriever, build_multi_retriever
from src.storage.document_db import DocumentDB
from src.text_utils import looks_bangla

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedQuery:
    """Retrieval + prompt state for one query (before generation)."""

    query: str
    retrieval: MultiRetrievalResult = field(default_factory=MultiRetrievalResult)
    contexts: list[Any] = field(default_factory=list)
    prompt: PromptBundle | None = None
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    best_rerank_score: float = 0.0
    gate_threshold: float = 0.0
    gated: bool = False

    @property
    def has_context(self) -> bool:
        """Whether usable (non-gated) context is available."""
        return bool(self.contexts) and not self.gated


class RAGPipeline:
    """Coordinates retrieval, reranking, prompting, generation and formatting."""

    def __init__(
        self,
        settings: Settings | None = None,
        db: DocumentDB | None = None,
        multi_retriever: MultiRetriever | None = None,
        reranker: Reranker | None = None,
        llm: LLMClient | None = None,
        embed_model: Any | None = None,
        top_k: int | None = None,
        rerank_top_n: int | None = None,
    ) -> None:
        """
        Args:
            settings: Project settings.
            db: Document database (used to hydrate graph results).
            multi_retriever: Custom multi-retriever (built automatically otherwise).
            reranker: Custom reranker.
            llm: Custom LLM client.
            embed_model: Shared embedding model.
            top_k: Number of fused candidates before reranking.
            rerank_top_n: Number of contexts kept after reranking (default 5).
        """
        self.settings = settings or get_settings()
        self.db = db or DocumentDB(settings=self.settings)
        self.top_k = int(top_k or self.settings.retrieval_top_k)
        self.rerank_top_n = int(rerank_top_n or self.settings.rerank_top_n)
        self.warnings: list[str] = []

        if multi_retriever is None:
            multi_retriever, warnings = build_multi_retriever(
                self.settings, embed_model=embed_model, db=self.db
            )
            self.warnings.extend(warnings)
        self.multi_retriever = multi_retriever
        self.reranker = reranker or Reranker(settings=self.settings, top_n=self.rerank_top_n)
        self.llm = llm or get_llm_client(self.settings)

    # ------------------------------------------------------------------ info
    def describe(self) -> dict[str, Any]:
        """Return a description of the active components (for the UI sidebar)."""
        return {
            "llm_provider": self.llm.provider,
            "llm_model": self.llm.model,
            "llm_available": self.llm.is_available,
            "reranker": self.reranker.description,
            "retrievers": [retriever.name for retriever in self.multi_retriever.retrievers],
            "top_k": self.top_k,
            "rerank_top_n": self.rerank_top_n,
            "warnings": self.warnings,
        }

    # ------------------------------------------------------------- retrieval
    def prepare(self, query: str) -> PreparedQuery:
        """Run retrieval, reranking and prompt construction for ``query``."""
        started = time.perf_counter()
        prepared = PreparedQuery(query=query.strip())
        if not prepared.query:
            prepared.errors.append("empty query")
            return prepared

        try:
            prepared.retrieval = self.multi_retriever.retrieve_detailed(
                prepared.query, top_k=self.top_k
            )
            prepared.errors.extend(prepared.retrieval.errors)
        except RetrievalError as exc:
            prepared.errors.append(str(exc))
            logger.error("Retrieval failed: %s", exc)
        retrieval_ms = (time.perf_counter() - started) * 1000
        prepared.timings_ms["retrieval_ms"] = round(retrieval_ms, 2)
        prepared.timings_ms.update(
            {f"retriever.{name}": value for name, value in prepared.retrieval.timings_ms.items()}
        )

        rerank_started = time.perf_counter()
        try:
            prepared.contexts = self.reranker.rerank(
                prepared.query, prepared.retrieval.contexts, top_n=self.rerank_top_n
            )
        except Exception as exc:
            prepared.errors.append(f"rerank: {exc}")
            logger.error("Reranking failed: %s", exc)
            prepared.contexts = list(prepared.retrieval.contexts)[: self.rerank_top_n]
        prepared.timings_ms["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)

        # Relevance gate: refuse instead of answering from unrelated passages.
        threshold = float(self.settings.min_rerank_score or self.reranker.default_min_score)
        best = max((float(context.rerank_score or 0.0) for context in prepared.contexts), default=0.0)
        prepared.best_rerank_score = best
        prepared.gate_threshold = threshold
        if prepared.contexts and threshold > 0 and best < threshold:
            prepared.gated = True
            prepared.errors.append(
                f"relevance gate: best rerank score {best:.3f} < {threshold:.3f}; answering with a refusal"
            )
            prepared.contexts = []
            logger.info("Relevance gate rejected all contexts (best=%.3f, threshold=%.3f)", best, threshold)

        prompt_started = time.perf_counter()
        prepared.prompt = build_prompt(
            prepared.query,
            prepared.contexts,
            max_context_tokens=self.settings.max_context_tokens,
            settings=self.settings,
        )
        prepared.timings_ms["prompt_ms"] = round((time.perf_counter() - prompt_started) * 1000, 2)
        prepared.timings_ms["total_retrieval_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return prepared

    # ----------------------------------------------------------- generation
    @staticmethod
    def _refusal(query: str) -> str:
        return NOT_FOUND_BN if looks_bangla(query) else NOT_FOUND_EN

    def answer(self, query: str) -> RAGAnswer:
        """Answer ``query`` end to end.

        Args:
            query: User question (Bangla or English).

        Returns:
            A :class:`RAGAnswer` with citations and diagnostics.  When nothing
            relevant was retrieved the answer states that the information is not
            in the documents — the LLM is not called in that case.
        """
        started = time.perf_counter()
        prepared = self.prepare(query)
        errors = list(prepared.errors)

        if not prepared.has_context:
            message = self._refusal(prepared.query or query)
            prepared.timings_ms["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            return format_response(
                query=query,
                answer=message,
                contexts=[],
                timings_ms=prepared.timings_ms,
                retriever_stats=prepared.retrieval.per_retriever_counts,
                errors=errors,
                provider=self.llm.provider,
                model=self.llm.model,
            )

        assert prepared.prompt is not None
        generation_started = time.perf_counter()
        try:
            answer_text = self.llm.complete(
                prepared.prompt.user,
                system=prepared.prompt.system,
                query=prepared.query,
                contexts=prepared.prompt.contexts,
            )
        except Exception as exc:
            errors.append(f"generation: {exc}")
            logger.exception("Generation failed; returning refusal message")
            answer_text = self._refusal(prepared.query)
        prepared.timings_ms["generation_ms"] = round(
            (time.perf_counter() - generation_started) * 1000, 2
        )
        prepared.timings_ms["total_ms"] = round((time.perf_counter() - started) * 1000, 2)

        if self.llm.used_fallback and self.llm.last_error:
            errors.append(f"llm fallback: {self.llm.last_error}")

        return format_response(
            query=query,
            answer=answer_text,
            contexts=prepared.prompt.contexts,
            timings_ms=prepared.timings_ms,
            retriever_stats=prepared.retrieval.per_retriever_counts,
            errors=errors,
            provider=self.llm.provider,
            model=self.llm.model,
        )

    def stream(self, prepared: PreparedQuery) -> Iterator[str]:
        """Stream the answer for an already prepared query."""
        if not prepared.has_context or prepared.prompt is None:
            yield self._refusal(prepared.query)
            return
        yield from self.llm.stream_complete(
            prepared.prompt.user,
            system=prepared.prompt.system,
            query=prepared.query,
            contexts=prepared.prompt.contexts,
        )

    def finalize(
        self,
        prepared: PreparedQuery,
        answer_text: str,
        generation_ms: float = 0.0,
    ) -> RAGAnswer:
        """Format a (possibly streamed) answer into a :class:`RAGAnswer`."""
        timings = dict(prepared.timings_ms)
        if generation_ms:
            timings["generation_ms"] = round(generation_ms, 2)
        errors = list(prepared.errors)
        if self.llm.used_fallback and self.llm.last_error:
            errors.append(f"llm fallback: {self.llm.last_error}")
        return format_response(
            query=prepared.query,
            answer=answer_text,
            contexts=prepared.prompt.contexts if prepared.prompt else [],
            timings_ms=timings,
            retriever_stats=prepared.retrieval.per_retriever_counts,
            errors=errors,
            provider=self.llm.provider,
            model=self.llm.model,
        )


def build_pipeline(
    settings: Settings | None = None,
    db: DocumentDB | None = None,
    llm: LLMClient | None = None,
) -> RAGPipeline:
    """Create a ready-to-use pipeline (loads indexes lazily on first query)."""
    settings = settings or get_settings()
    return RAGPipeline(settings=settings, db=db, llm=llm)


__all__ = ["RAGPipeline", "PreparedQuery", "build_pipeline"]
