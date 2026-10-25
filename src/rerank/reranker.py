"""Cross-encoder re-ranking with batching and GPU/CPU auto-detection.

Default model: **BAAI/bge-reranker-v2-m3** (multilingual, Bangla + English).
The reranker takes the merged candidate list from the multi-retriever, scores
every ``(query, passage)`` pair, removes duplicates and returns ``top_n``
(default 5) contexts.

Backends:
    * ``cross_encoder`` – sentence-transformers ``CrossEncoder`` (default)
    * ``cohere``        – Cohere Rerank API (cloud alternative)
    * ``lexical``       – dependency-free IDF-weighted overlap (offline fallback)
    * ``none``          – keep the fusion order, no re-scoring

Example:
    >>> from src.models import RetrievedContext
    >>> reranker = Reranker(backend="lexical")
    >>> ctx = [RetrievedContext(node_id="1", text="annual leave is 20 days"),
    ...        RetrievedContext(node_id="2", text="office address in Dhaka")]
    >>> reranker.rerank("how many annual leave days?", ctx, top_n=1)[0].node_id
    '1'
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import Settings, get_settings
from src.errors import RerankError
from src.indexing.embeddings import detect_device
from src.models import RetrievedContext
from src.processing.metadata import STOPWORDS
from src.text_utils import normalise_whitespace, tokenize_words

# Minimum rerank score below which a query is treated as unanswerable.
# Passages from the same page whose token sets overlap this much are treated as
# duplicates (typical for overlapping sentence windows).
NEAR_DUPLICATE_JACCARD: float = 0.8

DEFAULT_MIN_SCORES: dict[str, float] = {
    "lexical": 0.10,
    "cross_encoder": 0.02,
    "cohere": 0.05,
    "none": 0.0,
}

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RerankResult:
    """Reranked contexts plus diagnostics."""

    contexts: list[RetrievedContext]
    backend: str
    candidates: int
    duration_ms: float = 0.0
    fallback_reason: str | None = None


class BaseReranker:
    """Interface for reranker backends."""

    name = "base"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Return a relevance score per passage."""
        raise NotImplementedError


class LexicalReranker(BaseReranker):
    """Dependency-free reranker based on IDF-weighted query-term coverage.

    Works for any language (tokenisation is Unicode aware) and is used when the
    cross-encoder cannot be loaded (no network/no GPU/no model cache).
    """

    name = "lexical"

    def __init__(self, phrase_bonus: float = 0.15) -> None:
        self.phrase_bonus = phrase_bonus

    @staticmethod
    def _idf(documents: Sequence[Sequence[str]], term: str) -> float:
        total = len(documents)
        containing = sum(1 for document in documents if term in document)
        return math.log((total + 1) / (containing + 0.5))

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score passages by IDF-weighted overlap and exact phrase matches."""
        if not texts:
            return []
        # Content words only: stopwords carry no signal and would let unrelated
        # passages pass the relevance gate.
        query_terms = [
            term for term in tokenize_words(query) if len(term) > 1 and term not in STOPWORDS
        ]
        if not query_terms:
            return [0.0 for _ in texts]

        document_terms = [set(tokenize_words(text)) for text in texts]
        idf = {term: self._idf(document_terms, term) for term in set(query_terms)}
        total_idf = sum(idf.get(term, 0.0) for term in set(query_terms)) or 1.0
        query_phrase = normalise_whitespace(query).lower()

        scores: list[float] = []
        for text, terms in zip(texts, document_terms, strict=True):
            matched = {term for term in set(query_terms) if term in terms}
            coverage = sum(idf.get(term, 0.0) for term in matched) / total_idf
            repetition = (
                sum(text.lower().count(term) for term in matched) / (len(terms) + 5)
                if terms
                else 0.0
            )
            phrase = 1.0 if query_phrase and query_phrase in normalise_whitespace(text).lower() else 0.0
            scores.append(round(coverage + 0.1 * repetition + self.phrase_bonus * phrase, 6))
        return scores


class CrossEncoderReranker(BaseReranker):
    """sentence-transformers cross-encoder (bge-reranker-v2-m3 by default)."""

    name = "cross_encoder"

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 512,
    ) -> None:
        """
        Args:
            model_name: Hugging Face cross-encoder id.
            device: ``auto``, ``cpu``, ``cuda`` or ``mps``.
            batch_size: Pairs scored per forward pass.
            max_length: Maximum tokenised pair length.
        """
        self.model_name = model_name
        self.device = detect_device(device)
        self.batch_size = max(1, int(batch_size))
        self.max_length = int(max_length)
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazily load the cross-encoder."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker %s on %s", self.model_name, self.device)
                kwargs: dict[str, Any] = {"device": self.device}
                try:
                    self._model = CrossEncoder(self.model_name, max_length=self.max_length, **kwargs)
                except TypeError:  # older sentence-transformers
                    self._model = CrossEncoder(self.model_name, **kwargs)
            except Exception as exc:
                raise RerankError(
                    f"cannot load cross-encoder {self.model_name!r}: {exc}. "
                    "Set RERANKER_BACKEND=lexical for an offline fallback."
                ) from exc
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Score ``(query, passage)`` pairs in batches."""
        if not texts:
            return []
        model = self.model
        pairs = [[query, text] for text in texts]
        try:
            raw = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except TypeError:  # pragma: no cover - API drift
            raw = model.predict(pairs)
        except Exception as exc:
            raise RerankError(f"cross-encoder scoring failed: {exc}") from exc

        scores: list[float] = []
        for value in raw:
            score = float(value)
            # bge-reranker emits logits; map to 0..1 for comparable reporting.
            if score < 0.0 or score > 1.0:
                score = 1.0 / (1.0 + math.exp(-score))
            scores.append(score)
        return scores


class CohereReranker(BaseReranker):
    """Cohere Rerank backend (cloud)."""

    name = "cohere"

    def __init__(self, api_key: str, model: str = "rerank-multilingual-v3.0", timeout: float = 30.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Call the Cohere rerank endpoint."""
        if not texts:
            return []
        if not self.api_key:
            raise RerankError("COHERE_API_KEY is not configured")
        try:
            import httpx

            response = httpx.post(
                "https://api.cohere.com/v2/rerank",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "query": query,
                    "documents": list(texts),
                    "top_n": len(texts),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise RerankError(f"Cohere rerank failed: {exc}") from exc
        scores = [0.0] * len(texts)
        for item in payload.get("results", []):
            index = int(item.get("index", 0))
            if 0 <= index < len(scores):
                scores[index] = float(item.get("relevance_score", 0.0))
        return scores


class Reranker:
    """Facade that scores, deduplicates and truncates candidate contexts."""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        settings: Settings | None = None,
        top_n: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        """
        Args:
            backend: ``cross_encoder``, ``cohere``, ``lexical`` or ``none``.
            model_name: Cross-encoder model id override.
            settings: Project settings.
            top_n: Default number of contexts to keep (5).
            batch_size: Batch size for cross-encoder inference.
        """
        self.settings = settings or get_settings()
        self.backend_name = (backend or self.settings.reranker_backend).lower()
        self.top_n = int(top_n or self.settings.rerank_top_n)
        self.fallback_reason: str | None = None
        self._lexical = LexicalReranker()
        self._backend: BaseReranker = self._create_backend(model_name, batch_size)

    def _create_backend(self, model_name: str | None, batch_size: int | None) -> BaseReranker:
        name = self.backend_name
        if name == "none":
            return BaseReranker()
        if name == "lexical":
            return self._lexical
        if name == "cohere":
            return CohereReranker(
                api_key=self.settings.cohere_api_key,
                model=self.settings.cohere_rerank_model,
            )
        try:
            return CrossEncoderReranker(
                model_name=model_name or self.settings.reranker_model,
                device=self.settings.reranker_device,
                batch_size=batch_size or self.settings.reranker_batch_size,
                max_length=self.settings.reranker_max_length,
            )
        except Exception as exc:  # pragma: no cover - constructor is cheap
            if not self.settings.allow_model_fallbacks:
                raise
            self.fallback_reason = str(exc)
            logger.warning("Reranker backend %s unavailable (%s); using lexical", name, exc)
            return self._lexical

    @property
    def default_min_score(self) -> float:
        """Relevance gate appropriate for the active backend."""
        return DEFAULT_MIN_SCORES.get(self.backend, 0.0)

    @property
    def backend(self) -> str:
        """Name of the active backend (may differ from the requested one)."""
        return self._backend.name

    @property
    def description(self) -> str:
        """Human readable description used by the UI."""
        if isinstance(self._backend, CrossEncoderReranker):
            return f"cross-encoder ({self._backend.model_name}, {self._backend.device})"
        if isinstance(self._backend, CohereReranker):
            return f"cohere ({self._backend.model})"
        if isinstance(self._backend, LexicalReranker):
            return "lexical IDF reranker (offline fallback)"
        return "no reranking (fusion order)"

    @staticmethod
    def _dedupe(
        contexts: Sequence[RetrievedContext],
        overlap_threshold: float = NEAR_DUPLICATE_JACCARD,
    ) -> list[RetrievedContext]:
        """Drop duplicate and near-duplicate passages.

        Exact duplicates are merged (provenance is combined).  Near-duplicates —
        overlapping sentence windows from the same page — are collapsed so the
        final top-5 does not consist of five almost identical passages.
        """
        kept: list[RetrievedContext] = []
        seen_exact: dict[str, RetrievedContext] = {}
        for context in sorted(contexts, key=lambda item: -float(item.score or 0.0)):
            key = context.dedupe_key
            if key in seen_exact:
                seen_exact[key].merge_from(context)
                continue
            tokens = set(tokenize_words(context.text))
            duplicate = False
            if tokens:
                for existing in kept:
                    if existing.source != context.source or existing.page != context.page:
                        continue
                    existing_tokens = set(tokenize_words(existing.text))
                    if not existing_tokens:
                        continue
                    jaccard = len(tokens & existing_tokens) / len(tokens | existing_tokens)
                    if jaccard >= overlap_threshold:
                        existing.merge_from(context)
                        duplicate = True
                        break
            if duplicate:
                continue
            seen_exact[key] = context
            kept.append(context)
        return kept

    def _score(self, query: str, contexts: Sequence[RetrievedContext]) -> list[float]:
        texts = [context.text for context in contexts]
        try:
            scores = self._backend.score(query, texts)
        except Exception as exc:
            if not self.settings.allow_model_fallbacks or self._backend is self._lexical:
                raise RerankError(str(exc)) from exc
            self.fallback_reason = str(exc)
            logger.warning("Reranker scoring failed (%s); falling back to lexical", exc)
            self._backend = self._lexical
            scores = self._lexical.score(query, texts)
        return scores

    def rerank(
        self,
        query: str,
        contexts: Sequence[RetrievedContext],
        top_n: int | None = None,
    ) -> list[RetrievedContext]:
        """Re-score and truncate ``contexts``.

        Args:
            query: User query.
            contexts: Merged candidate contexts.
            top_n: Number of contexts to keep (default 5).

        Returns:
            Contexts sorted by rerank score, each with ``rerank_score`` set.
        """
        limit = int(top_n or self.top_n)
        candidates = self._dedupe([context for context in contexts if (context.text or "").strip()])
        if not candidates:
            return []

        if isinstance(self._backend, BaseReranker) and self.backend_name == "none":
            for rank, context in enumerate(candidates[:limit]):
                context.rerank_score = float(context.score)
                context.rank = rank
            return candidates[:limit]

        scores = self._score(query, candidates)
        for context, score in zip(candidates, scores, strict=True):
            context.rerank_score = float(score)

        ranked = sorted(
            candidates,
            key=lambda item: (-(item.rerank_score or 0.0), -float(item.score), item.node_id),
        )
        for rank, context in enumerate(ranked):
            context.rank = rank
        logger.info(
            "Reranked %s candidates with %s; keeping %s",
            len(candidates),
            self.description,
            min(limit, len(ranked)),
        )
        return ranked[:limit]

    def rerank_with_details(
        self,
        query: str,
        contexts: Sequence[RetrievedContext],
        top_n: int | None = None,
    ) -> RerankResult:
        """Rerank and return diagnostics alongside the contexts."""
        import time

        started = time.perf_counter()
        ranked = self.rerank(query, contexts, top_n=top_n)
        return RerankResult(
            contexts=ranked,
            backend=self.description,
            candidates=len(contexts),
            duration_ms=(time.perf_counter() - started) * 1000,
            fallback_reason=self.fallback_reason,
        )


__all__ = [
    "Reranker",
    "RerankResult",
    "BaseReranker",
    "LexicalReranker",
    "CrossEncoderReranker",
    "CohereReranker",
]
