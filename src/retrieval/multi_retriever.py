"""Parallel multi-retriever with merge, dedupe and rank fusion.

The three retrievers (vector, sentence-window, graph) run concurrently in a
thread pool.  Their results are merged with:

* **dedupe** by node id *and* by normalised text (window results can return the
  same passage with a different node id);
* **Reciprocal Rank Fusion (RRF)** — ``sum(weight_r / (k + rank_r))`` — which is
  scale-free and therefore works even though the three retrievers produce
  incomparable scores (cosine, graph weights, ...);
* **provenance merge** so the UI can show *which* retrievers found a passage.

Example:
    >>> from src.models import RetrievedContext
    >>> a = [RetrievedContext(node_id="n1", text="alpha", score=0.9, retriever="vector")]
    >>> b = [RetrievedContext(node_id="n1", text="alpha", score=0.4, retriever="graph")]
    >>> merged = merge_results({"vector": a, "graph": b})
    >>> len(merged), merged[0].retrievers
    (1, ['vector', 'graph'])
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.models import RetrievedContext
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


def _normalise_scores(contexts: Sequence[RetrievedContext]) -> list[float]:
    """Min-max normalise scores within one retriever's result list."""
    if not contexts:
        return []
    values = [float(context.score) for context in contexts]
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def merge_results(
    result_groups: Mapping[str, Sequence[RetrievedContext]],
    top_k: int = 10,
    weights: Mapping[str, float] | None = None,
    rrf_k: int = 60,
    use_rrf: bool = True,
) -> list[RetrievedContext]:
    """Merge and deduplicate retrievers' outputs into a single ranked list.

    Args:
        result_groups: Mapping ``retriever_name -> ranked contexts``.
        top_k: Maximum number of contexts to return.
        weights: Optional per-retriever weights (defaults to 1.0 each).
        rrf_k: RRF smoothing constant (standard value: 60).
        use_rrf: Use reciprocal rank fusion; otherwise weighted normalised scores.

    Returns:
        Deduplicated contexts sorted by fused score (best first), with
        ``retrievers`` listing every retriever that returned them.
    """
    weights = dict(weights or {})
    merged: dict[str, RetrievedContext] = {}
    by_text: dict[str, str] = {}
    fused: dict[str, float] = {}
    best_raw: dict[str, float] = {}

    for retriever_name, contexts in result_groups.items():
        if not contexts:
            continue
        weight = float(weights.get(retriever_name, 1.0))
        normalised = _normalise_scores(contexts)
        for rank, context in enumerate(contexts):
            key = context.node_id or context.dedupe_key
            text_key = context.dedupe_key
            # Deduplicate by node id first, then by identical text.
            if key not in merged and text_key in by_text:
                key = by_text[text_key]

            contribution = (
                weight / (rrf_k + rank + 1) if use_rrf else weight * normalised[rank]
            )

            if key in merged:
                existing = merged[key]
                raw_score = context.score
                existing.merge_from(context)
                existing.metadata.setdefault("retrieval_score", context.score)
                # Keep the longest text (windows carry more context).
                if len(context.text or "") > len(existing.text or ""):
                    existing.text = context.text
            else:
                clone = RetrievedContext(
                    node_id=context.node_id or key,
                    text=context.text,
                    score=context.score,
                    source=context.source,
                    page=context.page,
                    source_type=context.source_type,
                    retriever=context.retriever or retriever_name,
                    retrievers=list(context.retrievers) or [retriever_name],
                    rank=rank,
                    rerank_score=context.rerank_score,
                    metadata={**context.metadata, "retrieval_score": context.score},
                )
                merged[key] = clone
                by_text.setdefault(text_key, key)
                raw_score = context.score

            by_text.setdefault(text_key, key)
            fused[key] = fused.get(key, 0.0) + contribution
            best_raw[key] = max(best_raw.get(key, float("-inf")), float(raw_score or 0.0))

    ranked_keys = sorted(
        merged,
        key=lambda key: (-fused.get(key, 0.0), -best_raw.get(key, 0.0), key),
    )
    results: list[RetrievedContext] = []
    for position, key in enumerate(ranked_keys[: max(1, top_k)]):
        context = merged[key]
        context.score = round(fused.get(key, 0.0), 6)
        context.metadata["fusion_score"] = context.score
        context.metadata["max_retrieval_score"] = best_raw.get(key, 0.0)
        context.rank = position
        results.append(context)
    return results


@dataclass(slots=True)
class MultiRetrievalResult:
    """Detailed multi-retriever output (contexts + diagnostics)."""

    contexts: list[RetrievedContext] = field(default_factory=list)
    per_retriever_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    total_found: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "contexts": [context.to_dict() for context in self.contexts],
            "per_retriever_counts": self.per_retriever_counts,
            "errors": self.errors,
            "timings_ms": self.timings_ms,
            "total_found": self.total_found,
        }


class MultiRetriever:
    """Run every retriever in parallel and fuse the results."""

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        settings: Settings | None = None,
        weights: Mapping[str, float] | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        """
        Args:
            retrievers: Retriever instances to run.
            settings: Project settings.
            weights: Optional per-retriever fusion weights.
            timeout_seconds: Per-retriever timeout (defaults to settings).
        """
        self.settings = settings or get_settings()
        self.retrievers = list(retrievers)
        self.weights = dict(weights or {retriever.name: retriever.weight for retriever in self.retrievers})
        self.timeout_seconds = float(
            timeout_seconds if timeout_seconds is not None else self.settings.retriever_timeout_seconds
        )

    @property
    def available_retrievers(self) -> list[BaseRetriever]:
        """Retrievers whose index is present on disk."""
        return [retriever for retriever in self.retrievers if retriever.is_available()]

    def retrieve_detailed(
        self,
        query: str,
        top_k: int | None = None,
        per_retriever_k: int | None = None,
    ) -> MultiRetrievalResult:
        """Run all retrievers concurrently and return fused contexts + diagnostics.

        Args:
            query: User query.
            top_k: Number of fused contexts to return (default ``RETRIEVAL_TOP_K``).
            per_retriever_k: Candidates requested from each retriever
                (default ``RETRIEVAL_TOP_K``).

        Returns:
            A :class:`MultiRetrievalResult`; failures of individual retrievers are
            reported in ``errors`` instead of aborting the query.
        """
        if not query or not query.strip():
            return MultiRetrievalResult()

        top_k = int(top_k or self.settings.retrieval_top_k)
        per_retriever_k = int(per_retriever_k or self.settings.retrieval_top_k)
        result = MultiRetrievalResult()
        if not self.retrievers:
            result.errors.append("no retrievers configured")
            return result

        groups: dict[str, list[RetrievedContext]] = {}
        started = time.perf_counter()
        workers = max(1, min(len(self.retrievers), 8))

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="retriever") as pool:
            futures: dict[Future[list[RetrievedContext]], tuple[str, float]] = {}
            for retriever in self.retrievers:
                futures[pool.submit(retriever.retrieve, query, per_retriever_k)] = (
                    retriever.name,
                    time.perf_counter(),
                )
            for future in as_completed(futures, timeout=self.timeout_seconds + 5):
                name, submitted = futures[future]
                try:
                    contexts = future.result(timeout=self.timeout_seconds)
                    groups[name] = contexts or []
                    result.timings_ms[name] = round((time.perf_counter() - submitted) * 1000, 2)
                    result.per_retriever_counts[name] = len(groups[name])
                except Exception as exc:
                    logger.warning("Retriever %s failed: %s", name, exc)
                    result.errors.append(f"{name}: {exc}")
                    result.per_retriever_counts[name] = 0
                    result.timings_ms[name] = round((time.perf_counter() - submitted) * 1000, 2)

        result.total_found = sum(len(contexts) for contexts in groups.values())
        merged = merge_results(
            groups,
            top_k=top_k,
            weights=self.weights,
            rrf_k=self.settings.rrf_k,
            use_rrf=self.settings.use_rrf,
        )
        result.contexts = merged
        result.timings_ms["fusion"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Multi-retrieval: %s contexts from %s retrievers (%s unique after fusion) in %.0f ms",
            result.total_found,
            len(groups),
            len(merged),
            result.timings_ms["fusion"],
        )
        return result

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        per_retriever_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Return fused contexts only (convenience wrapper)."""
        return self.retrieve_detailed(query, top_k=top_k, per_retriever_k=per_retriever_k).contexts


__all__ = ["MultiRetriever", "MultiRetrievalResult", "merge_results"]
