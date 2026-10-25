"""Response formatting: citation extraction, resolution and rendering.

The LLM is instructed to cite ``[file_name, page]``.  This module:

1. parses such labels out of the answer;
2. resolves them against the retrieved contexts (source + page);
3. falls back to the top contexts when the model forgot to cite;
4. renders markdown / bullet citations for the UI and CLI.

Example:
    >>> from src.models import RetrievedContext
    >>> context = RetrievedContext(node_id="1", text="Leave is 20 days.", source="hr.pdf", page=4)
    >>> answer = format_response("How much leave?", "You get 20 days [hr.pdf, p.4].", [context])
    >>> answer.citations[0].key
    'hr.pdf, p.4'
"""

from __future__ import annotations

import logging
import re
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.generation.query_builder import is_refusal
from src.models import Citation, RAGAnswer, RetrievedContext

logger = logging.getLogger(__name__)

# matches [file.pdf, p.4] / [file.pdf, page 4] / [file.pdf] / [https://x/y]
_CITATION_RE = re.compile(
    r"\[([^\[\]\n]{2,300}?)\s*(?:,\s*(?:p\.?|pp\.?|page|পৃষ্ঠা)\s*(\d+)\s*)?\]",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[।.!?])\s+")
_MAX_SNIPPET = 240


def _normalise_source(name: str) -> str:
    """Normalise a source label for comparison (basename, lower case)."""
    cleaned = (name or "").strip().strip("[]").strip()
    cleaned = cleaned.split("?")[0].rstrip("/")
    return Path(cleaned).name.lower() if "://" not in cleaned else cleaned.lower()


def extract_citations(answer: str) -> list[tuple[str, int | None]]:
    """Extract ``(source, page)`` pairs from an answer's citation labels.

    Args:
        answer: Model answer containing ``[file, page]`` labels.

    Returns:
        Citation pairs in first-seen order (duplicates removed).
    """
    found: OrderedDict[tuple[str, int | None], None] = OrderedDict()
    for match in _CITATION_RE.finditer(answer or ""):
        source = match.group(1).strip()
        page = int(match.group(2)) if match.group(2) else None
        if not source or source.lower() in {"citation", "source", "sources"}:
            continue
        found[(source, page)] = None
    return list(found)


def _match_context(
    source: str,
    page: int | None,
    contexts: Sequence[RetrievedContext],
) -> RetrievedContext | None:
    """Find the context that best matches a parsed citation."""
    target = _normalise_source(source)
    if not target:
        return None
    by_source = [c for c in contexts if _normalise_source(c.source) == target]
    if not by_source:
        # tolerate partial labels such as "handbook" for "handbook.pdf"
        by_source = [
            c for c in contexts if target and target in _normalise_source(c.source)
        ]
    if not by_source:
        return None
    if page is not None:
        for context in by_source:
            if context.page == page:
                return context
    return by_source[0]


def _snippet(text: str, limit: int = _MAX_SNIPPET) -> str:
    cleaned = " ".join((text or "").split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1].rstrip() + "…"


def resolve_citations(
    answer: str,
    contexts: Sequence[RetrievedContext],
    fallback_count: int = 3,
) -> list[Citation]:
    """Resolve citation labels in ``answer`` against retrieved contexts.

    Args:
        answer: Model answer.
        contexts: Contexts given to the model (reranked, best first).
        fallback_count: Number of contexts to cite when the model cited nothing.

    Returns:
        De-duplicated citations ordered by first appearance.
    """
    citations: OrderedDict[str, Citation] = OrderedDict()
    for source, page in extract_citations(answer):
        context = _match_context(source, page, contexts)
        if context is None:
            key = f"{source}{f', p.{page}' if page else ''}"
            citations.setdefault(
                key,
                Citation(source=source, page=page, source_type="unknown", snippet="", score=None),
            )
            continue
        key = context.citation_key
        if key in citations:
            continue
        citations[key] = Citation(
            source=context.source,
            page=context.page,
            source_type=context.source_type,
            snippet=_snippet(context.text),
            score=context.rerank_score if context.rerank_score is not None else context.score,
        )

    if not citations and contexts:
        for context in contexts[:fallback_count]:
            citations.setdefault(
                context.citation_key,
                Citation(
                    source=context.source,
                    page=context.page,
                    source_type=context.source_type,
                    snippet=_snippet(context.text),
                    score=context.rerank_score if context.rerank_score is not None else context.score,
                ),
            )
    return list(citations.values())


def format_response(
    query: str,
    answer: str,
    contexts: Sequence[RetrievedContext] = (),
    timings_ms: dict[str, float] | None = None,
    retriever_stats: dict[str, int] | None = None,
    errors: Sequence[str] | None = None,
    provider: str = "",
    model: str = "",
) -> RAGAnswer:
    """Build the final :class:`RAGAnswer` object.

    Args:
        query: Original user question.
        answer: Raw model answer.
        contexts: Contexts used for generation.
        timings_ms: Stage timings.
        retriever_stats: Per-retriever result counts.
        errors: Non-fatal errors collected along the way.
        provider: LLM provider name.
        model: Model name.

    Returns:
        A structured answer with citations, contexts and diagnostics.
    """
    cleaned = (answer or "").strip()
    return RAGAnswer(
        answer=cleaned,
        citations=resolve_citations(cleaned, contexts),
        contexts=list(contexts),
        query=query,
        timings_ms=dict(timings_ms or {}),
        retriever_stats=dict(retriever_stats or {}),
        errors=list(errors or []),
        provider=provider,
        model=model,
        grounded=not is_refusal(cleaned),
    )


def bullet_citations(citations: Iterable[Citation]) -> list[str]:
    """Render citations as short bullet strings for the UI."""
    bullets: list[str] = []
    for citation in citations:
        label = f"**{citation.key}**"
        if citation.snippet:
            bullets.append(f"- {label} — {citation.snippet}")
        else:
            bullets.append(f"- {label}")
    return bullets


def group_contexts_by_source(contexts: Iterable[RetrievedContext]) -> dict[str, list[RetrievedContext]]:
    """Group contexts by source for the expandable "sources" panel."""
    grouped: dict[str, list[RetrievedContext]] = {}
    for context in contexts:
        grouped.setdefault(context.source or "unknown", []).append(context)
    return grouped


def render_markdown(answer: RAGAnswer) -> str:
    """Render an answer as markdown (used by the CLI, RAGAS logs and docs)."""
    lines = [answer.answer.strip(), ""]
    if answer.citations:
        lines.append("**Citations**")
        lines.extend(bullet_citations(answer.citations))
        lines.append("")
    if answer.contexts:
        lines.append("**Sources**")
        for context in answer.contexts:
            origin = ", ".join(context.retrievers) if context.retrievers else context.retriever
            lines.append(
                f"- `{context.citation_key}` (score={float(context.rerank_score or 0):.3f}, via {origin})"
            )
    if answer.errors:
        lines.append("")
        lines.append("**Warnings**")
        lines.extend(f"- {error}" for error in answer.errors)
    return "\n".join(lines).strip()


__all__ = [
    "extract_citations",
    "resolve_citations",
    "format_response",
    "bullet_citations",
    "group_contexts_by_source",
    "render_markdown",
]
