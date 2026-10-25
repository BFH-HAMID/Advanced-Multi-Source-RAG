"""Conversion helpers between project models and LlamaIndex nodes.

Keeping this logic in one place guarantees that the FAISS, sentence-window and
graph indexes all agree on node ids and metadata keys, which in turn makes the
retriever merge/dedupe step deterministic.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Any

from llama_index.core.schema import NodeWithScore, TextNode

from src.models import Chunk, RetrievedContext
from src.processing.metadata import build_node_metadata

logger = logging.getLogger(__name__)


def chunk_to_node(chunk: Chunk, text: str | None = None) -> TextNode:
    """Build a LlamaIndex :class:`TextNode` from a chunk.

    Args:
        chunk: Processed chunk.
        text: Optional text override (defaults to the chunk text).

    Returns:
        A node whose ``id_`` equals the chunk id.
    """
    return TextNode(
        id_=chunk.chunk_id,
        text=text if text is not None else chunk.text,
        metadata=build_node_metadata(chunk),
        # Keep metadata in the embedded text so BM25-style fallbacks can use it.
        excluded_embed_metadata_keys=["created_at", "chunk_index", "token_count"],
        excluded_llm_metadata_keys=["chunk_id", "doc_id", "tokens", "has_numbers"],
    )


def _coerce_page(value: Any) -> int | None:
    """Convert a stored page value to an int (``None`` when unavailable)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        page = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return page if page > 0 else None


def node_to_context(
    node_with_score: Any,
    retriever: str = "",
    text_override: str | None = None,
) -> RetrievedContext:
    """Convert a LlamaIndex node (with score) into a :class:`RetrievedContext`.

    Args:
        node_with_score: ``NodeWithScore`` or a bare ``TextNode``.
        retriever: Name of the producing retriever (for provenance).
        text_override: Use this text instead of the node text (e.g. a
            sentence window).

    Returns:
        A populated retrieved-context object.
    """
    node = getattr(node_with_score, "node", node_with_score)
    score = float(getattr(node_with_score, "score", 0.0) or 0.0)
    metadata: dict[str, Any] = dict(getattr(node, "metadata", {}) or {})
    chunk_id = str(metadata.get("chunk_id") or getattr(node, "node_id", ""))
    text = text_override if text_override is not None else node.get_content()
    return RetrievedContext(
        node_id=str(getattr(node, "node_id", chunk_id)),
        text=text,
        score=score,
        source=str(metadata.get("source", "")),
        page=_coerce_page(metadata.get("page")),
        source_type=str(metadata.get("source_type", "text")),
        retriever=retriever,
        retrievers=[retriever] if retriever else [],
        metadata={**metadata, "chunk_id": chunk_id},
    )


def contexts_from_nodes(
    nodes: Sequence[Any],
    retriever: str,
    text_overrides: Sequence[str] | None = None,
) -> list[RetrievedContext]:
    """Convert a sequence of scored nodes into retrieved contexts."""
    contexts: list[RetrievedContext] = []
    for index, node in enumerate(nodes):
        override = text_overrides[index] if text_overrides else None
        context = node_to_context(node, retriever=retriever, text_override=override)
        context.rank = index
        contexts.append(context)
    return contexts


def window_text(node: Any, fallback: str | None = None) -> str:
    """Return the sentence window stored in a node's metadata.

    Args:
        node: Sentence-window node.
        fallback: Text to use when no window is present.

    Returns:
        The surrounding window (``window`` metadata) or the node text.
    """
    metadata = getattr(node, "metadata", {}) or {}
    window = metadata.get("window")
    if isinstance(window, str) and window.strip():
        return window.strip()
    return fallback if fallback is not None else node.get_content()


def nodes_metadata(nodes: Iterable[TextNode]) -> list[dict[str, Any]]:
    """Extract metadata dictionaries from nodes (debugging helper)."""
    return [dict(node.metadata) for node in nodes]


__all__ = [
    "chunk_to_node",
    "node_to_context",
    "contexts_from_nodes",
    "window_text",
    "nodes_metadata",
    "NodeWithScore",
]
