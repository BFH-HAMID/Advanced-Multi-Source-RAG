"""Retriever factory: builds the three retrievers and the parallel merger."""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings, get_settings
from src.retrieval.base import BaseRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.multi_retriever import MultiRetrievalResult, MultiRetriever, merge_results
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.window_retriever import WindowRetriever
from src.storage.document_db import DocumentDB

logger = logging.getLogger(__name__)


def build_retrievers(
    settings: Settings | None = None,
    embed_model: Any | None = None,
    db: DocumentDB | None = None,
    include_unavailable: bool = True,
) -> tuple[list[BaseRetriever], list[str]]:
    """Create vector, sentence-window and graph retrievers.

    Args:
        settings: Project settings.
        embed_model: Shared embedding model (created per retriever when omitted).
        db: Document DB used by the graph retriever to hydrate chunk text.
        include_unavailable: Keep retrievers whose index is missing (they will
            report an error at query time) instead of dropping them.

    Returns:
        ``(retrievers, warnings)`` — warnings describe skipped indexes.
    """
    settings = settings or get_settings()
    db = db or DocumentDB(settings=settings)
    candidates: list[BaseRetriever] = [
        VectorRetriever(settings, embed_model=embed_model),
        WindowRetriever(settings, embed_model=embed_model),
        GraphRetriever(settings, db=db),
    ]

    retrievers: list[BaseRetriever] = []
    warnings: list[str] = []
    for retriever in candidates:
        try:
            available = retriever.is_available()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not check availability of %s: %s", retriever.name, exc)
            available = False
        if available:
            retrievers.append(retriever)
        else:
            message = f"{retriever.name} index not found; retriever disabled"
            logger.warning(message)
            warnings.append(message)
            if include_unavailable:
                retrievers.append(retriever)
    return retrievers, warnings


def build_multi_retriever(
    settings: Settings | None = None,
    embed_model: Any | None = None,
    db: DocumentDB | None = None,
) -> tuple[MultiRetriever, list[str]]:
    """Create a :class:`MultiRetriever` with all available retrievers."""
    settings = settings or get_settings()
    retrievers, warnings = build_retrievers(settings, embed_model=embed_model, db=db)
    return MultiRetriever(retrievers, settings=settings), warnings


__all__ = [
    "BaseRetriever",
    "VectorRetriever",
    "WindowRetriever",
    "GraphRetriever",
    "MultiRetriever",
    "MultiRetrievalResult",
    "merge_results",
    "build_retrievers",
    "build_multi_retriever",
]
