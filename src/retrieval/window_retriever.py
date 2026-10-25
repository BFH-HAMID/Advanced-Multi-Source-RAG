"""Sentence-window retriever.

Matches on individual sentences but returns the surrounding window
(``window_size`` sentences on each side) so the generator gets local context.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings, get_settings
from src.errors import IndexingError, RetrievalError
from src.indexing.sentence_window import SentenceWindowIndex
from src.models import RetrievedContext
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class WindowRetriever(BaseRetriever):
    """Retrieve sentence windows from the sentence-window index."""

    name = "sentence_window"

    def __init__(
        self,
        settings: Settings | None = None,
        index: SentenceWindowIndex | None = None,
        embed_model: Any | None = None,
        weight: float | None = None,
    ) -> None:
        settings = settings or get_settings()
        super().__init__(settings)
        self.index = index or SentenceWindowIndex(settings, embed_model=embed_model)
        self.weight = float(weight if weight is not None else settings.window_weight)

    def is_available(self) -> bool:
        """Return ``True`` when the sentence-window index exists on disk."""
        return self.index.exists()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Return the windows whose centre sentence best matches ``query``."""
        if not query or not query.strip():
            return []
        try:
            contexts = self.index.search(query, top_k=top_k)
        except IndexingError as exc:
            self.last_error = str(exc)
            raise RetrievalError(f"sentence-window retriever failed: {exc}") from exc
        return contexts


__all__ = ["WindowRetriever"]
