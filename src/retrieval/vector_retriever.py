"""Dense vector retriever (FAISS + bge-m3 / OpenAI embeddings)."""

from __future__ import annotations

import logging
from typing import Any

from src.config import Settings, get_settings
from src.errors import IndexingError, RetrievalError
from src.indexing.vector_index import VectorIndex
from src.models import RetrievedContext
from src.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)


class VectorRetriever(BaseRetriever):
    """Retrieve chunks from the FAISS vector index."""

    name = "vector"

    def __init__(
        self,
        settings: Settings | None = None,
        index: VectorIndex | None = None,
        embed_model: Any | None = None,
        weight: float | None = None,
    ) -> None:
        settings = settings or get_settings()
        super().__init__(settings)
        self.index = index or VectorIndex(settings, embed_model=embed_model)
        self.weight = float(weight if weight is not None else settings.vector_weight)

    def is_available(self) -> bool:
        """Return ``True`` when the FAISS index exists on disk."""
        return self.index.exists()

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Vector-search ``query`` and return the best chunks."""
        if not query or not query.strip():
            return []
        try:
            contexts = self.index.search(query, top_k=top_k)
        except IndexingError as exc:
            self.last_error = str(exc)
            raise RetrievalError(f"vector retriever failed: {exc}") from exc
        return contexts


__all__ = ["VectorRetriever"]
