"""Common retriever interface.

All retrievers return :class:`~src.models.RetrievedContext` objects so the merge
and rerank stages never need to know which index produced a result.
"""

from __future__ import annotations

import abc
import logging

from src.config import Settings, get_settings
from src.models import RetrievedContext

logger = logging.getLogger(__name__)


class BaseRetriever(abc.ABC):
    """Abstract retriever."""

    name: str = "base"
    weight: float = 1.0

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.last_error: str | None = None

    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Return up to ``top_k`` contexts for ``query``.

        Args:
            query: User query (Bangla or English).
            top_k: Maximum number of contexts.

        Returns:
            Ranked contexts (best first).
        """

    def is_available(self) -> bool:
        """Return ``True`` when the backing index exists."""
        return True

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"<{type(self).__name__} name={self.name!r} weight={self.weight}>"


__all__ = ["BaseRetriever"]
