"""Graph-based retriever.

Links query entities to knowledge-graph nodes, expands ``max_hops`` neighbours
and returns the chunks that support those entities and relations.  This brings
back evidence that shares *no vocabulary* with the query — the classic win of
graph retrieval over pure vector search.
"""

from __future__ import annotations

import logging

from src.config import Settings, get_settings
from src.errors import IndexingError, RetrievalError
from src.indexing.graph_index import GraphIndex
from src.models import RetrievedContext
from src.retrieval.base import BaseRetriever
from src.storage.document_db import DocumentDB

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    """Retrieve chunks via entity linking and graph traversal."""

    name = "graph"

    def __init__(
        self,
        settings: Settings | None = None,
        index: GraphIndex | None = None,
        db: DocumentDB | None = None,
        weight: float | None = None,
        max_hops: int | None = None,
    ) -> None:
        settings = settings or get_settings()
        super().__init__(settings)
        self.index = index or GraphIndex(settings)
        self.db = db
        self.weight = float(weight if weight is not None else settings.graph_weight)
        if max_hops is not None:
            self.index.max_hops = int(max_hops)

    def is_available(self) -> bool:
        """Return ``True`` when a persisted graph exists."""
        return self.index.exists()

    def _hydrate_from_db(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Fill in missing text/provenance from SQLite (graph store may be stale)."""
        if self.db is None:
            return contexts
        missing = [context for context in contexts if not context.text.strip() or not context.source]
        if not missing:
            return contexts
        try:
            chunk_map = {chunk.chunk_id: chunk for chunk in self.db.get_chunks_by_ids([c.node_id for c in missing])}
        except Exception as exc:  # pragma: no cover - DB optional
            logger.debug("Could not hydrate graph contexts from DB: %s", exc)
            return contexts
        for context in missing:
            chunk = chunk_map.get(context.node_id)
            if chunk:
                context.text = context.text or chunk.text
                context.source = context.source or chunk.source
                context.source_type = chunk.source_type.value
                context.page = context.page if context.page is not None else chunk.page
        return contexts

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Return chunks supporting the entities mentioned in ``query``."""
        if not query or not query.strip():
            return []
        try:
            if not self.index.graph.number_of_nodes():
                self.index.load()
            contexts = self.index.search(query, top_k=top_k)
        except IndexingError as exc:
            self.last_error = str(exc)
            raise RetrievalError(f"graph retriever failed: {exc}") from exc
        return self._hydrate_from_db(contexts)


__all__ = ["GraphRetriever"]
