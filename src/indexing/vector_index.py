"""FAISS vector index built on LlamaIndex's ``VectorStoreIndex``.

Design notes:
    * inner-product FAISS index over L2-normalised vectors == cosine similarity;
    * node ids are the chunk ids, so re-indexing is idempotent;
    * the index is persisted with ``StorageContext.persist`` (FAISS binary +
      docstore/index-store JSON), and the embedding dimension is recorded in
      the project manifest so a model change triggers a clean rebuild.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.errors import IndexingError
from src.indexing.embeddings import get_embed_model, get_embedding_dimension
from src.indexing.nodes import chunk_to_node, contexts_from_nodes
from src.models import Chunk, RetrievedContext, utc_now_iso

logger = logging.getLogger(__name__)

FAISS_FILE = "default__vector_store.json"


class VectorIndex:
    """Wrapper around a persisted FAISS-backed LlamaIndex vector index."""

    retriever_name = "vector"

    def __init__(
        self,
        settings: Settings | None = None,
        embed_model: Any | None = None,
        persist_dir: Path | str | None = None,
    ) -> None:
        """
        Args:
            settings: Project settings.
            embed_model: Optional shared embedding model (created lazily).
            persist_dir: Override for the index directory.
        """
        self.settings = settings or get_settings()
        self.persist_dir = Path(persist_dir or self.settings.vector_index_dir)
        self._embed_model = embed_model
        self._index: Any = None

    # ------------------------------------------------------------ properties
    @property
    def embed_model(self) -> Any:
        """Lazily create the embedding model."""
        if self._embed_model is None:
            self._embed_model = get_embed_model(self.settings)
        return self._embed_model

    @property
    def dimension(self) -> int:
        """Embedding dimension of the configured model."""
        return get_embedding_dimension(self.embed_model)

    def exists(self) -> bool:
        """Return ``True`` when a persisted index can be loaded."""
        return (self.persist_dir / FAISS_FILE).exists() and (self.persist_dir / "docstore.json").exists()

    # ------------------------------------------------------------------ build
    def _new_faiss_index(self, dimension: int) -> Any:
        try:
            import faiss
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise IndexingError("faiss-cpu is not installed; run `pip install faiss-cpu`") from exc
        index = faiss.IndexFlatIP(dimension)
        try:
            index = faiss.IndexIDMap2(index)
        except Exception:  # pragma: no cover - very old faiss
            logger.debug("IndexIDMap2 unavailable; using flat index")
        return index

    def _make_nodes(self, chunks: Sequence[Chunk]) -> list[Any]:
        """Create the nodes that will be indexed (overridable hook)."""
        return [chunk_to_node(chunk) for chunk in chunks]

    def build(self, chunks: Sequence[Chunk], persist: bool = True, show_progress: bool = False) -> int:
        """Build (or rebuild) the FAISS index from scratch.

        Args:
            chunks: Chunks to index.
            persist: Persist the index to ``persist_dir``.
            show_progress: Show a tqdm progress bar while embedding.

        Returns:
            Number of indexed chunks.
        """
        if not chunks:
            raise IndexingError("cannot build a vector index without chunks")

        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.vector_stores.faiss import FaissVectorStore

        dimension = self.dimension
        logger.info("Building FAISS index: %s chunks, %s dimensions", len(chunks), dimension)
        try:
            import faiss

            faiss_index = faiss.IndexFlatIP(dimension)
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            nodes = self._make_nodes(chunks)
            index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=show_progress,
            )
        except Exception as exc:
            raise IndexingError(f"failed to build FAISS index: {exc}") from exc

        self._index = index
        if persist:
            self.persist()
        logger.info("FAISS index built with %s nodes", len(nodes))
        return len(nodes)

    def add_chunks(self, chunks: Sequence[Chunk]) -> int:
        """Insert additional chunks into an already loaded index."""
        index = self.load()
        nodes = [chunk_to_node(chunk) for chunk in chunks]
        for node in nodes:
            index.insert_nodes([node])
        self.persist()
        logger.info("Inserted %s nodes into the FAISS index", len(nodes))
        return len(nodes)

    def persist(self) -> None:
        """Persist the index (FAISS binary + JSON stores) and metadata."""
        if self._index is None:
            raise IndexingError("no index in memory to persist")
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._index.storage_context.persist(persist_dir=str(self.persist_dir))
            meta = {
                "index_type": "faiss_flat_ip",
                "embedding_backend": self.settings.embedding_backend,
                "embedding_model": self.settings.embedding_model,
                "dimension": self.dimension,
                "updated_at": utc_now_iso(),
            }
            (self.persist_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:
            raise IndexingError(f"failed to persist FAISS index: {exc}") from exc

    # ------------------------------------------------------------------- load
    def load(self) -> Any:
        """Load the persisted index (cached in memory)."""
        if self._index is not None:
            return self._index
        if not self.exists():
            raise IndexingError(
                f"no vector index found at {self.persist_dir}; build it first "
                "(python scripts/build_index.py or the Streamlit UI)"
            )
        from llama_index.core import StorageContext, load_index_from_storage
        from llama_index.vector_stores.faiss import FaissVectorStore

        meta_path = self.persist_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                stored_dim = int(meta.get("dimension", 0) or 0)
                if stored_dim and stored_dim != self.dimension:
                    raise IndexingError(
                        f"embedding dimension mismatch: index={stored_dim}, model={self.dimension}. "
                        "Rebuild the index after changing the embedding model."
                    )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                if isinstance(exc, IndexingError):
                    raise
                logger.debug("Ignoring unreadable index metadata: %s", exc)

        try:
            vector_store = FaissVectorStore.from_persist_dir(str(self.persist_dir))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=str(self.persist_dir)
            )
            self._index = load_index_from_storage(storage_context, embed_model=self.embed_model)
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"failed to load FAISS index: {exc}") from exc
        return self._index

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Return the ``top_k`` most similar chunks for ``query``."""
        if not query or not query.strip():
            return []
        index = self.load()
        try:
            retriever = index.as_retriever(similarity_top_k=max(1, top_k))
            nodes = retriever.retrieve(query)
        except Exception as exc:
            raise IndexingError(f"vector search failed: {exc}") from exc
        contexts = contexts_from_nodes(nodes, retriever=self.retriever_name)
        logger.debug("Vector search returned %s contexts for %r", len(contexts), query[:60])
        return contexts

    def node_count(self) -> int:
        """Return the number of nodes stored in the docstore."""
        try:
            index = self.load()
            docstore = index.storage_context.docstore
            return len(docstore.docs)
        except Exception:  # pragma: no cover - defensive
            return 0


__all__ = ["VectorIndex", "FAISS_FILE"]
