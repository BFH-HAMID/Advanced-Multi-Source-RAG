"""Sentence-window index (window size = 3).

Why: embeddings of single sentences are precise but context-poor.  The
sentence-window technique embeds each **sentence** for high-precision matching,
then hands the LLM the surrounding **window** of ``2 * window_size + 1``
sentences for context.

Implementation: LlamaIndex's ``SentenceWindowNodeParser`` splits each chunk into
sentence nodes and stores the surrounding window in the ``window`` metadata key,
which this module treats as the retrieval payload.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import Document

from src.config import Settings, get_settings
from src.errors import IndexingError
from src.indexing.nodes import contexts_from_nodes
from src.indexing.vector_index import VectorIndex
from src.models import Chunk, RetrievedContext, stable_id
from src.processing.metadata import build_node_metadata
from src.text_utils import split_sentences

logger = logging.getLogger(__name__)


class SentenceWindowIndex(VectorIndex):
    """FAISS index over sentence nodes that carry their surrounding window."""

    retriever_name = "sentence_window"

    def __init__(
        self,
        settings: Settings | None = None,
        embed_model: Any | None = None,
        persist_dir: Path | str | None = None,
        window_size: int | None = None,
    ) -> None:
        """
        Args:
            settings: Project settings.
            embed_model: Optional shared embedding model.
            persist_dir: Index directory (defaults to ``data/indexes/sentence_window``).
            window_size: Sentences kept on each side of the matched sentence.
        """
        settings = settings or get_settings()
        super().__init__(
            settings=settings,
            embed_model=embed_model,
            persist_dir=persist_dir or settings.window_index_dir,
        )
        self.window_size = int(window_size if window_size is not None else settings.sentence_window_size)
        if self.window_size < 0:
            raise IndexingError("window_size must be >= 0")

    # ------------------------------------------------------------------ nodes
    def _make_nodes(self, chunks: Sequence[Chunk]) -> list[Any]:
        """Split chunks into sentence nodes carrying a ``window`` metadata key."""
        parser = SentenceWindowNodeParser.from_defaults(
            # Unicode-aware splitter: the default one mis-handles Bengali danda (।).
            sentence_splitter=split_sentences,
            window_size=self.window_size,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
            include_metadata=True,
            include_prev_next_rel=False,
        )
        documents = [
            Document(
                id_=chunk.chunk_id,
                text=chunk.text,
                metadata=build_node_metadata(chunk),
            )
            for chunk in chunks
        ]
        nodes = parser.get_nodes_from_documents(documents)
        for position, node in enumerate(nodes):
            chunk_id = str(node.metadata.get("chunk_id", ""))
            sentence = str(node.metadata.get("original_text", node.get_content()))
            # Deterministic ids keep rebuilds idempotent.
            node.id_ = stable_id(chunk_id, position, sentence)
            node.metadata["window_size"] = self.window_size
        logger.info("Sentence-window parser produced %s nodes (window=%s)", len(nodes), self.window_size)
        return nodes

    # ----------------------------------------------------------------- search
    def search(self, query: str, top_k: int = 10) -> list[RetrievedContext]:
        """Match sentences, then return their surrounding windows as context."""
        if not query or not query.strip():
            return []
        index = self.load()
        try:
            retriever = index.as_retriever(similarity_top_k=max(1, top_k))
            nodes = retriever.retrieve(query)
        except Exception as exc:
            raise IndexingError(f"sentence-window search failed: {exc}") from exc

        contexts = contexts_from_nodes(nodes, retriever=self.retriever_name)
        for context, node in zip(contexts, nodes, strict=True):
            metadata = getattr(node.node, "metadata", {}) or {}
            window = metadata.get("window")
            sentence = metadata.get("original_text") or node.node.get_content()
            if isinstance(window, str) and window.strip():
                context.text = window.strip()
            context.metadata["matched_sentence"] = sentence
            context.metadata["window_size"] = self.window_size
        logger.debug("Sentence-window search returned %s windows", len(contexts))
        return contexts


__all__ = ["SentenceWindowIndex"]
