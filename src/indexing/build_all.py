"""Build every index (vector, sentence-window, graph) and write the manifest.

The manifest (``data/indexes/manifest.json``) records what is inside the
indexes — chunk fingerprint, embedding model/dimension, graph size, timestamps.
The Streamlit UI uses it to detect when stored indexes are stale.

Example:
    >>> from src.indexing.build_all import IndexBuilder
    >>> builder = IndexBuilder(settings)          # doctest: +SKIP
    >>> report = builder.build_all()               # doctest: +SKIP
    >>> report.vector_nodes > 0                    # doctest: +SKIP
    True
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.errors import IndexingError
from src.indexing.embeddings import get_embed_model, get_embedding_dimension
from src.indexing.graph_index import GraphIndex
from src.indexing.sentence_window import SentenceWindowIndex
from src.indexing.vector_index import VectorIndex
from src.models import Chunk, utc_now_iso
from src.storage.document_db import DocumentDB

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


def chunks_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Return a stable fingerprint of the chunk set (ids + text lengths)."""
    digest = hashlib.sha256()
    for chunk in sorted(chunks, key=lambda item: item.chunk_id):
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(str(len(chunk.text)).encode("utf-8"))
    return digest.hexdigest()[:32]


@dataclass(slots=True)
class IndexBuildReport:
    """Outcome of a full index build."""

    chunks: int = 0
    vector_nodes: int = 0
    window_nodes: int = 0
    graph_entities: int = 0
    graph_relations: int = 0
    graph_triples: int = 0
    graph_llm_used: bool = False
    embedding_backend: str = ""
    embedding_model: str = ""
    embedding_dimension: int = 0
    fingerprint: str = ""
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    built_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunks": self.chunks,
            "vector_nodes": self.vector_nodes,
            "window_nodes": self.window_nodes,
            "graph_entities": self.graph_entities,
            "graph_relations": self.graph_relations,
            "graph_triples": self.graph_triples,
            "graph_llm_used": self.graph_llm_used,
            "embedding_backend": self.embedding_backend,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "fingerprint": self.fingerprint,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors,
            "built_at": self.built_at,
        }


class IndexBuilder:
    """Coordinates the three indexes so they always stay in sync."""

    def __init__(
        self,
        settings: Settings | None = None,
        embed_model: Any | None = None,
        llm: Any | None = None,
        db: DocumentDB | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db = db or DocumentDB(settings=self.settings)
        self.embed_model = embed_model or get_embed_model(self.settings)
        self.llm = llm
        self.vector_index = VectorIndex(self.settings, embed_model=self.embed_model)
        self.window_index = SentenceWindowIndex(self.settings, embed_model=self.embed_model)
        self.graph_index = GraphIndex(self.settings, llm=llm)

    # ------------------------------------------------------------- building
    def build_all(
        self,
        chunks: Sequence[Chunk] | None = None,
        persist: bool = True,
        skip_graph: bool = False,
        skip_window: bool = False,
        progress: ProgressCallback | None = None,
    ) -> IndexBuildReport:
        """Build vector, sentence-window and graph indexes.

        Args:
            chunks: Chunks to index; when ``None`` they are read from SQLite.
            persist: Persist all indexes to disk.
            skip_graph: Skip the (slow) graph build.
            skip_window: Skip the sentence-window index.
            progress: Optional ``(stage, done, total)`` callback for the UI.

        Returns:
            An :class:`IndexBuildReport`.

        Raises:
            IndexingError: If no chunks are available or the vector index fails.
        """
        started = time.perf_counter()
        if chunks is None:
            chunks = self.db.get_chunks()
        chunks = list(chunks)
        if not chunks:
            raise IndexingError(
                "no chunks found; ingest documents first (Streamlit sidebar or scripts/build_index.py)"
            )

        report = IndexBuildReport(
            chunks=len(chunks),
            embedding_backend=type(self.embed_model).__name__,
            embedding_model=str(
                getattr(self.embed_model, "wrapped_model_name", "")
                or getattr(self.embed_model, "model_name", "")
                or self.settings.embedding_model
            ),
            fingerprint=chunks_fingerprint(chunks),
        )
        try:
            report.embedding_dimension = get_embedding_dimension(self.embed_model)
        except Exception as exc:
            report.embedding_dimension = 0
            report.errors.append(f"embedding dimension probe failed: {exc}")

        def notify(stage: str, done: int, total: int) -> None:
            if progress:
                with contextlib.suppress(Exception):  # pragma: no cover - UI callback
                    progress(stage, done, total)

        # 1) FAISS vector index -------------------------------------------------
        notify("vector", 0, len(chunks))
        report.vector_nodes = self.vector_index.build(chunks, persist=persist)
        notify("vector", len(chunks), len(chunks))

        # 2) Sentence-window index ---------------------------------------------
        if skip_window:
            logger.info("Skipping sentence-window index (requested)")
        else:
            try:
                report.window_nodes = self.window_index.build(chunks, persist=persist)
                notify("window", len(chunks), len(chunks))
            except Exception as exc:
                logger.exception("Sentence-window index build failed")
                report.errors.append(f"sentence-window index: {exc}")

        # 3) Graph index --------------------------------------------------------
        if skip_graph:
            logger.info("Skipping graph index (requested)")
        else:
            try:
                stats = self.graph_index.build(
                    chunks,
                    persist=persist,
                    progress=lambda done, total: notify("graph", done, total),
                )
                report.graph_entities = stats.entities
                report.graph_relations = stats.relations
                report.graph_triples = stats.triples
                report.graph_llm_used = stats.llm_used
                report.errors.extend(stats.errors[:5])
            except Exception as exc:
                logger.exception("Graph index build failed")
                report.errors.append(f"graph index: {exc}")

        report.duration_ms = (time.perf_counter() - started) * 1000
        if persist:
            self.write_manifest(report)
        logger.info("Index build finished in %.1f ms (%s chunks)", report.duration_ms, report.chunks)
        return report

    # -------------------------------------------------------------- manifest
    def write_manifest(self, report: IndexBuildReport) -> Path:
        """Persist the build report as ``manifest.json``."""
        self.settings.index_dir.mkdir(parents=True, exist_ok=True)
        payload = report.to_dict()
        payload["documents"] = self.db.count_documents()
        try:
            self.settings.manifest_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # pragma: no cover - fs dependent
            logger.warning("Could not write index manifest: %s", exc)
        return self.settings.manifest_path

    def read_manifest(self) -> dict[str, Any]:
        """Return the manifest, or an empty dict when it does not exist."""
        if not self.settings.manifest_path.exists():
            return {}
        try:
            return json.loads(self.settings.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Unreadable index manifest: %s", exc)
            return {}

    def is_stale(self) -> bool:
        """Return ``True`` when the DB chunks differ from the indexed chunks."""
        manifest = self.read_manifest()
        if not manifest:
            return True
        stored_chunks = int(manifest.get("chunks", 0) or 0)
        if stored_chunks != self.db.count_chunks():
            return True
        current = chunks_fingerprint(self.db.get_chunks())
        return bool(manifest.get("fingerprint")) and manifest.get("fingerprint") != current

    def indexes_exist(self) -> dict[str, bool]:
        """Report which indexes are available on disk."""
        return {
            "vector": self.vector_index.exists(),
            "sentence_window": self.window_index.exists(),
            "graph": self.graph_index.exists(),
        }

    def describe(self) -> dict[str, Any]:
        """Return a UI-friendly description of the current index state."""
        manifest = self.read_manifest()
        return {
            "manifest": manifest,
            "indexes": self.indexes_exist(),
            "stale": self.is_stale(),
            "graph": self.graph_index.stats() if self.graph_index.exists() else {},
        }


def build_indexes(
    settings: Settings | None = None,
    skip_graph: bool = False,
    progress: ProgressCallback | None = None,
) -> IndexBuildReport:
    """Convenience wrapper used by scripts and the Streamlit UI."""
    builder = IndexBuilder(settings)
    return builder.build_all(skip_graph=skip_graph, progress=progress)


__all__ = ["IndexBuilder", "IndexBuildReport", "build_indexes", "chunks_fingerprint"]
