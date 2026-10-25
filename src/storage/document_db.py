"""Unified Document DB (SQLite) for raw text and chunks.

The database is the single source of truth for the whole pipeline:

* ``documents``  – raw text per page/row-block/URL with metadata
* ``chunks``     – cleaned, token-bounded chunks (rebuilt deterministically)
* ``ingestion_log`` – audit trail of ingestion runs
* ``app_meta``   – small key/value store (index manifest pointers, ...)

Connections are created per call (thread-safe for Streamlit and the
``ThreadPoolExecutor`` retrievers) with WAL journaling for concurrent reads.

Example:
    >>> db = DocumentDB(":memory:")
    >>> db.insert_documents(docs)
    1
    >>> db.count_documents()
    1
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings
from src.errors import StorageError
from src.models import Chunk, RawDocument, SourceType, utc_now_iso

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    source_type  TEXT NOT NULL,
    page         INTEGER,
    title        TEXT,
    text         TEXT NOT NULL,
    char_count   INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(source_type);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    source      TEXT NOT NULL,
    source_type TEXT NOT NULL,
    page        INTEGER,
    title       TEXT,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,
    source_type TEXT NOT NULL,
    status      TEXT NOT NULL,
    message     TEXT NOT NULL DEFAULT '',
    doc_count   INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _json_loads(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {"value": loaded}
    except json.JSONDecodeError:
        return {}


def _json_dumps(value: dict[str, Any] | None) -> str:
    try:
        return json.dumps(value or {}, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return "{}"


class DocumentDB:
    """Thin, thread-safe repository over a SQLite database."""

    def __init__(self, db_path: str | Path | None = None, settings: Settings | None = None) -> None:
        """Create the store and ensure the schema exists.

        Args:
            db_path: SQLite file path, or ``":memory:"`` for tests. Defaults to
                ``<data_dir>/db/documents.db`` from the settings.
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        if db_path is None:
            db_path = self.settings.db_path
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            # Keep one connection alive so the in-memory DB is not dropped.
            self._memory_connection = self._connect()
        self._init_schema()
        logger.debug("DocumentDB ready at %s", self.db_path)

    # ------------------------------------------------------------ plumbing
    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False,
            )
        except sqlite3.Error as exc:  # pragma: no cover - fs dependent
            raise StorageError(f"cannot open database {self.db_path}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.Error:  # pragma: no cover - :memory: / read-only fs
            pass
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a transaction-scoped connection (commits or rolls back)."""
        with self._lock:
            if self._memory_connection is not None:
                conn = self._memory_connection
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                return
            conn = self._connect()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self) -> None:
        try:
            with self.connection() as conn:
                conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot initialise schema: {exc}") from exc

    # ----------------------------------------------------------- documents
    def insert_documents(self, documents: Iterable[RawDocument]) -> int:
        """Insert raw documents, ignoring exact duplicates.

        Args:
            documents: Raw documents from a loader.

        Returns:
            Number of newly inserted documents.
        """
        rows = []
        for doc in documents:
            rows.append(
                (
                    doc.doc_id,
                    doc.source,
                    str(doc.source_type),
                    doc.page,
                    doc.title,
                    doc.text,
                    doc.char_count,
                    doc.doc_id,
                    doc.created_at,
                    _json_dumps(doc.metadata),
                )
            )
        if not rows:
            return 0
        try:
            with self.connection() as conn:
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR IGNORE INTO documents
                        (doc_id, source, source_type, page, title, text,
                         char_count, content_hash, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                inserted = conn.total_changes - before
        except sqlite3.Error as exc:
            raise StorageError(f"failed to insert documents: {exc}") from exc
        logger.info("Stored %s/%s raw documents (duplicates skipped)", inserted, len(rows))
        return inserted

    def get_document(self, doc_id: str) -> RawDocument | None:
        """Return a single raw document by id."""
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(
        self,
        source_type: str | None = None,
        source: str | None = None,
        limit: int = 1000,
    ) -> list[RawDocument]:
        """List raw documents, optionally filtered by type or source."""
        query = "SELECT * FROM documents"
        clauses: list[str] = []
        params: list[Any] = []
        if source_type:
            clauses.append("source_type = ?")
            params.append(str(source_type))
        if source:
            clauses.append("source = ?")
            params.append(source)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY source, page LIMIT ?"
        params.append(limit)
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_document(row) for row in rows]

    def iter_documents(self, source_type: str | None = None) -> Iterator[RawDocument]:
        """Iterate over all raw documents without loading them all at once."""
        offset = 0
        page_size = 500
        while True:
            query = "SELECT * FROM documents"
            params: list[Any] = []
            if source_type:
                query += " WHERE source_type = ?"
                params.append(str(source_type))
            query += " ORDER BY source, page LIMIT ? OFFSET ?"
            with self.connection() as conn:
                rows = conn.execute(query, [*params, page_size, offset]).fetchall()
            if not rows:
                return
            for row in rows:
                yield self._row_to_document(row)
            offset += page_size

    def count_documents(self, source_type: str | None = None) -> int:
        """Count stored raw documents."""
        query = "SELECT COUNT(*) AS n FROM documents"
        params: list[Any] = []
        if source_type:
            query += " WHERE source_type = ?"
            params.append(str(source_type))
        with self.connection() as conn:
            row = conn.execute(query, params).fetchone()
        return int(row["n"]) if row else 0

    def list_sources(self) -> list[str]:
        """Return every distinct source name in insertion order."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT source, MIN(created_at) AS first_seen FROM documents GROUP BY source ORDER BY first_seen"
            ).fetchall()
        return [str(row["source"]) for row in rows]

    def delete_source(self, source: str) -> int:
        """Delete all documents and chunks for ``source``."""
        try:
            with self.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
                cursor = conn.execute("DELETE FROM documents WHERE source = ?", (source,))
                deleted = cursor.rowcount
        except sqlite3.Error as exc:
            raise StorageError(f"failed to delete source {source}: {exc}") from exc
        logger.info("Deleted source %s (%s documents)", source, deleted)
        return int(deleted)

    def replace_documents(self, source: str, documents: Sequence[RawDocument]) -> int:
        """Replace every stored document of ``source`` with a new version."""
        self.delete_source(source)
        return self.insert_documents(documents)

    def clear_all(self) -> None:
        """Delete every document, chunk and log entry (keeps the schema)."""
        with self.connection() as conn:
            conn.executescript(
                "DELETE FROM chunks; DELETE FROM documents; DELETE FROM ingestion_log; DELETE FROM app_meta;"
            )
        logger.warning("Document database cleared")

    # -------------------------------------------------------------- chunks
    def insert_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Upsert processed chunks."""
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source,
                str(chunk.source_type),
                chunk.page,
                chunk.title,
                chunk.chunk_index,
                chunk.text,
                chunk.token_count,
                chunk.created_at,
                _json_dumps(chunk.metadata),
            )
            for chunk in chunks
        ]
        if not rows:
            return 0
        try:
            with self.connection() as conn:
                before = conn.total_changes
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks
                        (chunk_id, doc_id, source, source_type, page, title,
                         chunk_index, text, token_count, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                changed = conn.total_changes - before
        except sqlite3.Error as exc:
            raise StorageError(f"failed to insert chunks: {exc}") from exc
        logger.debug("Upserted %s chunks", changed)
        return int(changed)

    def get_chunks(self, source: str | None = None, doc_id: str | None = None) -> list[Chunk]:
        """Return chunks ordered by source/page/index."""
        query = "SELECT * FROM chunks"
        clauses: list[str] = []
        params: list[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if doc_id:
            clauses.append("doc_id = ?")
            params.append(doc_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY source, page, chunk_index"
        with self.connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def get_chunks_by_ids(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch chunks by id, preserving the requested order."""
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", list(chunk_ids)
            ).fetchall()
        by_id = {str(row["chunk_id"]): self._row_to_chunk(row) for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def count_chunks(self) -> int:
        """Count stored chunks."""
        with self.connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()
        return int(row["n"]) if row else 0

    def delete_chunks(self) -> None:
        """Delete every chunk (used before a full re-processing run)."""
        with self.connection() as conn:
            conn.execute("DELETE FROM chunks")

    # ----------------------------------------------------------- audit log
    def log_ingestion(
        self,
        source: str,
        source_type: str,
        status: str,
        message: str = "",
        doc_count: int = 0,
        chunk_count: int = 0,
    ) -> None:
        """Append an entry to the ingestion audit log."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_log
                    (source, source_type, status, message, doc_count, chunk_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source, str(source_type), status, message[:2000], doc_count, chunk_count, utc_now_iso()),
            )

    def recent_ingestions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent ingestion log entries."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM ingestion_log ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]

    # -------------------------------------------------------------- meta
    def set_meta(self, key: str, value: Any) -> None:
        """Store a JSON value under ``key``."""
        payload = value if isinstance(value, str) else _json_dumps({"value": value})
        with self.connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO app_meta (key, value) VALUES (?, ?)", (key, payload)
            )

    def get_meta(self, key: str, default: Any = None) -> Any:
        """Read a value previously stored with :meth:`set_meta`."""
        with self.connection() as conn:
            row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        raw = str(row["value"])
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if isinstance(parsed, dict) and set(parsed) == {"value"}:
            return parsed["value"]
        return parsed

    # ------------------------------------------------------------- stats
    def get_stats(self) -> dict[str, Any]:
        """Return counts and size information used by the Streamlit sidebar."""
        with self.connection() as conn:
            documents = int(conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
            chunks = int(conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
            chars = int(conn.execute("SELECT COALESCE(SUM(char_count), 0) AS n FROM documents").fetchone()["n"])
            tokens = int(conn.execute("SELECT COALESCE(SUM(token_count), 0) AS n FROM chunks").fetchone()["n"])
            by_type = {
                str(row["source_type"]): int(row["n"])
                for row in conn.execute(
                    "SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type"
                ).fetchall()
            }
            last = conn.execute(
                "SELECT created_at FROM ingestion_log ORDER BY id DESC LIMIT 1"
            ).fetchone()
        size_bytes = Path(self.db_path).stat().st_size if self.db_path != ":memory:" and Path(self.db_path).exists() else 0
        return {
            "documents": documents,
            "chunks": chunks,
            "characters": chars,
            "tokens": tokens,
            "documents_by_type": by_type,
            "sources": len(self.list_sources()),
            "last_ingestion": str(last["created_at"]) if last else None,
            "db_path": self.db_path,
            "db_size_bytes": size_bytes,
        }

    # ------------------------------------------------------------ mapping
    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> RawDocument:
        return RawDocument(
            doc_id=str(row["doc_id"]),
            source=str(row["source"]),
            source_type=SourceType(str(row["source_type"])),
            text=str(row["text"]),
            page=int(row["page"]) if row["page"] is not None else None,
            title=row["title"],
            created_at=str(row["created_at"]),
            metadata=_json_loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=str(row["chunk_id"]),
            doc_id=str(row["doc_id"]),
            text=str(row["text"]),
            chunk_index=int(row["chunk_index"]),
            source=str(row["source"]),
            source_type=SourceType(str(row["source_type"])),
            page=int(row["page"]) if row["page"] is not None else None,
            title=row["title"],
            token_count=int(row["token_count"]),
            created_at=str(row["created_at"]),
            metadata=_json_loads(row["metadata"]),
        )


__all__ = ["DocumentDB"]
