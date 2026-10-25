"""Processing pipeline: clean → chunk → tag → store.

This is step 4 of the architecture ("Processing") and is reused by the ingestion
pipeline and by re-indexing runs that start from the Unified Document DB.

Example:
    >>> processor = DocumentProcessor()
    >>> chunks, stats = processor.process(raw_documents)   # doctest: +SKIP
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from src.config import Settings, get_settings
from src.models import Chunk, RawDocument
from src.processing.chunker import TokenChunker
from src.processing.cleaner import TextCleaner
from src.processing.metadata import MetadataTagger
from src.storage.document_db import DocumentDB

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessResult:
    """Statistics of a processing run."""

    documents_in: int = 0
    documents_out: int = 0
    chunks: int = 0
    tokens: int = 0
    dropped_documents: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "documents_in": self.documents_in,
            "documents_out": self.documents_out,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "dropped_documents": self.dropped_documents,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors,
        }


class DocumentProcessor:
    """Clean, chunk and tag raw documents."""

    def __init__(
        self,
        settings: Settings | None = None,
        cleaner: TextCleaner | None = None,
        chunker: TokenChunker | None = None,
        tagger: MetadataTagger | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cleaner = cleaner or TextCleaner(self.settings)
        self.chunker = chunker or TokenChunker(settings=self.settings)
        self.tagger = tagger or MetadataTagger(self.settings)

    def process(self, documents: Sequence[RawDocument]) -> tuple[list[Chunk], ProcessResult]:
        """Clean, chunk and tag ``documents``.

        Args:
            documents: Raw documents from the loaders.

        Returns:
            ``(chunks, stats)`` — chunks are deduplicated by content.
        """
        started = time.perf_counter()
        stats = ProcessResult(documents_in=len(documents))
        if not documents:
            stats.duration_ms = (time.perf_counter() - started) * 1000
            return [], stats

        try:
            cleaned = self.cleaner.clean_documents(documents)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Cleaning failed; continuing with raw text")
            stats.errors.append(f"cleaning: {exc}")
            cleaned = list(documents)

        stats.documents_out = len(cleaned)
        stats.dropped_documents = stats.documents_in - stats.documents_out

        chunks = self.chunker.chunk_documents(cleaned)
        chunks = self.tagger.tag(chunks)

        # Deduplicate identical chunks (common with repeated headers/footers).
        seen: set[str] = set()
        unique: list[Chunk] = []
        for chunk in chunks:
            key = chunk.dedupe_key
            if key in seen:
                logger.debug("Dropping duplicate chunk %s", chunk.chunk_id)
                continue
            seen.add(key)
            unique.append(chunk)

        stats.chunks = len(unique)
        stats.tokens = sum(chunk.token_count for chunk in unique)
        stats.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Processed %s documents -> %s chunks (%s tokens) in %.0f ms",
            stats.documents_out,
            stats.chunks,
            stats.tokens,
            stats.duration_ms,
        )
        return unique, stats

    def process_and_store(
        self,
        documents: Sequence[RawDocument],
        db: DocumentDB,
        replace_sources: bool = True,
    ) -> tuple[list[Chunk], ProcessResult]:
        """Process documents and persist both raw documents and chunks.

        Args:
            documents: Raw documents.
            db: Document database.
            replace_sources: Delete previously stored chunks for the touched
                sources first (keeps re-uploads consistent).

        Returns:
            ``(chunks, stats)``.
        """
        chunks, stats = self.process(documents)
        db.insert_documents(documents)
        sources = {document.source for document in documents}
        if replace_sources and sources:
            for source in sources:
                with db.connection() as conn:
                    conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        db.insert_chunks(chunks)
        return chunks, stats

    def reprocess_from_db(self, db: DocumentDB) -> tuple[list[Chunk], ProcessResult]:
        """Re-chunk every document stored in the database.

        Useful after changing chunk size/overlap: raw text is preserved in
        SQLite, so only the chunk table needs to be rebuilt.
        """
        documents = list(db.iter_documents())
        db.delete_chunks()
        return self.process_and_store(documents, db, replace_sources=False)


__all__ = ["DocumentProcessor", "ProcessResult"]
