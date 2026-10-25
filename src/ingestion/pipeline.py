"""Ingestion pipeline (step 2 → 4 of the architecture).

Flow for every source::

    load (PDF/CSV/URL/text) -> clean -> chunk (600/80) -> tag metadata
        -> insert raw documents + chunks into the Unified Document DB

Uploaded files are first written to ``data/raw/`` so they can be re-processed
without re-uploading.

Example:
    >>> pipeline = IngestionPipeline()               # doctest: +SKIP
    >>> report = pipeline.ingest_files(["data/samples/employees.csv"])  # doctest: +SKIP
    >>> report.chunks > 0                            # doctest: +SKIP
    True
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from src.config import Settings, get_settings
from src.errors import IngestionError, LoaderError
from src.ingestion.loaders import LoaderRegistry
from src.models import Chunk, RawDocument, SourceType, utc_now_iso
from src.processing.processor import DocumentProcessor
from src.storage.document_db import DocumentDB

logger = logging.getLogger(__name__)

_UNSAFE_FILENAME_RE = re.compile(r"[^\w.\- ()\u0980-\u09FF]+", re.UNICODE)
_URL_LIST_RE = re.compile(r"[\s,]+")


def safe_filename(filename: str) -> str:
    """Sanitise an uploaded file name (keeps Unicode letters, drops paths)."""
    name = Path(str(filename or "upload")).name
    name = _UNSAFE_FILENAME_RE.sub("_", name).strip(" ._")
    if not name:
        name = "upload"
    if len(name) > 120:
        stem, suffix = Path(name).stem[:100], Path(name).suffix[:10]
        name = f"{stem}{suffix}"
    return name


@dataclass(slots=True)
class SourceSummary:
    """Per-source ingestion result."""

    source: str
    source_type: str
    status: str
    documents: int = 0
    chunks: int = 0
    characters: int = 0
    message: str = ""


@dataclass(slots=True)
class IngestionReport:
    """Aggregate result of an ingestion run."""

    sources: list[str] = field(default_factory=list)
    documents: int = 0
    chunks: int = 0
    tokens: int = 0
    new_documents: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)
    details: list[SourceSummary] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)

    @property
    def ok(self) -> bool:
        """``True`` when at least one source produced chunks."""
        return self.chunks > 0

    def to_dict(self) -> dict[str, object]:
        return {
            "sources": self.sources,
            "documents": self.documents,
            "chunks": self.chunks,
            "tokens": self.tokens,
            "new_documents": self.new_documents,
            "duration_ms": round(self.duration_ms, 2),
            "errors": self.errors,
            "details": [
                {
                    "source": detail.source,
                    "source_type": detail.source_type,
                    "status": detail.status,
                    "documents": detail.documents,
                    "chunks": detail.chunks,
                    "characters": detail.characters,
                    "message": detail.message,
                }
                for detail in self.details
            ],
            "created_at": self.created_at,
        }


class IngestionPipeline:
    """Load, process and persist documents from files, URLs or pasted text."""

    def __init__(
        self,
        settings: Settings | None = None,
        db: DocumentDB | None = None,
        registry: LoaderRegistry | None = None,
        processor: DocumentProcessor | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        self.db = db or DocumentDB(settings=self.settings)
        self.registry = registry or LoaderRegistry(self.settings)
        self.processor = processor or DocumentProcessor(self.settings)

    # ------------------------------------------------------------------ files
    def save_upload(self, filename: str, data: bytes) -> Path:
        """Persist uploaded bytes under ``data/raw`` and return the path.

        Args:
            filename: Original file name from the browser.
            data: File content.

        Returns:
            Path of the stored file.

        Raises:
            IngestionError: When the file is empty or cannot be written.
        """
        if not data:
            raise IngestionError(f"{filename}: uploaded file is empty")
        name = safe_filename(filename)
        target = self.settings.raw_dir / name
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise IngestionError(f"{filename}: cannot save upload ({exc})") from exc
        logger.info("Saved upload %s (%s bytes)", target.name, len(data))
        return target

    def ingest_files(
        self,
        paths: Sequence[Path | str],
        replace: bool = True,
    ) -> IngestionReport:
        """Ingest local files (PDF/CSV/TXT/MD).

        Args:
            paths: File paths.
            replace: Replace previously stored chunks for the same source.

        Returns:
            An :class:`IngestionReport` (errors are collected, not raised).
        """
        started = time.perf_counter()
        report = IngestionReport()
        grouped: dict[str, list[RawDocument]] = {}

        for path in paths:
            path = Path(path)
            try:
                documents = self.registry.load(path)
            except LoaderError as exc:
                self._fail(report, path.name, self._guess_type(path.name), str(exc))
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Unexpected error loading %s", path)
                self._fail(report, path.name, self._guess_type(path.name), str(exc))
                continue
            grouped.setdefault(documents[0].source, []).extend(documents)

        self._process_grouped(grouped, report, replace=replace)
        report.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Ingested %s file(s): %s documents, %s chunks in %.0f ms",
            len(paths),
            report.documents,
            report.chunks,
            report.duration_ms,
        )
        return report

    def ingest_urls(self, urls: Iterable[str], replace: bool = True) -> IngestionReport:
        """Ingest one or more web pages.

        Args:
            urls: URL strings (comma/space separated strings are accepted).
            replace: Replace previously stored chunks for the same source.

        Returns:
            An :class:`IngestionReport`.
        """
        started = time.perf_counter()
        report = IngestionReport()
        grouped: dict[str, list[RawDocument]] = {}

        cleaned_urls: list[str] = []
        for raw in urls:
            for candidate in _URL_LIST_RE.split(str(raw).strip()):
                if not candidate:
                    continue
                parsed = urlparse(candidate)
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    self._fail(report, candidate, SourceType.URL.value, "invalid URL")
                    continue
                cleaned_urls.append(candidate)

        for url in cleaned_urls:
            try:
                documents = self.registry.url_loader.load(url)
            except LoaderError as exc:
                self._fail(report, url, SourceType.URL.value, str(exc))
                continue
            except Exception as exc:  # pragma: no cover - network dependent
                logger.exception("Unexpected error loading %s", url)
                self._fail(report, url, SourceType.URL.value, str(exc))
                continue
            grouped.setdefault(documents[0].source, []).extend(documents)

        self._process_grouped(grouped, report, replace=replace)
        report.duration_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "Ingested %s URL(s): %s documents, %s chunks in %.0f ms",
            len(cleaned_urls),
            report.documents,
            report.chunks,
            report.duration_ms,
        )
        return report

    def ingest_text(self, text: str, name: str = "pasted_text.txt") -> IngestionReport:
        """Ingest raw pasted text as a text document."""
        started = time.perf_counter()
        report = IngestionReport()
        if not (text or "").strip():
            self._fail(report, name, SourceType.TEXT.value, "empty text")
            return report
        document = RawDocument(
            source=safe_filename(name),
            source_type=SourceType.TEXT,
            text=text,
            title=Path(name).stem,
            metadata={"loader": "pasted_text"},
        )
        self._process_grouped({document.source: [document]}, report, replace=True)
        report.duration_ms = (time.perf_counter() - started) * 1000
        return report

    # ------------------------------------------------------------- internals
    @staticmethod
    def _guess_type(source: str) -> str:
        suffix = Path(str(source).split("?")[0]).suffix.lower()
        return {
            ".pdf": SourceType.PDF.value,
            ".csv": SourceType.CSV.value,
            ".tsv": SourceType.CSV.value,
            ".txt": SourceType.TEXT.value,
            ".md": SourceType.TEXT.value,
        }.get(suffix, SourceType.URL.value if "://" in str(source) else SourceType.TEXT.value)

    def _fail(self, report: IngestionReport, source: str, source_type: str, message: str) -> None:
        logger.error("Ingestion failed for %s: %s", source, message)
        report.errors.append(f"{source}: {message}")
        report.details.append(
            SourceSummary(
                source=source,
                source_type=source_type,
                status="error",
                message=message,
            )
        )
        try:
            self.db.log_ingestion(source, source_type, "error", message)
        except Exception:  # pragma: no cover - logging must never break ingestion
            logger.debug("Could not write ingestion log entry", exc_info=True)

    def _process_grouped(
        self,
        grouped: dict[str, list[RawDocument]],
        report: IngestionReport,
        replace: bool = True,
    ) -> None:
        """Process grouped documents and record per-source summaries."""
        for source, documents in grouped.items():
            source_type = str(documents[0].source_type)
            try:
                chunks, inserted = self._store_source(source, documents, replace=replace)
            except Exception as exc:
                logger.exception("Processing failed for %s", source)
                self._fail(report, source, source_type, str(exc))
                continue

            characters = sum(document.char_count for document in documents)
            report.sources.append(source)
            report.documents += len(documents)
            report.new_documents += inserted
            report.chunks += len(chunks)
            report.tokens += sum(chunk.token_count for chunk in chunks)
            report.details.append(
                SourceSummary(
                    source=source,
                    source_type=source_type,
                    status="ok",
                    documents=len(documents),
                    chunks=len(chunks),
                    characters=characters,
                )
            )
            try:
                self.db.log_ingestion(
                    source, source_type, "ok", "", len(documents), len(chunks)
                )
            except Exception:  # pragma: no cover
                logger.debug("Could not write ingestion log entry", exc_info=True)

    def _store_source(
        self, source: str, documents: list[RawDocument], replace: bool
    ) -> tuple[list[Chunk], int]:
        """Store one source's documents and chunks, returning the chunks."""
        if replace:
            # Re-uploading a changed file must not leave stale rows behind.
            existing = self.db.list_documents(source=source, limit=10_000)
            existing_ids = {document.doc_id for document in existing}
            new_ids = {document.doc_id for document in documents}
            if existing_ids - new_ids:
                logger.info(
                    "Source %s changed; replacing %s stale document(s)",
                    source,
                    len(existing_ids - new_ids),
                )
                self.db.delete_source(source)

        chunks, _stats = self.processor.process(documents)
        inserted = self.db.insert_documents(documents)
        if replace:
            with self.db.connection() as conn:
                conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
        self.db.insert_chunks(chunks)
        logger.info("Stored %s: %s documents (%s new), %s chunks", source, len(documents), inserted, len(chunks))
        return chunks, inserted


__all__ = [
    "IngestionPipeline",
    "IngestionReport",
    "SourceSummary",
    "safe_filename",
]
