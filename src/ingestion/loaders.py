"""Document loaders for PDF, CSV, URL and plain-text sources.

Each loader converts an external artefact into a list of
:class:`~src.models.RawDocument` objects with consistent metadata
(``source``, ``source_type``, ``page``, ``title``, ``created_at``).

Example:
    >>> from pathlib import Path
    >>> from src.ingestion.loaders import PDFLoader
    >>> loader = PDFLoader()
    >>> docs = loader.load(Path("data/samples/handbook.pdf"))
    >>> docs[0].page
    1
"""

from __future__ import annotations

import abc
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import Settings, get_settings
from src.errors import LoaderError
from src.models import RawDocument, SourceType
from src.text_utils import normalise_whitespace

logger = logging.getLogger(__name__)

URL_SCHEMES = {"http", "https"}
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 EnterpriseRAG/1.0"
)


@dataclass(frozen=True)
class LoaderResult:
    """Outcome of loading one source, including non-fatal warnings."""

    documents: list[RawDocument]
    warnings: list[str]
    source: str
    source_type: SourceType

    @property
    def char_count(self) -> int:
        return sum(doc.char_count for doc in self.documents)


class BaseLoader(abc.ABC):
    """Abstract base class for every loader."""

    source_type: SourceType
    suffixes: tuple[str, ...] = ()

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def supports(self, path: Path | str) -> bool:
        """Return ``True`` when this loader handles the given file suffix."""
        return Path(path).suffix.lower() in self.suffixes

    @abc.abstractmethod
    def load(self, source: Path | str) -> list[RawDocument]:
        """Load ``source`` and return raw documents.

        Raises:
            LoaderError: When the source cannot be parsed.
        """

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _require_file(path: Path) -> Path:
        if not path.exists():
            raise LoaderError(str(path), "file does not exist")
        if path.is_dir():
            raise LoaderError(str(path), "expected a file, got a directory")
        if path.stat().st_size == 0:
            raise LoaderError(str(path), "file is empty")
        return path


class PDFLoader(BaseLoader):
    """Extract text page-by-page from PDF files using PyMuPDF (``pymupdf``)."""

    source_type = SourceType.PDF
    suffixes = (".pdf",)

    def load(self, source: Path | str) -> list[RawDocument]:
        """Load a PDF and return one :class:`RawDocument` per non-empty page."""
        path = self._require_file(Path(source))
        try:
            import pymupdf  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - fallback for older installs
            import fitz as pymupdf  # type: ignore[no-redef]

        try:
            pdf = pymupdf.open(path)
        except Exception as exc:
            raise LoaderError(str(path), f"cannot open PDF ({exc})") from exc

        documents: list[RawDocument] = []
        warnings: list[str] = []
        try:
            if getattr(pdf, "needs_pass", False):
                raise LoaderError(str(path), "PDF is password protected")

            meta: dict[str, Any] = dict(getattr(pdf, "metadata", {}) or {})
            title = (meta.get("title") or "").strip() or path.stem
            page_count = int(pdf.page_count)
            limit = self.settings.max_pdf_pages or page_count
            if limit < page_count:
                logger.info("PDF %s: limiting to %s of %s pages", path.name, limit, page_count)

            for page_index in range(min(page_count, limit)):
                try:
                    page = pdf.load_page(page_index)
                    raw_text = page.get_text("text") or ""
                except Exception as exc:  # pragma: no cover - corrupt page
                    warnings.append(f"page {page_index + 1} unreadable: {exc}")
                    logger.warning("PDF %s page %s failed: %s", path.name, page_index + 1, exc)
                    continue

                text = normalise_whitespace(raw_text)
                if not text:
                    continue
                documents.append(
                    RawDocument(
                        source=path.name,
                        source_type=SourceType.PDF,
                        text=text,
                        page=page_index + 1,
                        title=title,
                        metadata={
                            "loader": "pdf",
                            "path": str(path),
                            "page_count": page_count,
                            "author": (meta.get("author") or "").strip(),
                            "subject": (meta.get("subject") or "").strip(),
                            "file_size": path.stat().st_size,
                        },
                    )
                )
        finally:
            pdf.close()

        if not documents:
            raise LoaderError(
                str(path),
                "no extractable text found (scanned/image-only PDF requires OCR)",
            )
        if warnings:
            logger.warning("PDF %s produced %s warnings", path.name, len(warnings))
        logger.info("Loaded PDF %s (%s pages, %s docs)", path.name, len(documents), len(documents))
        return documents


class CSVLoader(BaseLoader):
    """Turn CSV/TSV tables into text blocks of ``csv_rows_per_document`` rows."""

    source_type = SourceType.CSV
    suffixes = (".csv", ".tsv")

    _ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

    def _read_frame(self, path: Path):  # type: ignore[no-untyped-def]
        import pandas as pd

        sep = "\t" if path.suffix.lower() == ".tsv" else None
        last_error: Exception | None = None
        for encoding in self._ENCODINGS:
            for candidate_sep in (sep, ","):
                try:
                    kwargs: dict[str, Any] = {
                        "encoding": encoding,
                        "dtype": str,
                        "keep_default_na": False,
                        "na_filter": False,
                        "on_bad_lines": "warn",
                    }
                    if candidate_sep is None:
                        kwargs.update(engine="python", sep=None)
                    else:
                        kwargs.update(sep=candidate_sep, engine="python")
                    frame = pd.read_csv(path, **kwargs)
                    if frame.shape[1] == 0:
                        raise ValueError("no columns detected")
                    frame.columns = [str(column).strip() or f"col_{i}" for i, column in enumerate(frame.columns)]
                    return frame
                except UnicodeDecodeError as exc:
                    last_error = exc
                    break  # try the next encoding
                except Exception as exc:
                    last_error = exc
                    continue
        raise LoaderError(str(path), f"cannot parse CSV ({last_error})")

    def load(self, source: Path | str) -> list[RawDocument]:
        """Load a CSV/TSV file and return row-block documents."""
        path = self._require_file(Path(source))
        frame = self._read_frame(path)
        if frame.empty:
            raise LoaderError(str(path), "CSV contains no data rows")

        frame = frame.dropna(how="all")
        total_rows = int(len(frame))
        columns = [str(column) for column in frame.columns]
        block_size = max(1, int(self.settings.csv_rows_per_document))

        documents: list[RawDocument] = []
        for block_index, start in enumerate(range(0, total_rows, block_size)):
            block = frame.iloc[start : start + block_size]
            lines = [
                f"Table: {path.name} (rows {start + 1}-{start + len(block)} of {total_rows})",
                f"Columns: {', '.join(columns)}",
                "",
            ]
            for offset, (_, row) in enumerate(block.iterrows()):
                cells = []
                for column in columns:
                    value = normalise_whitespace(str(row[column]))
                    if value:
                        cells.append(f"{column}: {value}")
                if cells:
                    row_text = f"Row {start + offset + 1}: " + " | ".join(cells)
                    if not row_text.rstrip().endswith((".", "!", "?", "।")):
                        row_text = row_text.rstrip() + "."
                    lines.append(row_text)

            text = "\n".join(lines).strip()
            if len(text) < self.settings.min_chunk_chars:
                continue
            documents.append(
                RawDocument(
                    source=path.name,
                    source_type=SourceType.CSV,
                    text=text,
                    page=block_index + 1,
                    title=path.stem,
                    metadata={
                        "loader": "csv",
                        "path": str(path),
                        "columns": columns,
                        "row_start": start + 1,
                        "row_end": start + len(block),
                        "total_rows": total_rows,
                    },
                )
            )

        if not documents:
            raise LoaderError(str(path), "CSV produced no usable text")
        logger.info("Loaded CSV %s (%s rows -> %s docs)", path.name, total_rows, len(documents))
        return documents


class URLLoader(BaseLoader):
    """Fetch a web page and extract its main content with trafilatura."""

    source_type = SourceType.URL
    suffixes = (".html", ".htm")

    def _fetch(self, url: str) -> str:
        try:
            import trafilatura

            html = trafilatura.fetch_url(url)
            if html:
                return html
            logger.debug("trafilatura.fetch_url returned no content for %s", url)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning("trafilatura fetch failed for %s: %s", url, exc)

        try:  # pragma: no cover - network dependent
            import httpx

            response = httpx.get(
                url,
                follow_redirects=True,
                timeout=30.0,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.text
        except Exception as exc:
            raise LoaderError(url, f"failed to download page ({exc})") from exc

    @staticmethod
    def _extract(html: str, url: str | None) -> tuple[str, str, str | None]:
        """Return ``(text, title, date)`` extracted from HTML."""
        import trafilatura

        text = ""
        title = ""
        date: str | None = None
        try:
            doc = trafilatura.bare_extraction(
                html,
                url=url,
                with_metadata=True,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
            if doc is not None:
                text = (getattr(doc, "text", "") or "") if not isinstance(doc, dict) else (doc.get("text") or "")
                title = (getattr(doc, "title", "") or "") if not isinstance(doc, dict) else (doc.get("title") or "")
                date = (getattr(doc, "date", None)) if not isinstance(doc, dict) else doc.get("date")
        except Exception as exc:  # pragma: no cover - parser dependent
            logger.debug("bare_extraction failed: %s", exc)

        if not text.strip():
            try:
                text = trafilatura.extract(
                    html,
                    include_comments=False,
                    include_tables=True,
                    favor_recall=True,
                ) or ""
            except Exception as exc:  # pragma: no cover - parser dependent
                logger.debug("trafilatura.extract failed: %s", exc)

        if not text.strip():  # last resort: strip tags
            try:
                from lxml import html as lxml_html

                tree = lxml_html.fromstring(html)
                for bad in tree.xpath("//script|//style|//noscript"):
                    bad.getparent().remove(bad)
                text = tree.text_content()
            except Exception:
                text = ""

        return normalise_whitespace(text), (title or "").strip(), str(date) if date else None

    def load_html(self, html: str, url: str, source_name: str | None = None) -> list[RawDocument]:
        """Load documents from an already-downloaded HTML string."""
        if not html or not html.strip():
            raise LoaderError(url, "empty HTML payload")
        text, title, published = self._extract(html, url)
        if len(text) < self.settings.min_chunk_chars:
            raise LoaderError(url, "no main content could be extracted from the page")
        host = urlparse(url).netloc or source_name or url
        return [
            RawDocument(
                source=source_name or url,
                source_type=SourceType.URL,
                text=text,
                page=None,
                title=title or host,
                metadata={
                    "loader": "url",
                    "url": url,
                    "host": host,
                    "published": published,
                },
            )
        ]

    def load(self, source: Path | str) -> list[RawDocument]:
        """Download ``source`` (an http/https URL) and extract its main content."""
        url = str(source).strip()
        parsed = urlparse(url)
        if parsed.scheme.lower() not in URL_SCHEMES or not parsed.netloc:
            raise LoaderError(url, "invalid URL (expected http:// or https://)")

        local = Path(url)
        if local.is_file() and local.suffix.lower() in self.suffixes:
            html = local.read_text(encoding="utf-8", errors="replace")
            return self.load_html(html, url=url, source_name=local.name)

        html = self._fetch(url)
        documents = self.load_html(html, url=url)
        logger.info("Loaded URL %s (%s chars)", url, len(documents[0].text))
        return documents


class TextLoader(BaseLoader):
    """Load ``.txt``/``.md`` files, splitting on double newlines when large."""

    source_type = SourceType.TEXT
    suffixes = (".txt", ".md", ".markdown", ".log")

    def load(self, source: Path | str) -> list[RawDocument]:
        """Load a plain-text file as a single document."""
        path = self._require_file(Path(source))
        text = ""
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = path.read_text(encoding=encoding)
                break
            except UnicodeDecodeError:
                continue
        text = normalise_whitespace(text)
        if not text:
            raise LoaderError(str(path), "file has no text content")
        return [
            RawDocument(
                source=path.name,
                source_type=SourceType.TEXT,
                text=text,
                page=None,
                title=path.stem,
                metadata={"loader": "text", "path": str(path)},
            )
        ]


class LoaderRegistry:
    """Dispatch files to the correct loader based on their suffix."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.loaders: list[BaseLoader] = [
            PDFLoader(self.settings),
            CSVLoader(self.settings),
            TextLoader(self.settings),
        ]
        self.url_loader = URLLoader(self.settings)

    @property
    def supported_suffixes(self) -> list[str]:
        """All file suffixes accepted by :meth:`load`."""
        return sorted({suffix for loader in self.loaders for suffix in loader.suffixes})

    def get_loader(self, path: Path | str) -> BaseLoader:
        """Return the loader registered for ``path``'s suffix."""
        for loader in self.loaders:
            if loader.supports(path):
                return loader
        raise LoaderError(str(path), f"unsupported file type (supported: {', '.join(self.supported_suffixes)})")

    def load(self, path: Path | str) -> list[RawDocument]:
        """Load a local file with the matching loader."""
        return self.get_loader(path).load(path)

    def load_many(self, paths: Iterable[Path | str]) -> tuple[list[RawDocument], list[str]]:
        """Load several files, collecting errors instead of aborting."""
        documents: list[RawDocument] = []
        errors: list[str] = []
        for path in paths:
            try:
                documents.extend(self.load(path))
            except LoaderError as exc:
                logger.error("Loader failed: %s", exc)
                errors.append(str(exc))
            except Exception as exc:  # defensive: never let one file kill a batch
                logger.exception("Unexpected loader error for %s", path)
                errors.append(f"{path}: {exc}")
        return documents, errors

    def load_urls(self, urls: Iterable[str]) -> tuple[list[RawDocument], list[str]]:
        """Load several URLs, collecting per-URL errors."""
        documents: list[RawDocument] = []
        errors: list[str] = []
        for url in urls:
            try:
                documents.extend(self.url_loader.load(url))
            except LoaderError as exc:
                logger.error("URL loader failed: %s", exc)
                errors.append(str(exc))
            except Exception as exc:
                logger.exception("Unexpected URL loader error for %s", url)
                errors.append(f"{url}: {exc}")
        return documents, errors


def load_path(path: Path | str, settings: Settings | None = None) -> list[RawDocument]:
    """Convenience wrapper: load a single local file."""
    return LoaderRegistry(settings).load(path)


def load_url(url: str, settings: Settings | None = None) -> list[RawDocument]:
    """Convenience wrapper: load a single URL."""
    return LoaderRegistry(settings).url_loader.load(url)


__all__ = [
    "LoaderResult",
    "BaseLoader",
    "PDFLoader",
    "CSVLoader",
    "URLLoader",
    "TextLoader",
    "LoaderRegistry",
    "load_path",
    "load_url",
]
