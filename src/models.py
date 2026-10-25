"""Typed domain models shared by every layer of the pipeline.

The pipeline has four data shapes:

``RawDocument``  -> one page of a PDF, one CSV row block, one web page, ...
``Chunk``        -> a token-bounded slice of a raw document plus metadata
``RetrievedContext`` -> a chunk returned by a retriever (post merge/rerank)
``RAGAnswer``    -> the final answer, its citations and diagnostics

Everything is dataclass based so it can be serialised to JSON/SQLite without
depending on any specific framework.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def utc_now_iso() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def stable_id(*parts: object, length: int = 24) -> str:
    """Build a deterministic, collision-resistant identifier from ``parts``."""
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def normalise_for_dedupe(text: str) -> str:
    """Normalise text for deduplication comparisons."""
    return _WHITESPACE_RE.sub(" ", text or "").strip().lower()


class SourceType(str, Enum):  # noqa: UP042 - StrEnum needs Python 3.11+
    """Kinds of sources supported by the ingestion pipeline."""

    PDF = "pdf"
    CSV = "csv"
    URL = "url"
    TEXT = "text"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


@dataclass(slots=True)
class RawDocument:
    """A unit of raw text as produced by a loader.

    Attributes:
        doc_id: Stable identifier derived from source, page and content hash.
        source: Human readable source name (file name or URL).
        source_type: One of :class:`SourceType`.
        text: Extracted raw text.
        page: 1-based page (PDF), partition number (CSV) or ``None`` for URLs.
        title: Document title when known.
        created_at: Ingestion timestamp (UTC, ISO-8601).
        metadata: Free-form extra metadata (loader name, row numbers, ...).
    """

    source: str
    source_type: SourceType
    text: str
    page: int | None = None
    title: str | None = None
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)
        if not self.doc_id:
            content_hash = hashlib.sha256((self.text or "").encode("utf-8")).hexdigest()[:16]
            self.doc_id = stable_id(self.source, self.page, content_hash)

    @property
    def char_count(self) -> int:
        return len(self.text or "")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = str(self.source_type)
        return data


@dataclass(slots=True)
class Chunk:
    """A cleaned, token-bounded slice of a :class:`RawDocument`."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    source: str
    source_type: SourceType
    page: int | None = None
    title: str | None = None
    token_count: int = 0
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.source_type, str):
            self.source_type = SourceType(self.source_type)

    @property
    def citation_key(self) -> str:
        """Return the canonical ``[file, page]`` citation key."""
        return format_citation_key(self.source, self.page)

    @property
    def dedupe_key(self) -> str:
        return stable_id(normalise_for_dedupe(self.text))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_type"] = str(self.source_type)
        return data


def format_citation_key(source: str, page: int | None) -> str:
    """Format a citation key such as ``[handbook.pdf, p.4]``.

    Args:
        source: File name or URL.
        page: 1-based page number, or ``None`` when the source has no pages.

    Returns:
        A human readable citation key without markdown decoration.
    """
    name = (source or "unknown").strip()
    if page is None:
        return f"{name}"
    return f"{name}, p.{page}"


@dataclass(slots=True)
class RetrievedContext:
    """A retrieved chunk with scoring and provenance information."""

    node_id: str
    text: str
    score: float = 0.0
    source: str = ""
    page: int | None = None
    source_type: str = "text"
    retriever: str = ""
    retrievers: list[str] = field(default_factory=list)
    rank: int = 0
    rerank_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation_key(self) -> str:
        return format_citation_key(self.source, self.page)

    @property
    def dedupe_key(self) -> str:
        return stable_id(normalise_for_dedupe(self.text))

    def merge_from(self, other: RetrievedContext) -> None:
        """Merge provenance/score information from a duplicate context."""
        for name in other.retrievers:
            if name not in self.retrievers:
                self.retrievers.append(name)
        if other.retriever and other.retriever not in self.retrievers:
            self.retrievers.append(other.retriever)
        self.score = max(self.score, other.score)
        self.rerank_score = (
            other.rerank_score
            if self.rerank_score is None
            else max(self.rerank_score, other.rerank_score or float("-inf"))
        )
        for key, value in other.metadata.items():
            self.metadata.setdefault(key, value)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["citation_key"] = self.citation_key
        return data


@dataclass(slots=True)
class Citation:
    """A resolved, verifiable citation used in the final response."""

    source: str
    page: int | None = None
    source_type: str = "text"
    snippet: str = ""
    score: float | None = None

    @property
    def key(self) -> str:
        return format_citation_key(self.source, self.page)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["key"] = self.key
        return data


@dataclass(slots=True)
class RAGAnswer:
    """Final answer plus everything needed to render and audit it."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[RetrievedContext] = field(default_factory=list)
    query: str = ""
    timings_ms: dict[str, float] = field(default_factory=dict)
    retriever_stats: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    grounded: bool = True

    @property
    def citation_keys(self) -> list[str]:
        return [citation.key for citation in self.citations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "contexts": [c.to_dict() for c in self.contexts],
            "timings_ms": self.timings_ms,
            "retriever_stats": self.retriever_stats,
            "errors": self.errors,
            "provider": self.provider,
            "model": self.model,
            "grounded": self.grounded,
        }
