"""Project-wide exception hierarchy.

Every module raises a subclass of :class:`RAGError`, which lets callers
(Streamlit, CLI, tests) catch a single family of errors while still being able
to discriminate between layers.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all project errors."""


class ConfigurationError(RAGError):
    """Raised when configuration is missing or invalid."""


class IngestionError(RAGError):
    """Raised when the ingestion pipeline cannot process a source."""


class LoaderError(IngestionError):
    """Raised by a document loader (PDF/CSV/URL/text)."""

    def __init__(self, source: str, message: str) -> None:
        super().__init__(f"{source}: {message}")
        self.source = source
        self.message = message


class StorageError(RAGError):
    """Raised when the document database cannot be read or written."""


class IndexingError(RAGError):
    """Raised when an index cannot be built, loaded or persisted."""


class RetrievalError(RAGError):
    """Raised when a retriever fails."""


class RerankError(RAGError):
    """Raised when the reranker cannot score candidates."""


class GenerationError(RAGError):
    """Raised when the LLM backend fails to produce an answer."""


class EvaluationError(RAGError):
    """Raised when the RAGAS evaluation cannot run."""


__all__ = [
    "RAGError",
    "ConfigurationError",
    "IngestionError",
    "LoaderError",
    "StorageError",
    "IndexingError",
    "RetrievalError",
    "RerankError",
    "GenerationError",
    "EvaluationError",
]
