"""Central configuration for the Advanced Multi-Source RAG project.

All tunables are read from environment variables (or a ``.env`` file at the
repository root) and validated with :mod:`pydantic-settings`.  Import
:func:`get_settings` to obtain a cached, immutable-by-convention settings
object instead of reading ``os.environ`` directly.

Example:
    >>> from src.config import get_settings
    >>> settings = get_settings()
    >>> settings.chunk_size_tokens
    600
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root: <repo>/src/config.py -> <repo>
BASE_DIR: Path = Path(__file__).resolve().parent.parent

LLMProvider = Literal["openai", "anthropic", "ollama", "extractive"]
EmbeddingBackend = Literal["sentence_transformers", "openai", "hashing"]
RerankerBackend = Literal["cross_encoder", "cohere", "lexical", "none"]

DEFAULT_LLM_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-latest",
    "ollama": "qwen2.5:7b",
    "extractive": "extractive-offline",
}


class Settings(BaseSettings):
    """Runtime configuration for ingestion, indexing, retrieval and generation.

    Values are loaded from the environment with the field name upper-cased,
    e.g. ``llm_provider`` maps to ``LLM_PROVIDER``.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Advanced Multi-Source RAG"
    log_level: str = "INFO"
    data_dir: Path = Field(default=BASE_DIR / "data")

    # --------------------------------------------------------------- LLM
    llm_provider: LLMProvider = "openai"
    llm_model: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 2
    openai_api_key: str = ""
    openai_base_url: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    ollama_api_key: str = ""

    # ---------------------------------------------------------- embeddings
    embedding_backend: EmbeddingBackend = "sentence_transformers"
    embedding_model: str = "BAAI/bge-m3"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024
    embedding_device: str = "auto"
    embedding_batch_size: int = 16
    embedding_normalize: bool = True
    embedding_cache: bool = True

    # ------------------------------------------------------------- rerank
    reranker_backend: RerankerBackend = "cross_encoder"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "auto"
    reranker_batch_size: int = 8
    reranker_max_length: int = 512
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-multilingual-v3.0"

    # ------------------------------------------------------- chunking etc.
    chunk_size_tokens: int = 600
    chunk_overlap_tokens: int = 80
    sentence_window_size: int = 3
    min_chunk_chars: int = 40
    csv_rows_per_document: int = 6
    max_pdf_pages: int = 0  # 0 == no limit

    # ------------------------------------------------------------ retrieval
    retrieval_top_k: int = 10
    rerank_top_n: int = 5
    retriever_timeout_seconds: float = 60.0
    vector_weight: float = 1.0
    window_weight: float = 1.0
    graph_weight: float = 0.9
    use_rrf: bool = True
    rrf_k: int = 60
    min_rerank_score: float = 0.0  # 0 = backend default relevance gate
    max_context_tokens: int = 3000

    # --------------------------------------------------------------- graph
    graph_max_hops: int = 2
    graph_extraction_batch_size: int = 8
    graph_max_chunks: int = 0  # 0 == no limit
    graph_min_entity_len: int = 2
    graph_max_entities_per_chunk: int = 20

    # ----------------------------------------------------------- behaviour
    allow_model_fallbacks: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8501

    # ----------------------------------------------------------- validators
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper().strip()
        valid = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        if level not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}, got {value!r}")
        return level

    @field_validator("chunk_overlap_tokens")
    @classmethod
    def _validate_overlap(cls, value: int, info: ValidationInfo) -> int:
        size = int(info.data.get("chunk_size_tokens", 600))
        if value < 0:
            raise ValueError("chunk_overlap_tokens must be >= 0")
        if value >= size:
            raise ValueError(
                f"chunk_overlap_tokens ({value}) must be smaller than "
                f"chunk_size_tokens ({size})"
            )
        return value

    @field_validator("embedding_dim")
    @classmethod
    def _validate_dim(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("embedding_dim must be positive")
        return value

    @field_validator("llm_provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: str) -> str:
        return str(value).strip().lower()

    # ------------------------------------------------------------ derived
    @property
    def resolved_llm_model(self) -> str:
        """Return the configured model, or the provider's sensible default."""
        return self.llm_model.strip() or DEFAULT_LLM_MODELS[self.llm_provider]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def db_dir(self) -> Path:
        return self.data_dir / "db"

    @property
    def db_path(self) -> Path:
        return Path(os.getenv("RAG_DB_PATH", str(self.db_dir / "documents.db")))

    @property
    def vector_index_dir(self) -> Path:
        return self.index_dir / "vector"

    @property
    def window_index_dir(self) -> Path:
        return self.index_dir / "sentence_window"

    @property
    def graph_dir(self) -> Path:
        return self.index_dir / "graph"

    @property
    def graph_path(self) -> Path:
        return self.graph_dir / "graph.json"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    @property
    def embedding_cache_path(self) -> Path:
        return self.index_dir / "embedding_cache.db"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "evaluation"

    def ensure_directories(self) -> None:
        """Create every directory the pipeline writes to."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.processed_dir,
            self.index_dir,
            self.db_dir,
            self.vector_index_dir,
            self.window_index_dir,
            self.graph_dir,
            self.eval_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict[str, object]:
        """Return a log-safe view of the settings (secrets masked)."""
        data = self.model_dump()
        for key in ("openai_api_key", "anthropic_api_key", "cohere_api_key", "ollama_api_key"):
            value = str(data.get(key) or "")
            data[key] = f"***{value[-4:]}" if value else ""
        data["resolved_llm_model"] = self.resolved_llm_model
        data["db_path"] = str(self.db_path)
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    settings = Settings()
    settings.ensure_directories()
    return settings


def configure_logging(level: str | None = None) -> None:
    """Configure root logging once, with a compact, readable format."""
    resolved = (level or os.getenv("LOG_LEVEL") or get_settings().log_level).upper()
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))
    # Third-party libraries are chatty at INFO level.
    for noisy in ("httpx", "httpcore", "urllib3", "faiss", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def reset_settings_cache() -> None:
    """Clear the settings cache (used by tests that mutate the environment)."""
    get_settings.cache_clear()
