"""Embedding models plugged into LlamaIndex.

Three backends are supported, selected with ``EMBEDDING_BACKEND``:

``sentence_transformers``
    Local multilingual model — defaults to **BAAI/bge-m3**, which handles
    Bangla and English in the same vector space (1024 dimensions).
``openai``
    ``text-embedding-3-small`` (or any OpenAI-compatible embedding endpoint).
``hashing``
    Dependency-free character/word n-gram hashing embedder used offline (CI,
    air-gapped demos, and as an automatic fallback when a model cannot be
    downloaded).

A transparent SQLite embedding cache sits in front of the backend so that
re-indexing unchanged chunks is essentially free.

Example:
    >>> from src.indexing.embeddings import get_embed_model
    >>> embed_model = get_embed_model(backend="hashing", dim=256)
    >>> len(embed_model.get_text_embedding("hello world"))
    256
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import math
import sqlite3
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from llama_index.core.embeddings import BaseEmbedding
from pydantic import ConfigDict, Field, PrivateAttr

from src.config import Settings, get_settings
from src.errors import IndexingError
from src.text_utils import character_ngrams, tokenize_words

logger = logging.getLogger(__name__)

Embedding = list[float]


def detect_device(preference: str = "auto") -> str:
    """Return ``"cuda"``, ``"mps"`` or ``"cpu"`` for the given preference.

    Args:
        preference: ``auto``, ``cpu``, ``cuda`` or ``mps``.

    Returns:
        The resolved device string.
    """
    preference = (preference or "auto").lower()
    if preference in {"cpu", "cuda", "mps"}:
        return preference
    try:  # pragma: no cover - hardware dependent
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # pragma: no cover - torch optional
        pass
    return "cpu"


def _normalise(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


class HashingEmbedding(BaseEmbedding):
    """Offline, deterministic embedding based on hashed word and char n-grams.

    This is **not** a semantic model; it is a robust lexical embedder that keeps
    the whole pipeline runnable without a model download (and serves as the
    automatic fallback if ``bge-m3`` cannot be fetched).  Bengali text is
    handled through character n-grams, which is a strong baseline for the
    language's rich morphology.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embed_dim: int = Field(default=1024, description="Vector dimension")
    min_n: int = Field(default=3, description="Minimum character n-gram size")
    max_n: int = Field(default=5, description="Maximum character n-gram size")
    word_weight: float = Field(default=1.0, description="Weight of whole-word features")
    gram_weight: float = Field(default=0.5, description="Weight of character n-gram features")
    task_prefix: str = Field(default="", description="Optional instruction prefix")

    @classmethod
    def from_settings(cls, settings: Settings) -> HashingEmbedding:
        """Build the fallback embedder from project settings."""
        return cls(
            model_name="hashing-ngram-v1",
            embed_dim=int(settings.embedding_dim),
            min_n=3,
            max_n=5,
        )

    @property
    def dimension(self) -> int:
        """Vector dimension produced by this embedder."""
        return self.embed_dim

    # -------------------------------------------------------------- encoding
    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        index = value % self.embed_dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        return index, sign

    def _encode(self, text: str) -> Embedding:
        vector = np.zeros(self.embed_dim, dtype=np.float32)
        words = tokenize_words(text or "")
        if not words and not (text or "").strip():
            return vector.tolist()

        word_counts: dict[str, int] = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        for word, count in word_counts.items():
            index, sign = self._bucket(f"w:{word}")
            vector[index] += sign * self.word_weight * (1.0 + math.log(count))

        gram_counts: dict[str, int] = {}
        for gram in character_ngrams(text or "", self.min_n, self.max_n):
            gram_counts[gram] = gram_counts.get(gram, 0) + 1
        for gram, count in gram_counts.items():
            index, sign = self._bucket(f"g:{gram}")
            vector[index] += sign * self.gram_weight * (1.0 + math.log(count))

        return _normalise(vector).tolist()

    # ---------------------------------------------------------- llama hooks
    def _get_query_embedding(self, query: str) -> Embedding:
        return self._encode(f"{self.task_prefix}{query}")

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._encode(text)

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return [self._encode(text) for text in texts]

    async def _aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return await asyncio.to_thread(self._get_text_embedding_batch, texts)


class SentenceTransformerEmbedding(BaseEmbedding):
    """Multilingual embedding through ``sentence-transformers`` (default bge-m3)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name_or_path: str = "BAAI/bge-m3"
    device: str = "cpu"
    batch_size: int = 16
    normalize: bool = True
    query_instruction: str = ""
    max_length: int = 512

    _model: Any = PrivateAttr(default=None)
    _dimension: int | None = PrivateAttr(default=None)
    _load_error: str | None = PrivateAttr(default=None)

    @classmethod
    def from_settings(cls, settings: Settings) -> SentenceTransformerEmbedding:
        """Build from project settings (device ``auto`` is resolved)."""
        return cls(
            model_name_or_path=settings.embedding_model,
            device=detect_device(settings.embedding_device),
            batch_size=int(settings.embedding_batch_size),
            normalize=bool(settings.embedding_normalize),
        )

    @property
    def model(self) -> Any:
        """Lazily load (and cache) the underlying SentenceTransformer."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading embedding model %s on %s", self.model_name_or_path, self.device)
                self._model = SentenceTransformer(self.model_name_or_path, device=self.device)
                with contextlib.suppress(Exception):
                    self._model.max_seq_length = min(self.max_length, self._model.max_seq_length)
                self._dimension = int(self._model.get_sentence_embedding_dimension())
            except Exception as exc:
                self._load_error = str(exc)
                raise IndexingError(
                    f"cannot load embedding model {self.model_name_or_path!r}: {exc}. "
                    "Set EMBEDDING_BACKEND=hashing for a fully offline fallback."
                ) from exc
        return self._model

    @property
    def dimension(self) -> int:
        """Embedding dimension of the loaded model."""
        if self._dimension is None:
            _ = self.model
        return int(self._dimension or 1024)

    def _encode(self, texts: Sequence[str], is_query: bool) -> list[Embedding]:
        model = self.model
        prepared = [
            f"{self.query_instruction}{text}" if (is_query and self.query_instruction) else (text or " ")
            for text in texts
        ]
        try:
            vectors = model.encode(
                prepared,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise IndexingError(f"embedding failed: {exc}") from exc
        return [np.asarray(vector, dtype=np.float32).tolist() for vector in vectors]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._encode([query], is_query=True)[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._encode([text], is_query=False)[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return self._encode(texts, is_query=False)

    async def _aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return await asyncio.to_thread(self._get_text_embedding_batch, texts)


class OpenAIEmbedding(BaseEmbedding):
    """Embedding through the OpenAI (or compatible) embeddings API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name_or_path: str = "text-embedding-3-small"
    api_key: str = ""
    base_url: str = ""
    batch_size: int = 64
    configured_dim: int = 0
    normalize: bool = True

    _client: Any = PrivateAttr(default=None)
    _dimension: int | None = PrivateAttr(default=None)

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIEmbedding:
        """Build from project settings."""
        return cls(
            model_name_or_path=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            batch_size=int(settings.embedding_batch_size),
            configured_dim=int(settings.embedding_dim),
            normalize=bool(settings.embedding_normalize),
        )

    @property
    def client(self) -> Any:
        if self._client is None:
            if not self.api_key:
                raise IndexingError("OPENAI_API_KEY is required for the openai embedding backend")
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key, base_url=self.base_url or None)
            except Exception as exc:  # pragma: no cover - import/network dependent
                raise IndexingError(f"cannot create OpenAI client: {exc}") from exc
        return self._client

    def _request(self, texts: Sequence[str]) -> list[Embedding]:
        vectors: list[Embedding] = []
        batch_size = max(1, self.batch_size)
        for start in range(0, len(texts), batch_size):
            batch = [text if text.strip() else " " for text in texts[start : start + batch_size]]
            response = self.client.embeddings.create(model=self.model_name_or_path, input=batch)
            ordered = sorted(response.data, key=lambda item: item.index)
            for item in ordered:
                vector = np.asarray(item.embedding, dtype=np.float32)
                if self.normalize:
                    vector = _normalise(vector)
                vectors.append(vector.tolist())
        if vectors and self._dimension is None:
            self._dimension = len(vectors[0])
        return vectors

    @property
    def dimension(self) -> int:
        """Embedding dimension (probed once, then cached)."""
        if self._dimension is None:
            if self.configured_dim and self.model_name_or_path.startswith("text-embedding"):
                # Known OpenAI dimensions for the small/large v3 families.
                self._dimension = 1536 if "small" in self.model_name_or_path else (
                    3072 if "large" in self.model_name_or_path else self.configured_dim
                )
            else:
                self._dimension = self.configured_dim or len(self._request(["dimension probe"])[0])
        return int(self._dimension)

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._request([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._request([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return self._request(texts)

    async def _aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return await asyncio.to_thread(self._get_text_embedding_batch, texts)


class CachedEmbedding(BaseEmbedding):
    """SQLite-backed cache wrapper that avoids recomputing embeddings."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    inner: BaseEmbedding
    cache_path: str = ":memory:"

    _lock: Any = PrivateAttr(default_factory=threading.RLock)
    _conn: sqlite3.Connection | None = PrivateAttr(default=None)
    _hits: int = PrivateAttr(default=0)
    _misses: int = PrivateAttr(default=0)

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if self.cache_path != ":memory:":
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.cache_path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, dim INTEGER, vector BLOB)"
            )
            self._conn.commit()
        return self._conn

    def _key(self, text: str) -> str:
        signature = f"{type(self.inner).__name__}:{self.wrapped_model_name}:{self.dimension}"
        return hashlib.sha256(f"{signature}\x1f{text}".encode()).hexdigest()

    def _load(self, keys: Sequence[str]) -> dict[str, Embedding]:
        if not keys:
            return {}
        found: dict[str, Embedding] = {}
        conn = self._connection()
        with self._lock:
            for start in range(0, len(keys), 500):
                batch = keys[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", batch
                ).fetchall()
                for key, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32).tolist()
        return found

    def _store(self, items: dict[str, Embedding]) -> None:
        if not items:
            return
        conn = self._connection()
        with self._lock:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vector) VALUES (?, ?, ?)",
                [
                    (key, len(vector), np.asarray(vector, dtype=np.float32).tobytes())
                    for key, vector in items.items()
                ],
            )
            conn.commit()

    @property
    def dimension(self) -> int:
        """Dimension of the wrapped model."""
        return getattr(self.inner, "dimension", 0) or len(self.inner.get_text_embedding("probe"))

    @property
    def wrapped_model_name(self) -> str:
        """Name of the wrapped model (safe for logging/manifest)."""
        inner_name = type(self.inner).__name__
        return str(getattr(self.inner, "model_name_or_path", "") or inner_name)

    @property
    def stats(self) -> dict[str, int]:
        """Cache hit/miss counters."""
        return {"hits": self._hits, "misses": self._misses}

    def _encode(self, texts: Sequence[str], is_query: bool) -> list[Embedding]:
        keys = [self._key(text) for text in texts]
        cached = self._load(keys)
        missing_positions = [i for i, key in enumerate(keys) if key not in cached]
        vectors: list[Embedding | None] = [cached.get(key) for key in keys]

        if missing_positions:
            missing_texts = [texts[i] for i in missing_positions]
            if is_query:
                fresh = [self.inner.get_query_embedding(text) for text in missing_texts]
            else:
                fresh = self.inner.get_text_embedding_batch(missing_texts, show_progress=False)
            to_store: dict[str, Embedding] = {}
            for position, vector in zip(missing_positions, fresh, strict=True):
                vectors[position] = list(vector)
                to_store[keys[position]] = list(vector)
            self._store(to_store)
            self._misses += len(missing_positions)
        self._hits += len(texts) - len(missing_positions)
        return [vector or [] for vector in vectors]

    def _get_query_embedding(self, query: str) -> Embedding:
        return self._encode([query], is_query=True)[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        return await asyncio.to_thread(self._get_query_embedding, query)

    def _get_text_embedding(self, text: str) -> Embedding:
        return self._encode([text], is_query=False)[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        return await asyncio.to_thread(self._get_text_embedding, text)

    def _get_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return self._encode(texts, is_query=False)

    async def _aget_text_embedding_batch(
        self, texts: list[str], show_progress: bool = False, **kwargs: Any
    ) -> list[Embedding]:
        return await asyncio.to_thread(self._get_text_embedding_batch, texts)


def get_embedding_dimension(embed_model: BaseEmbedding) -> int:
    """Return the embedding dimension of any supported embedder."""
    dimension = getattr(embed_model, "dimension", None)
    if isinstance(dimension, int) and dimension > 0:
        return dimension
    probe = embed_model.get_text_embedding("dimension probe")
    return len(probe)


def _build_backend(
    backend: str,
    settings: Settings,
    fallback: Callable[[Exception], BaseEmbedding],
) -> BaseEmbedding:
    try:
        if backend == "sentence_transformers":
            return SentenceTransformerEmbedding.from_settings(settings)
        if backend == "openai":
            return OpenAIEmbedding.from_settings(settings)
        if backend == "hashing":
            return HashingEmbedding.from_settings(settings)
        raise IndexingError(f"unknown embedding backend {backend!r}")
    except Exception as exc:
        if not settings.allow_model_fallbacks:
            raise
        logger.warning("Embedding backend %s unavailable (%s); using fallback", backend, exc)
        return fallback(exc)


def get_embed_model(
    settings: Settings | None = None,
    backend: str | None = None,
    dim: int | None = None,
    use_cache: bool | None = None,
) -> BaseEmbedding:
    """Create the configured embedding model with automatic fallbacks and caching.

    Args:
        settings: Project settings (defaults to :func:`src.config.get_settings`).
        backend: Override for ``EMBEDDING_BACKEND``.
        dim: Override for the hashing fallback dimension.
        use_cache: Override for the on-disk embedding cache.

    Returns:
        A LlamaIndex ``BaseEmbedding`` instance.
    """
    settings = settings or get_settings()
    chosen = (backend or settings.embedding_backend).lower()

    def fallback(exc: Exception) -> BaseEmbedding:
        model = HashingEmbedding.from_settings(settings)
        if dim:
            model.embed_dim = int(dim)
        logger.warning("Falling back to offline hashing embeddings (%s)", exc)
        return model

    model = _build_backend(chosen, settings, fallback)

    # If sentence-transformers fails lazily (during load), retry with hashing.
    if isinstance(model, (SentenceTransformerEmbedding, OpenAIEmbedding)):
        try:
            _ = get_embedding_dimension(model)
        except Exception as exc:
            if not settings.allow_model_fallbacks:
                raise
            logger.warning("Embedding model %s failed to initialise: %s", chosen, exc)
            model = fallback(exc)

    cache_enabled = settings.embedding_cache if use_cache is None else use_cache
    if cache_enabled and not isinstance(model, CachedEmbedding):
        model = CachedEmbedding(inner=model, cache_path=str(settings.embedding_cache_path))
    return model


__all__ = [
    "HashingEmbedding",
    "SentenceTransformerEmbedding",
    "OpenAIEmbedding",
    "CachedEmbedding",
    "get_embed_model",
    "get_embedding_dimension",
    "detect_device",
]
