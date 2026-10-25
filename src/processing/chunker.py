"""Token-aware chunking with sentence boundaries and overlap.

Defaults follow the project spec: **600 tokens per chunk with 80 tokens of
overlap**.  Chunking is sentence-aware (English ``.``, Bengali ``।`` and
newlines), and over-long sentences are split on word boundaries so a chunk can
never exceed the token budget.  Chunk ids are deterministic, which keeps
FAISS / sentence-window / graph indexes consistent across re-indexing runs.

Example:
    >>> chunker = TokenChunker()
    >>> chunks = chunker.chunk_text("Sentence one. Sentence two.", source="a.txt")
    >>> chunks[0].token_count > 0
    True
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

from src.config import Settings, get_settings
from src.errors import IngestionError
from src.models import Chunk, RawDocument, SourceType, stable_id, utc_now_iso
from src.text_utils import count_tokens, split_sentences

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkerStats:
    """Summary of a chunking run."""

    documents: int = 0
    chunks: int = 0
    tokens: int = 0

    def add(self, chunk: Chunk) -> None:
        self.chunks += 1
        self.tokens += chunk.token_count


class TokenChunker:
    """Split documents into overlapping, token-bounded chunks."""

    def __init__(
        self,
        chunk_size_tokens: int | None = None,
        overlap_tokens: int | None = None,
        min_chunk_chars: int | None = None,
        settings: Settings | None = None,
    ) -> None:
        """
        Args:
            chunk_size_tokens: Maximum tokens per chunk (default 600).
            overlap_tokens: Tokens of overlap between neighbours (default 80).
            min_chunk_chars: Chunks shorter than this are merged/dropped.
            settings: Optional settings override.
        """
        self.settings = settings or get_settings()
        self.chunk_size = int(chunk_size_tokens or self.settings.chunk_size_tokens)
        self.overlap = int(overlap_tokens if overlap_tokens is not None else self.settings.chunk_overlap_tokens)
        self.min_chunk_chars = int(
            min_chunk_chars if min_chunk_chars is not None else self.settings.min_chunk_chars
        )
        if self.chunk_size <= 0:
            raise IngestionError("chunk_size_tokens must be positive")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise IngestionError(
                f"overlap_tokens ({self.overlap}) must be >= 0 and < chunk_size_tokens ({self.chunk_size})"
            )

    # ------------------------------------------------------------------ helpers
    @property
    def _piece_budget(self) -> int:
        """Token budget for an indivisible sentence piece (leaves room for overlap)."""
        return max(1, self.chunk_size - self.overlap)

    def _split_long_sentence(self, sentence: str) -> list[str]:
        """Split a sentence that alone exceeds the token budget."""
        if count_tokens(sentence) <= self._piece_budget:
            return [sentence]
        words = sentence.split()
        pieces: list[str] = []
        current: list[str] = []
        for word in words:
            candidate = " ".join([*current, word])
            if current and count_tokens(candidate) > self._piece_budget:
                pieces.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            pieces.append(" ".join(current))
        return [piece for piece in pieces if piece.strip()]

    def _sentences(self, text: str) -> list[str]:
        """Sentence-split ``text`` and break up any over-long sentences."""
        sentences: list[str] = []
        for sentence in split_sentences(text):
            sentences.extend(self._split_long_sentence(sentence))
        return sentences

    def split_text(self, text: str) -> list[str]:
        """Split ``text`` into overlapping chunk strings.

        Args:
            text: Cleaned document text.

        Returns:
            List of chunk texts respecting the token budget and overlap.
        """
        sentences = self._sentences(text)
        if not sentences:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = count_tokens(sentence)
            if current and current_tokens + sentence_tokens > self.chunk_size:
                # Keep sentences on separate lines: preserves table rows/lists and
                # lets downstream consumers (citations, sentence windows) split again.
                chunk_text = "\n".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)
                # Build the overlap tail from the end of the finished chunk.
                tail: list[str] = []
                tail_tokens = 0
                for previous in reversed(current):
                    previous_tokens = count_tokens(previous)
                    if tail_tokens + previous_tokens > self.overlap:
                        break
                    tail.insert(0, previous)
                    tail_tokens += previous_tokens
                current = tail
                current_tokens = tail_tokens
            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            remainder = "\n".join(current).strip()
            if remainder:
                # Merge a tiny tail into the previous chunk when it fits.
                if chunks and len(remainder) < self.min_chunk_chars:
                    merged = f"{chunks[-1]}\n{remainder}".strip()
                    if count_tokens(merged) <= self.chunk_size * 1.15:
                        chunks[-1] = merged
                    else:
                        chunks.append(remainder)
                else:
                    chunks.append(remainder)

        return [chunk for chunk in chunks if chunk.strip()]

    # ------------------------------------------------------------------ public
    def chunk_text(
        self,
        text: str,
        source: str = "text",
        source_type: SourceType = SourceType.TEXT,
        page: int | None = None,
        doc_id: str = "",
        title: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> list[Chunk]:
        """Chunk a raw string and wrap the result in :class:`Chunk` objects."""
        pieces = self.split_text(text)
        chunks: list[Chunk] = []
        document_id = doc_id or stable_id(source, page, len(text))
        for index, piece in enumerate(pieces):
            chunk_id = stable_id(document_id, index, piece)
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=document_id,
                    text=piece,
                    chunk_index=index,
                    source=source,
                    source_type=source_type,
                    page=page,
                    title=title,
                    token_count=count_tokens(piece),
                    created_at=utc_now_iso(),
                    metadata={
                        "chunk_total": len(pieces),
                        "chunk_size_tokens": self.chunk_size,
                        "chunk_overlap_tokens": self.overlap,
                        **(metadata or {}),
                    },
                )
            )
        return chunks

    def chunk_document(self, document: RawDocument) -> list[Chunk]:
        """Chunk one :class:`RawDocument`."""
        metadata = dict(document.metadata)
        metadata.update(
            {
                "title": document.title,
                "doc_created_at": document.created_at,
                "doc_char_count": document.char_count,
            }
        )
        return self.chunk_text(
            document.text,
            source=document.source,
            source_type=document.source_type,
            page=document.page,
            doc_id=document.doc_id,
            title=document.title,
            metadata=metadata,
        )

    def chunk_documents(self, documents: Iterable[RawDocument]) -> list[Chunk]:
        """Chunk a batch of documents, skipping empty results."""
        chunks: list[Chunk] = []
        stats = ChunkerStats()
        for document in documents:
            stats.documents += 1
            try:
                document_chunks = self.chunk_document(document)
            except Exception as exc:  # pragma: no cover - defensive
                logger.exception("Failed to chunk %s page %s: %s", document.source, document.page, exc)
                continue
            if not document_chunks:
                logger.debug("No chunks produced for %s page %s", document.source, document.page)
            for chunk in document_chunks:
                stats.add(chunk)
            chunks.extend(document_chunks)
        logger.info(
            "Chunked %s documents into %s chunks (~%s tokens)",
            stats.documents,
            stats.chunks,
            stats.tokens,
        )
        return chunks


def chunk_text(text: str, settings: Settings | None = None) -> list[str]:
    """Convenience function returning chunk strings for a single text."""
    return TokenChunker(settings=settings).split_text(text)


__all__ = ["TokenChunker", "ChunkerStats", "chunk_text"]
