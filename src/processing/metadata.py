"""Metadata tagging for chunks.

Every chunk carries the metadata the rest of the system needs:

* provenance – ``source``, ``source_type``, ``page``, ``title``, ``doc_id``
* structure  – ``chunk_index``, ``chunk_total``, ``section``
* language   – ``language`` (``en``, ``bn`` or ``mixed``) for bilingual search
* signals    – ``keywords``, ``char_count``, ``token_count``, ``has_numbers``

:func:`build_node_metadata` flattens a chunk's metadata into the primitive
key/value pairs accepted by LlamaIndex node metadata.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Any

from src.config import Settings, get_settings
from src.models import Chunk
from src.text_utils import tokenize_words

logger = logging.getLogger(__name__)

STOPWORDS_EN: frozenset[str] = frozenset(
    ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "its", "itself", "just", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves", "also", "may", "might", "shall", "us", "per", "via", "etc", "e.g", "i.e"]
)

STOPWORDS_BN: frozenset[str] = frozenset(
    ["এবং", "বা", "কিন্তু", "তবে", "যে", "যা", "যার", "যাদের", "এই", "সেই", "এটা", "ওটা", "এখানে", "সেখানে", "তখন", "এখন", "আরও", "আর", "অথবা", "যদি", "তাহলে", "কেন", "কী", "কি", "কোন", "কোনও", "কোনো", "করে", "করা", "করেন", "করছে", "হয়", "হবে", "হয়েছে", "হয়েছিল", "ছিল", "ছিলেন", "থেকে", "জন্য", "সাথে", "মধ্যে", "পরে", "আগে", "সব", "সকল", "একটি", "একটা", "এক", "দুই", "তিন", "আমি", "আমার", "আমরা", "আমাদের", "আপনি", "আপনার", "তোমার", "তার", "তাদের", "তিনি", "তারা", "এটা", "ইহা", "উহা", "না", "নেই", "আছে", "With", "From", "This", "That", "The", "And"]
)

STOPWORDS: frozenset[str] = STOPWORDS_EN | STOPWORDS_BN

_SENTENCE_END_CHARS = "।.!?…:;"
_HEADING_MAX_CHARS = 90
_NUMBER_RE = re.compile(r"\d")


class MetadataTagger:
    """Enrich chunks with derived metadata (language, keywords, section)."""

    def __init__(self, settings: Settings | None = None, keyword_count: int = 8) -> None:
        self.settings = settings or get_settings()
        self.keyword_count = keyword_count

    # ------------------------------------------------------------- detectors
    @staticmethod
    def detect_language(text: str) -> str:
        """Classify text as ``bn``, ``en`` or ``mixed``.

        Args:
            text: Chunk text.

        Returns:
            ``"bn"`` when Bengali dominates, ``"en"`` when Latin dominates and
            ``"mixed"`` when both scripts are well represented.
        """
        if not text:
            return "en"
        bengali = sum(1 for char in text if "\u0980" <= char <= "\u09FF")
        latin = sum(1 for char in text if char.isascii() and char.isalpha())
        total = bengali + latin
        if total == 0:
            return "en"
        bengali_ratio = bengali / total
        if bengali_ratio >= 0.75:
            return "bn"
        if bengali_ratio <= 0.15:
            return "en"
        return "mixed"

    @staticmethod
    def extract_section(text: str) -> str | None:
        """Return the leading heading of a chunk when it looks like one."""
        first_line = (text or "").strip().split("\n", 1)[0].strip()
        if not first_line or len(first_line) > _HEADING_MAX_CHARS:
            return None
        if first_line.endswith(tuple(_SENTENCE_END_CHARS)):
            return None
        if len(tokenize_words(first_line)) > 12:
            return None
        return first_line

    @classmethod
    def extract_keywords(cls, text: str, top_n: int = 8) -> list[str]:
        """Return the most frequent content words of ``text``.

        A deterministic, dependency-free TF heuristic that works for both
        scripts: words shorter than three characters, stopwords and pure numbers
        are ignored.

        Args:
            text: Chunk text.
            top_n: Number of keywords to return.

        Returns:
            Keywords ordered by descending frequency, then first appearance.
        """
        if not text:
            return []
        counts: dict[str, int] = {}
        first_seen: dict[str, int] = {}
        for position, word in enumerate(tokenize_words(text)):
            if len(word) < 3 or word.isdigit() or word in STOPWORDS:
                continue
            counts[word] = counts.get(word, 0) + 1
            first_seen.setdefault(word, position)
        ranked = sorted(counts.items(), key=lambda item: (-item[1], first_seen[item[0]]))
        return [word for word, _ in ranked[:top_n]]

    # ------------------------------------------------------------- tagging
    def tag_chunk(self, chunk: Chunk) -> Chunk:
        """Add derived metadata to ``chunk`` in place and return it."""
        from src.text_utils import count_tokens

        language = self.detect_language(chunk.text)
        keywords = self.extract_keywords(chunk.text, self.keyword_count)
        section = self.extract_section(chunk.text)
        chunk.token_count = chunk.token_count or count_tokens(chunk.text)
        chunk.metadata.update(
            {
                "language": language,
                "keywords": keywords,
                "section": section,
                "char_count": len(chunk.text),
                "has_numbers": bool(_NUMBER_RE.search(chunk.text)),
                "citation": chunk.citation_key,
                "tagger": "heuristic-v1",
            }
        )
        return chunk

    def tag(self, chunks: Iterable[Chunk]) -> list[Chunk]:
        """Tag a batch of chunks."""
        tagged = [self.tag_chunk(chunk) for chunk in chunks]
        logger.debug("Tagged %s chunks with language/keyword metadata", len(tagged))
        return tagged


# LlamaIndex reserves these metadata keys for its own use.
_RESERVED_METADATA_KEYS = {
    "window",
    "original_text",
    "document_id",
    "ref_doc_id",
    "relationships",
    "embedding",
    "text",
    "node_id",
}


def build_node_metadata(chunk: Chunk, max_string_length: int = 400) -> dict[str, Any]:
    """Flatten chunk metadata into JSON-friendly primitives for LlamaIndex.

    ``None`` values are dropped, lists are comma-joined, long strings are
    truncated and reserved keys are removed.

    Args:
        chunk: Processed chunk.
        max_string_length: Maximum length of a string metadata value.

    Returns:
        A flat ``dict`` of primitive metadata values.
    """
    metadata: dict[str, Any] = {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "source_type": str(chunk.source_type),
        "page": chunk.page if chunk.page is not None else -1,
        "title": chunk.title or "",
        "chunk_index": chunk.chunk_index,
        "token_count": chunk.token_count,
        "created_at": chunk.created_at,
    }
    for key, value in chunk.metadata.items():
        if key in _RESERVED_METADATA_KEYS or key in metadata:
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value)
        if isinstance(value, str) and len(value) > max_string_length:
            value = value[:max_string_length]
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def metadata_from_node(node_metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert stored node metadata back into chunk-style metadata.

    ``page`` stored as ``-1`` is converted back to ``None``.
    """
    meta = dict(node_metadata or {})
    if meta.get("page") in (-1, "-1", None):
        meta["page"] = None
    return meta


__all__ = [
    "MetadataTagger",
    "STOPWORDS",
    "STOPWORDS_EN",
    "STOPWORDS_BN",
    "build_node_metadata",
    "metadata_from_node",
]
