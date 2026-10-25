"""Text cleaning utilities.

Cleaning happens in two phases:

1. **Text level** – Unicode normalisation, control-character removal, PDF
   hyphenation repair, boilerplate/page-number removal, paragraph dedupe.
2. **Corpus level** – lines repeated on most pages of the same document
   (running headers/footers) are removed across all pages.

The cleaner is intentionally conservative: it never drops a line that carries
content, only obvious furniture (page numbers, "Confidential", etc.).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence

from src.config import Settings, get_settings
from src.models import RawDocument
from src.text_utils import normalise_whitespace

logger = logging.getLogger(__name__)

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")
_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page|পৃষ্ঠা|pg\.?)\s*\d+(?:\s*(?:of|/|এর)\s*\d+)?\s*$",
    re.IGNORECASE,
)
_BARE_NUMBER_RE = re.compile(r"^\s*[-–—]?\s*\d{1,4}\s*[-–—]?\s*$")
_ROMAN_RE = re.compile(r"^\s*[ivxlcdm]{1,7}\s*$", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[•·▪◦‣∙*]\s*")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")

# Short, furniture-like lines that can safely be dropped when they repeat.
_BOILERPLATE_HINTS = (
    "confidential",
    "all rights reserved",
    "copyright",
    "internal use only",
    "do not distribute",
    "সর্বস্বত্ব সংরক্ষিত",
    "গোপনীয়",
)

_MAX_HEADER_LINE_LEN = 120

_MD_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_MD_EMPHASIS_RE = re.compile(r"(\*\*|__)(.*?)\1")
_MD_CODE_RE = re.compile(r"`([^`]*)`")
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)


class TextCleaner:
    """Configurable cleaner for raw document text."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    # ------------------------------------------------------------- primitives
    @staticmethod
    def strip_control_characters(text: str) -> str:
        """Remove non-printable control characters while keeping newlines/tabs."""
        return _CONTROL_RE.sub("", text or "")

    @staticmethod
    def fix_hyphenation(text: str) -> str:
        """Rejoin words broken across lines with a hyphen, e.g. ``multi-\\nsource``."""
        return _HYPHEN_BREAK_RE.sub(r"\1\2", text or "")

    @staticmethod
    def remove_urls(text: str) -> str:
        """Strip http(s) URLs (keeps citations clean)."""
        return _URL_RE.sub(" ", text or "")

    @staticmethod
    def remove_emails(text: str) -> str:
        """Strip e-mail addresses."""
        return _EMAIL_RE.sub(" ", text or "")

    @staticmethod
    def is_noise_line(line: str) -> bool:
        """Return ``True`` for lines that are pure page furniture.

        Detects page numbers ("Page 4", "পৃষ্ঠা ৪"), bare numbers, roman
        numerals and very short punctuation-only lines.
        """
        stripped = (line or "").strip()
        if not stripped:
            return True
        if len(stripped) <= 3 and not any(char.isalnum() for char in stripped):
            return True
        if _PAGE_NUMBER_RE.match(stripped):
            return True
        if _BARE_NUMBER_RE.match(stripped) and len(stripped) <= 6:
            return True
        return bool(_ROMAN_RE.match(stripped) and len(stripped) <= 7)

    @staticmethod
    def strip_markdown(text: str) -> str:
        """Convert light markdown (headings, links, emphasis, code) to plain text."""
        if not text:
            return ""
        text = _MD_LINK_RE.sub(r"\1", text)
        text = _MD_HEADING_RE.sub("", text)
        text = _MD_EMPHASIS_RE.sub(r"\2", text)
        text = _MD_CODE_RE.sub(r"\1", text)
        text = _MD_BULLET_RE.sub("", text)
        return text

    @staticmethod
    def strip_bullets(line: str) -> str:
        """Remove a leading bullet glyph so text reads like a sentence."""
        return _BULLET_RE.sub("", line or "")

    # --------------------------------------------------------------- cleaning
    def clean(self, text: str, remove_urls: bool = False, strip_markdown: bool = False) -> str:
        """Clean a single text block.

        Args:
            text: Raw text.
            remove_urls: Also strip URLs and e-mail addresses.
            strip_markdown: Convert markdown syntax to plain text.

        Returns:
            Cleaned text with normalised whitespace and no furniture lines.
        """
        if not text:
            return ""
        cleaned = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
        cleaned = self.strip_control_characters(cleaned)
        cleaned = self.fix_hyphenation(cleaned)
        if strip_markdown:
            cleaned = self.strip_markdown(cleaned)
        if remove_urls:
            cleaned = self.remove_emails(self.remove_urls(cleaned))

        kept_lines: list[str] = []
        for raw_line in cleaned.split("\n"):
            line = self.strip_bullets(raw_line.strip())
            if not line:
                kept_lines.append("")
                continue
            if self.is_noise_line(line):
                continue
            kept_lines.append(_MULTI_SPACE_RE.sub(" ", line))

        cleaned = "\n".join(kept_lines)
        cleaned = self.dedupe_paragraphs(cleaned)
        return normalise_whitespace(cleaned)

    @staticmethod
    def dedupe_paragraphs(text: str) -> str:
        """Drop repeated paragraphs inside one document (keeps first occurrence)."""
        if not text:
            return ""
        blocks = [block for block in re.split(r"\n{2,}", text) if block.strip()]
        seen: set[str] = set()
        result: list[str] = []
        for block in blocks:
            key = normalise_whitespace(block).lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(block.strip())
        return "\n\n".join(result)

    def remove_repeated_boilerplate(self, texts: Sequence[str], threshold: float = 0.6) -> list[str]:
        """Remove running headers/footers that repeat across pages.

        Args:
            texts: Page texts belonging to the same source.
            threshold: Fraction of pages a line must appear on to be removed
                (only applies when there are at least three pages).

        Returns:
            Cleaned page texts in the original order.
        """
        if len(texts) < 3:
            return list(texts)
        counts: Counter[str] = Counter()
        for text in texts:
            seen_on_page = {
                normalise_whitespace(line).lower()
                for line in (text or "").split("\n")
                if line.strip() and len(line.strip()) <= _MAX_HEADER_LINE_LEN
            }
            counts.update(seen_on_page)

        min_pages = max(2, int(len(texts) * threshold))
        repeated = {line for line, count in counts.items() if count >= min_pages}
        if not repeated:
            return list(texts)

        filtered: list[str] = []
        removed = 0
        for text in texts:
            kept = []
            for line in (text or "").split("\n"):
                key = normalise_whitespace(line).lower()
                is_boilerplate = key in repeated
                has_hint = any(hint in key for hint in _BOILERPLATE_HINTS)
                if is_boilerplate or (has_hint and len(key) <= _MAX_HEADER_LINE_LEN):
                    removed += 1
                    continue
                kept.append(line)
            filtered.append("\n".join(kept))
        if removed:
            logger.debug("Removed %s repeated boilerplate lines across %s pages", removed, len(texts))
        return filtered

    # --------------------------------------------------------------- documents
    def clean_documents(
        self,
        documents: Iterable[RawDocument],
        remove_urls: bool = False,
    ) -> list[RawDocument]:
        """Clean a batch of documents, applying cross-page boilerplate removal.

        Documents are grouped by source so that page-level furniture can be
        detected; the returned documents keep their identity and metadata.

        Args:
            documents: Raw documents.
            remove_urls: Strip URLs/e-mails while cleaning.

        Returns:
            The same documents with cleaned text (empty documents are dropped).
        """
        docs = list(documents)
        grouped: dict[str, list[int]] = {}
        for index, doc in enumerate(docs):
            grouped.setdefault(doc.source, []).append(index)

        cleaned: list[RawDocument] = []
        for indices in grouped.values():
            texts = [docs[index].text for index in indices]
            if len(texts) >= 3:
                texts = self.remove_repeated_boilerplate(texts)
            for index, text in zip(indices, texts, strict=True):
                doc = docs[index]
                is_markdown = str(doc.metadata.get("loader", "")).lower() in {"markdown", "text"} and str(
                    doc.metadata.get("path", "")
                ).lower().endswith((".md", ".markdown"))
                new_text = self.clean(text, remove_urls=remove_urls, strip_markdown=is_markdown)
                if len(new_text) < max(1, self.settings.min_chunk_chars // 2):
                    logger.debug("Dropping near-empty document %s page %s", doc.source, doc.page)
                    continue
                doc.text = new_text
                cleaned.append(doc)
        logger.debug("Cleaned %s/%s documents", len(cleaned), len(docs))
        return cleaned


def clean_text(text: str, settings: Settings | None = None, remove_urls: bool = False) -> str:
    """Convenience wrapper around :meth:`TextCleaner.clean`."""
    return TextCleaner(settings).clean(text, remove_urls=remove_urls)


__all__ = ["TextCleaner", "clean_text"]
