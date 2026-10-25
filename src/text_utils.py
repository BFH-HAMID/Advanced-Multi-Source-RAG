"""Unicode-aware text utilities shared across modules.

The project must handle Bangla and English equally well, therefore token
counting is *estimator based* by default: ``tiktoken`` BPE files need a network
download and under-count Bengali text, so we use a Unicode-aware word/punctuation
tokenizer that works fully offline and treats Bengali combining characters
correctly.

If a ``tiktoken`` encoding is available offline (``TIKTOKEN_CACHE_DIR`` warm or
a local cache present), it is used automatically for English-token parity.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from collections.abc import Iterable, Iterator

logger = logging.getLogger(__name__)

# Bengali block + Bengali danda, general punctuation, ASCII/Latin punctuation.
_SENTENCE_END = "।॥.!?…؟"
_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u200b\u200c\u200d\u2060]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[\w\u0980-\u09FF]+", re.UNICODE)

# Common abbreviations that should not terminate an English sentence.
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "etc",
    "e.g",
    "i.e",
    "vs",
    "fig",
    "no",
    "inc",
    "ltd",
    "co",
    "st",
    "approx",
    "dept",
    "est",
}


def normalise_whitespace(text: str) -> str:
    """Collapse runs of spaces and excessive newlines while keeping paragraphs."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MULTI_NEWLINE_RE.sub("\n\n", text).strip()


def tokenize_words(text: str, lowercase: bool = True) -> list[str]:
    """Split ``text`` into word tokens, preserving Bengali words.

    Args:
        text: Input text.
        lowercase: Whether to case-fold the tokens.

    Returns:
        List of word tokens.
    """
    if not text:
        return []
    tokens = _WORD_RE.findall(text)
    return [token.lower() for token in tokens] if lowercase else tokens


def character_ngrams(text: str, min_n: int = 3, max_n: int = 5) -> list[str]:
    """Return character n-grams of the normalised text (used by the fallback embedder)."""
    cleaned = normalise_whitespace(text).lower()
    if not cleaned:
        return []
    grams: list[str] = []
    for word in cleaned.split():
        padded = f"_{word}_"
        for n in range(min_n, max_n + 1):
            if len(padded) < n:
                continue
            grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


@functools.lru_cache(maxsize=1)
def _tiktoken_encoder():  # type: ignore[no-untyped-def]
    """Return a cached ``tiktoken`` encoding, or ``None`` when unavailable offline."""
    try:  # pragma: no cover - depends on network/cache availability
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover - offline environments
        logger.debug("tiktoken unavailable, falling back to heuristic counting: %s", exc)
        return None


@functools.lru_cache(maxsize=1)
def tiktoken_available() -> bool:
    """Report whether a ``tiktoken`` encoder is usable in this environment."""
    return _tiktoken_encoder() is not None


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` without external tokenizers.

    The heuristic counts word and punctuation tokens; Bengali text tokenises
    into noticeably more pieces than whitespace splitting would suggest, which
    keeps chunks safely inside the configured token budget.
    """
    if not text:
        return 0
    tokens = _TOKEN_RE.findall(text)
    # Sub-word correction: long words (esp. Bengali) usually cost >1 token.
    extra = sum(max(0, (len(token) - 5) // 4) for token in tokens if token.isalnum())
    return len(tokens) + extra


def count_tokens(text: str, use_tiktoken: bool | None = None) -> int:
    """Count tokens with ``tiktoken`` when available, else use the estimator."""
    if not text:
        return 0
    if use_tiktoken is False:
        return estimate_tokens(text)
    encoder = _tiktoken_encoder()
    if encoder is None:
        return estimate_tokens(text)
    try:  # pragma: no cover - only when tiktoken cache is warm
        return len(encoder.encode(text, disallowed_special=()))
    except Exception:
        return estimate_tokens(text)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate ``text`` so that :func:`count_tokens` stays below ``max_tokens``."""
    if max_tokens <= 0 or not text:
        return ""
    if count_tokens(text) <= max_tokens:
        return text
    # Split on whitespace but keep the separators so newlines (table rows,
    # bullet lists) survive truncation.
    pieces = re.findall(r"\S+\s*", text)
    low, high = 0, len(pieces)
    while low < high:
        mid = (low + high + 1) // 2
        if count_tokens("".join(pieces[:mid])) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return "".join(pieces[:low]).rstrip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, supporting English and Bengali punctuation.

    The splitter handles ``।`` (Bengali danda), ``॥``, ``!``, ``?``, ``…``,
    ellipses, common abbreviations and newline-delimited fragments.

    Args:
        text: Input text.

    Returns:
        A list of non-empty, stripped sentences.
    """
    if not text:
        return []

    sentences: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(text)

    while index < length:
        char = text[index]
        buffer.append(char)
        if char in _SENTENCE_END or (char == "\n" and "".join(buffer).strip()):
            is_newline = char == "\n"
            candidate = "".join(buffer).strip()
            # Consume repeated terminators, e.g. "?!" or "..." / "।।"
            while index + 1 < length and text[index + 1] in _SENTENCE_END:
                index += 1
                buffer.append(text[index])
            if is_newline:
                if candidate:
                    sentences.append(candidate)
                buffer = []
            else:
                trailing_space = index + 1 < length and text[index + 1].isspace()
                if trailing_space:
                    candidate = "".join(buffer).strip()
                    last_word = re.split(r"[\s(]", candidate.rstrip("".join(_SENTENCE_END)))[-1].lower()
                    last_word = last_word.strip(".,;:()[]\"'")
                    if last_word in _ABBREVIATIONS or re.fullmatch(r"[a-z]", last_word or ""):
                        buffer.append(text[index + 1] if index + 1 < length else "")
                    else:
                        sentences.append(candidate)
                        buffer = []
                # else: number/decimal such as "3.14" -> keep buffering
        index += 1

    tail = "".join(buffer).strip()
    if tail:
        sentences.append(tail)

    cleaned = [normalise_whitespace(s) for s in sentences]
    return [s for s in cleaned if s]


def iter_words(text: str) -> Iterator[str]:
    """Yield lower-cased words from ``text``."""
    yield from tokenize_words(text)


def unique_preserving_order(items: Iterable[str]) -> list[str]:
    """Deduplicate ``items`` while preserving first-seen order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def looks_bangla(text: str, threshold: float = 0.15) -> bool:
    """Heuristically detect whether ``text`` is predominantly Bengali script."""
    if not text:
        return False
    bengali = sum(1 for char in text if "\u0980" <= char <= "\u09FF")
    letters = sum(1 for char in text if char.isalpha())
    if letters == 0:
        return False
    return (bengali / letters) >= threshold
