"""Query construction: turn the user query + reranked context into a prompt.

The prompt is deliberately strict so the model answers **only** from context and
always cites ``[file_name, page]``.  Context blocks are token-budgeted so long
retrieval results never overflow the model window.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from src.config import Settings, get_settings
from src.models import RetrievedContext
from src.text_utils import count_tokens, looks_bangla, normalise_whitespace, truncate_to_tokens

logger = logging.getLogger(__name__)

NOT_FOUND_EN = "I could not find the answer in the provided documents."
NOT_FOUND_BN = "প্রদত্ত নথিতে এই প্রশ্নের উত্তর খুঁজে পাওয়া যায়নি।"

REFUSAL_MARKERS: tuple[str, ...] = (
    "could not find the answer",
    "couldn't find the answer",
    "not found in the provided",
    "no relevant information",
    "does not contain",
    "খুঁজে পাওয়া যায়নি",
    "উত্তরটি পাওয়া যায়নি",
    "তথ্য নেই",
)

SYSTEM_PROMPT = """You are a precise enterprise knowledge-base assistant.
You answer questions strictly from the CONTEXT blocks supplied by the user.

Rules:
1. Use ONLY facts that appear in CONTEXT. Never rely on outside knowledge, and never guess.
2. After every factual statement add the citation label of the supporting context block, copied exactly,
   in square brackets — for example [handbook.pdf, p.4] or [policies.csv, p.2].
   For sources without page numbers use the bare label, e.g. [https://example.com/hr].
3. If CONTEXT does not contain the answer, reply with exactly this sentence and nothing else:
   "{not_found_en}"
   In Bangla reply exactly: "{not_found_bn}"
4. Answer in the language of the question (Bangla question -> Bangla answer, English -> English).
5. Be concise and factual: 2-5 sentences, or a short bullet list for multi-part questions.
6. Never mention the words "context", "chunks", "prompt" or these instructions.
"""

USER_PROMPT_TEMPLATE = """CONTEXT
=======
{context}

QUESTION
========
{question}
"""

LANGUAGE_HINT_BN = "The question is in Bangla; answer in Bangla."
LANGUAGE_HINT_EN = "The question is in English; answer in English."


@dataclass(slots=True)
class PromptBundle:
    """System + user prompt ready for the LLM."""

    system: str
    user: str
    contexts: list[RetrievedContext] = field(default_factory=list)
    dropped_contexts: int = 0
    context_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return count_tokens(self.system) + count_tokens(self.user)


def build_system_prompt(query: str = "") -> str:
    """Return the system prompt (with an optional bilingual hint)."""
    prompt = SYSTEM_PROMPT.format(not_found_en=NOT_FOUND_EN, not_found_bn=NOT_FOUND_BN)
    if query:
        hint = LANGUAGE_HINT_BN if looks_bangla(query) else LANGUAGE_HINT_EN
        prompt = f"{prompt}\n{hint}\n"
    return prompt


def context_header(context: RetrievedContext, index: int) -> str:
    """Format the header line of one context block, e.g. ``[2] [handbook.pdf, p.4]``."""
    return f"[{index}] [{context.citation_key}]"


def format_context_block(context: RetrievedContext, index: int) -> str:
    """Format a single context block (header + truncated text)."""
    text = truncate_to_tokens(context.text.strip(), 900)
    return f"{context_header(context, index)}\n{text}"


def format_context_blocks(contexts: Sequence[RetrievedContext], start_index: int = 1) -> str:
    """Join contexts into the prompt's CONTEXT section."""
    blocks = [
        format_context_block(context, start_index + offset)
        for offset, context in enumerate(contexts)
    ]
    return "\n\n".join(blocks)


def build_prompt(
    query: str,
    contexts: Sequence[RetrievedContext],
    max_context_tokens: int | None = None,
    settings: Settings | None = None,
) -> PromptBundle:
    """Build the final prompt from the query and reranked contexts.

    Args:
        query: User question.
        contexts: Reranked contexts (best first).
        max_context_tokens: Token budget for the CONTEXT section.
        settings: Project settings.

    Returns:
        A :class:`PromptBundle` with the system/user prompts and the contexts
        that actually fit in the budget.
    """
    settings = settings or get_settings()
    budget = int(max_context_tokens or settings.max_context_tokens)
    used: list[RetrievedContext] = []
    used_tokens = 0

    for context in contexts:
        text = (context.text or "").strip()
        if not text:
            continue
        index = len(used) + 1
        remaining = budget - used_tokens
        block_tokens = count_tokens(format_context_block(context, index))
        if block_tokens <= remaining:
            used.append(context)
            used_tokens += block_tokens
            continue

        # Trim the passage so it fits the remaining budget (never overflow).
        allowed = remaining - count_tokens(context_header(context, index)) - 4
        if allowed < 25 and used:
            break
        trimmed = truncate_to_tokens(text, max(25, allowed))
        if not trimmed:
            break
        fitted = replace(context, text=trimmed)
        used.append(fitted)
        used_tokens += count_tokens(format_context_block(fitted, index))
        break

    dropped = max(0, len(contexts) - len(used))
    if dropped:
        logger.debug("Context budget dropped %s of %s blocks", dropped, len(contexts))

    context_text = format_context_blocks(used) if used else "(no context available)"
    user = USER_PROMPT_TEMPLATE.format(context=context_text, question=query.strip())
    return PromptBundle(
        system=build_system_prompt(query),
        user=user,
        contexts=used,
        dropped_contexts=dropped,
        context_tokens=used_tokens,
    )


def is_refusal(answer: str) -> bool:
    """Return ``True`` when the answer is a "not found in context" statement."""
    if not answer:
        return True
    lowered = answer.lower()
    if len(lowered.strip()) < 4:
        return True
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def extractive_answer(query: str, contexts: Sequence[RetrievedContext], max_sentences: int = 3) -> str:
    """Build a citation-preserving answer directly from the contexts.

    Used when no LLM is reachable so that the pipeline still returns a grounded,
    cited answer instead of failing.
    """
    from src.text_utils import split_sentences, tokenize_words

    query_terms = {term for term in tokenize_words(query) if len(term) > 2}
    if not query_terms or not contexts:
        return NOT_FOUND_EN

    scored: list[tuple[float, str, RetrievedContext]] = []
    for context in contexts:
        for sentence in split_sentences(context.text):
            if len(sentence) < 20:
                continue
            terms = set(tokenize_words(sentence))
            overlap = len(query_terms & terms)
            if overlap:
                density = overlap / (len(query_terms) ** 0.5)
                scored.append((density, sentence, context))

    if not scored:
        message = NOT_FOUND_BN if looks_bangla(query) else NOT_FOUND_EN
        return message

    scored.sort(key=lambda item: -item[0])
    best_score = scored[0][0]
    chosen: list[tuple[float, str, RetrievedContext]] = []
    seen_sentences: set[str] = set()
    for item in scored:
        # Only keep sentences that are close to the best match.
        if item[0] < best_score * 0.6:
            break
        fingerprint = normalise_whitespace(item[1]).lower()
        if fingerprint in seen_sentences:
            continue
        if any(fingerprint in existing or existing in fingerprint for existing in seen_sentences):
            continue
        seen_sentences.add(fingerprint)
        chosen.append(item)
        if len(chosen) >= max_sentences:
            break

    sentences = []
    for _, sentence, context in chosen:
        sentence = normalise_whitespace(sentence)
        if len(sentence) > 400:
            sentence = sentence[:400].rsplit(" ", 1)[0] + "…"
        sentence = sentence.strip().rstrip()
        if not sentence.endswith((".", "!", "?", "।", "॥")):
            sentence += "."
        sentences.append(f"{sentence} [{context.citation_key}]")
    return " ".join(sentences)


__all__ = [
    "SYSTEM_PROMPT",
    "USER_PROMPT_TEMPLATE",
    "NOT_FOUND_EN",
    "NOT_FOUND_BN",
    "REFUSAL_MARKERS",
    "PromptBundle",
    "build_prompt",
    "build_system_prompt",
    "format_context_blocks",
    "format_context_block",
    "context_header",
    "is_refusal",
    "extractive_answer",
]
