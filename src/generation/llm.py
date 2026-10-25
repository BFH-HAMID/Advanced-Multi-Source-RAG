"""LLM backends and the resilient client used by the generation stage.

Supported providers (``LLM_PROVIDER``):

=============  ==========================================================
``openai``     OpenAI or any OpenAI-compatible endpoint (``OPENAI_BASE_URL``)
``anthropic``  Claude API (``claude-3-5-sonnet-latest`` by default)
``ollama``     Local models through the Ollama daemon
``extractive`` No model: builds a grounded, cited answer from the contexts
=============  ==========================================================

The client retries transient failures, falls back to the extractive backend
when a provider is unreachable (unless ``ALLOW_MODEL_FALLBACKS=false``) and
exposes both blocking and streaming completion.

Example:
    >>> from src.config import Settings
    >>> from src.generation.llm import get_llm_client
    >>> client = get_llm_client(Settings(llm_provider="extractive"))
    >>> client.provider
    'extractive'
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any

from src.config import Settings, get_settings
from src.errors import ConfigurationError, GenerationError
from src.models import RetrievedContext
from src.text_utils import looks_bangla

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response.

    Handles markdown fences, leading prose, trailing commas and (as a last
    resort) Python-style single quotes.

    Args:
        text: Raw model output.

    Returns:
        Parsed dictionary (empty when nothing could be parsed).
    """
    if not text:
        return {}
    candidate = text.strip()
    fence = _JSON_FENCE_RE.search(candidate)
    if fence:
        candidate = fence.group(1).strip()

    for attempt in (candidate,):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
        except json.JSONDecodeError:
            pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start != -1 and end > start:
        snippet = candidate[start : end + 1]
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(snippet)
                if isinstance(parsed, dict):
                    return parsed
            except (ValueError, SyntaxError):
                logger.debug("Could not parse JSON object from model output")

    # Last resort: strip trailing commas and retry.
    if start != -1 and end > start:
        snippet = re.sub(r",\s*([}\]])", r"\1", candidate[start : end + 1])
        try:
            parsed = json.loads(snippet)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
    return {}


class BaseLLMBackend(ABC):
    """Interface implemented by every LLM backend."""

    provider: str = "base"

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 1024, timeout: float = 120.0) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    @property
    def is_available(self) -> bool:
        """Whether the backend can be used with the current configuration."""
        return True

    @abstractmethod
    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Return the model's completion for ``prompt``."""

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> Iterator[str]:
        """Yield answer chunks (default: a single chunk)."""
        yield self.complete(prompt, system=system, query=query, contexts=contexts)


class OpenAIBackend(BaseLLMBackend):
    """OpenAI chat-completions backend (also works with compatible gateways)."""

    provider = "openai"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout)
        self.api_key = api_key
        self.base_url = base_url
        self._client: Any = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url or None,
                    timeout=self.timeout,
                )
            except Exception as exc:  # pragma: no cover - import dependent
                raise GenerationError(f"cannot create OpenAI client: {exc}") from exc
        return self._client

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Call ``chat.completions`` (with a JSON-mode fallback for gateways)."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(prompt, system),
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            if json_mode:
                logger.debug("JSON mode unsupported, retrying without response_format: %s", exc)
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
            else:
                raise GenerationError(f"OpenAI completion failed: {exc}") from exc
        try:
            content = response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - malformed response
            raise GenerationError(f"unexpected OpenAI response: {exc}") from exc
        return content.strip()

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> Iterator[str]:
        """Stream tokens from the chat completions API."""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self._messages(prompt, system),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content if event.choices else None
                if delta:
                    yield delta
        except Exception as exc:
            raise GenerationError(f"OpenAI streaming failed: {exc}") from exc


class AnthropicBackend(BaseLLMBackend):
    """Claude backend using the Messages API."""

    provider = "anthropic"

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout)
        self.api_key = api_key
        self._client: Any = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)
            except Exception as exc:  # pragma: no cover - import dependent
                raise GenerationError(f"cannot create Anthropic client: {exc}") from exc
        return self._client

    def _payload(self, prompt: str, system: str | None, max_tokens: int | None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            payload["system"] = system
        return payload

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Call the Claude Messages API."""
        payload = self._payload(prompt, system, max_tokens)
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            response = self.client.messages.create(**payload)
        except Exception as exc:
            raise GenerationError(f"Anthropic completion failed: {exc}") from exc
        parts = [block.text for block in getattr(response, "content", []) if getattr(block, "text", None)]
        return "".join(parts).strip()

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> Iterator[str]:
        """Stream Claude output using the messages stream helper."""
        try:
            with self.client.messages.stream(**self._payload(prompt, system, None)) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as exc:
            raise GenerationError(f"Anthropic streaming failed: {exc}") from exc


class OllamaBackend(BaseLLMBackend):
    """Local model backend through the Ollama daemon."""

    provider = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model, temperature, max_tokens, timeout)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                from ollama import Client

                headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
                self._client = Client(host=self.base_url, headers=headers, timeout=self.timeout)
            except Exception as exc:  # pragma: no cover - import dependent
                raise GenerationError(f"cannot create Ollama client: {exc}") from exc
        return self._client

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Call ``/api/chat`` on the local Ollama server."""
        options = {
            "temperature": self.temperature if temperature is None else temperature,
            "num_predict": max_tokens or self.max_tokens,
        }
        try:
            response = self.client.chat(
                model=self.model,
                messages=self._messages(prompt, system),
                options=options,
                format="json" if json_mode else "",
            )
        except Exception as exc:
            raise GenerationError(
                f"Ollama completion failed ({self.base_url}/{self.model}): {exc}"
            ) from exc
        if isinstance(response, dict):
            content = (response.get("message") or {}).get("content", "")
        else:  # pragma: no cover - client API drift
            content = getattr(getattr(response, "message", None), "content", "")
        return str(content).strip()

    def stream(
        self,
        prompt: str,
        system: str | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> Iterator[str]:
        """Stream tokens from Ollama."""
        try:
            stream = self.client.chat(
                model=self.model,
                messages=self._messages(prompt, system),
                options={"temperature": self.temperature, "num_predict": self.max_tokens},
                stream=True,
            )
            for chunk in stream:
                content = (chunk.get("message") or {}).get("content", "") if isinstance(chunk, dict) else ""
                if content:
                    yield content
        except Exception as exc:
            raise GenerationError(f"Ollama streaming failed: {exc}") from exc


class ExtractiveBackend(BaseLLMBackend):
    """No-model backend: returns a grounded, cited extract from the contexts.

    This keeps the system usable (and honest) in air-gapped environments and
    makes the test-suite deterministic.
    """

    provider = "extractive"

    def __init__(self, model: str = "extractive-offline") -> None:
        super().__init__(model=model, temperature=0.0, max_tokens=512)

    @property
    def is_available(self) -> bool:
        return True

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Compose an answer from the most query-relevant context sentences."""
        if json_mode:
            return json.dumps({"triples": []})
        from src.generation.query_builder import NOT_FOUND_BN, NOT_FOUND_EN, extractive_answer

        effective_query = query or self._query_from_prompt(prompt)
        if not contexts:
            return NOT_FOUND_BN if looks_bangla(effective_query) else NOT_FOUND_EN
        return extractive_answer(effective_query, contexts)

    @staticmethod
    def _query_from_prompt(prompt: str) -> str:
        """Best-effort extraction of the question from a rendered prompt."""
        marker = "QUESTION"
        if marker in prompt:
            tail = prompt.split(marker, 1)[1]
            return tail.replace("=", "").strip()
        return prompt.strip()[-500:]


class LLMClient:
    """Resilient facade over a single LLM backend."""

    def __init__(
        self,
        backend: BaseLLMBackend,
        fallback: BaseLLMBackend | None = None,
        max_retries: int = 2,
        retry_backoff: float = 1.5,
        allow_fallback: bool = True,
    ) -> None:
        self.backend = backend
        self.fallback = fallback
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = retry_backoff
        self.allow_fallback = allow_fallback
        self.last_error: str | None = None
        self.used_fallback = False

    @property
    def provider(self) -> str:
        """Active provider name."""
        return self.backend.provider

    @property
    def model(self) -> str:
        """Active model name."""
        return self.backend.model

    @property
    def is_available(self) -> bool:
        """Whether the primary backend is usable."""
        return self.backend.is_available

    def _switch_to_fallback(self, reason: Exception | str) -> bool:
        """Switch to the fallback backend; return ``True`` on success."""
        if not self.allow_fallback or self.fallback is None or self.backend is self.fallback:
            return False
        self.last_error = str(reason)
        self.used_fallback = True
        logger.warning(
            "LLM provider %s failed (%s); falling back to %s",
            self.backend.provider,
            reason,
            self.fallback.provider,
        )
        self.backend = self.fallback
        return True

    def complete(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
        temperature: float | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> str:
        """Generate a completion with retries and fallback.

        Raises:
            GenerationError: When every backend fails.
        """
        if not self.backend.is_available and not self._switch_to_fallback("backend not configured"):
                raise GenerationError(
                    f"LLM provider {self.backend.provider!r} is not configured. "
                    "Set the API key/base URL in .env or choose another LLM_PROVIDER."
                )

        last_exception: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self.backend.complete(
                    prompt,
                    system=system,
                    json_mode=json_mode,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    query=query,
                    contexts=contexts,
                )
            except Exception as exc:
                last_exception = exc
                logger.warning(
                    "LLM attempt %s/%s failed on %s: %s",
                    attempt + 1,
                    self.max_retries + 1,
                    self.backend.provider,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(min(8.0, self.retry_backoff ** (attempt + 1)))

        if last_exception is not None and self._switch_to_fallback(last_exception):
            return self.backend.complete(
                prompt,
                system=system,
                json_mode=json_mode,
                max_tokens=max_tokens,
                temperature=temperature,
                query=query,
                contexts=contexts,
            )
        raise GenerationError(f"LLM generation failed: {last_exception}") from last_exception

    def complete_json(self, prompt: str, system: str | None = None) -> dict[str, Any]:
        """Generate and parse a JSON object (never raises on parse errors)."""
        try:
            raw = self.complete(prompt, system=system, json_mode=True)
        except GenerationError as exc:
            logger.warning("JSON completion failed: %s", exc)
            raise
        parsed = parse_json_object(raw)
        if not parsed:
            logger.debug("Could not parse JSON from: %s", raw[:200])
        return parsed

    def stream_complete(
        self,
        prompt: str,
        system: str | None = None,
        query: str | None = None,
        contexts: Sequence[RetrievedContext] | None = None,
    ) -> Iterator[str]:
        """Stream answer chunks, falling back to a single chunk when needed."""
        try:
            yielded = False
            for chunk in self.backend.stream(prompt, system=system, query=query, contexts=contexts):
                yielded = True
                yield chunk
            if not yielded:
                raise GenerationError("empty stream")
        except Exception as exc:
            logger.warning("Streaming failed on %s (%s); using non-streaming path", self.provider, exc)
            try:
                answer = self.complete(prompt, system=system, query=query, contexts=contexts)
            except GenerationError:
                raise
            yield answer


def build_backend(settings: Settings, provider: str | None = None) -> BaseLLMBackend:
    """Instantiate the backend described by ``settings``.

    Args:
        settings: Project settings.
        provider: Optional provider override.

    Returns:
        A backend instance (possibly unavailable; check ``is_available``).

    Raises:
        ConfigurationError: For an unknown provider.
    """
    name = (provider or settings.llm_provider).lower()
    model = settings.llm_model.strip() or settings.resolved_llm_model
    if name == "openai":
        return OpenAIBackend(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if name == "anthropic":
        return AnthropicBackend(
            model=model if settings.llm_model.strip() else "claude-3-5-sonnet-latest",
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if name == "ollama":
        return OllamaBackend(
            model=model,
            base_url=settings.ollama_base_url,
            api_key=settings.ollama_api_key,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    if name == "extractive":
        return ExtractiveBackend()
    raise ConfigurationError(f"unknown LLM provider {name!r}")


def get_llm_client(settings: Settings | None = None, provider: str | None = None) -> LLMClient:
    """Create the LLM client with a grounded fallback backend.

    Args:
        settings: Project settings.
        provider: Optional provider override.

    Returns:
        A configured :class:`LLMClient`.
    """
    settings = settings or get_settings()
    backend = build_backend(settings, provider)
    fallback = ExtractiveBackend() if settings.allow_model_fallbacks else None
    client = LLMClient(
        backend=backend,
        fallback=fallback,
        max_retries=settings.llm_max_retries,
        allow_fallback=settings.allow_model_fallbacks,
    )
    if not backend.is_available:
        logger.warning(
            "LLM provider %s has no credentials; the grounded extractive fallback will be used",
            backend.provider,
        )
    return client


__all__ = [
    "BaseLLMBackend",
    "OpenAIBackend",
    "AnthropicBackend",
    "OllamaBackend",
    "ExtractiveBackend",
    "LLMClient",
    "get_llm_client",
    "build_backend",
    "parse_json_object",
]
