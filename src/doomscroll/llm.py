"""LLM provider adapters.

Three real backends: Anthropic, OpenAI, OpenRouter (OpenAI-compatible).
One fake backend for tests and stub-mode experiments.

All concrete clients structurally satisfy doomscroll.consolidation.LLMClient
(Protocol). Same client serves both reaction calls and consolidation calls;
the prompt is what differs, not the transport.

Model choice should NOT vary across the 4 agent variants -- mixing models
would confound the cognitive distortion experiment. Pick one model per
experiment run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LLMConfig:
    """Provider-agnostic configuration for an LLM client."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    api_key: str | None = None  # None -> read from env
    base_url: str | None = None  # only used by OpenAI-compatible clients


class FakeLLM:
    """Deterministic fake. Returns whatever was queued, in order. For tests."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = list(responses or [])
        self.calls: list[str] = []

    def queue(self, response: str) -> None:
        self._responses.append(response)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append(prompt)
        if not self._responses:
            return ""
        return self._responses.pop(0)


class AnthropicClient:
    """Claude via the Anthropic SDK."""

    def __init__(self, config: LLMConfig):
        from anthropic import Anthropic  # lazy import: dep is optional at module level

        self._config = config
        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set and no api_key in LLMConfig"
            )
        self._client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate all text blocks (Claude can return multiple).
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


class OpenAIClient:
    """GPT via the OpenAI SDK. Also serves OpenRouter via base_url override."""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        self._config = config
        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set and no api_key in LLMConfig")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self._client = OpenAI(**kwargs)

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model,
            max_tokens=max_tokens or self._config.max_tokens,
            temperature=self._config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        choice = response.choices[0]
        return choice.message.content or ""


class OpenRouterClient(OpenAIClient):
    """OpenRouter is OpenAI-compatible -- different default base_url and env var."""

    def __init__(self, config: LLMConfig):
        api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set and no api_key in LLMConfig"
            )
        effective = LLMConfig(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=api_key,
            base_url=config.base_url or "https://openrouter.ai/api/v1",
        )
        super().__init__(effective)


PROVIDERS = {
    "anthropic": AnthropicClient,
    "openai": OpenAIClient,
    "openrouter": OpenRouterClient,
    "fake": FakeLLM,
}


def make_llm(provider: str, config: LLMConfig | None = None):
    """Factory: build a client by provider name.

    'fake' returns a FakeLLM with empty queue (call .queue() to seed responses).
    """
    if provider not in PROVIDERS:
        raise ValueError(
            f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}"
        )
    cls = PROVIDERS[provider]
    if provider == "fake":
        return cls()
    if config is None:
        raise ValueError(f"provider {provider!r} requires an LLMConfig")
    return cls(config)
