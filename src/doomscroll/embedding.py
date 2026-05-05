"""Embedding provider adapters.

Anthropic and OpenRouter do not provide first-party embeddings APIs as of
2026, so OpenAI is the primary real backend. A FakeEmbedder provides
deterministic vectors for tests and stub-mode runs.

Usage pattern: pre-embed the entire MoltBook snapshot ONCE before any agent
runs (per Issue 1 + performance section of the eng review). The agent loop
should not call the embedding API during a session.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dim: int = 1536  # default for text-embedding-3-small
    api_key: str | None = None


class Embedder(Protocol):
    def embed(self, text: str) -> NDArray[np.float64]: ...
    def embed_batch(self, texts: list[str]) -> NDArray[np.float64]: ...


class FakeEmbedder:
    """Deterministic hash-based embedder. Same text always produces same vector.

    Not semantically meaningful -- two synonymous texts get unrelated vectors.
    Useful for tests and pipeline shakedowns; useless for actual experiments.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _vec(self, text: str) -> NDArray[np.float64]:
        # Stretch a SHA-256 hash into a deterministic vector of arbitrary dim.
        h = hashlib.sha256(text.encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(self.dim)
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def embed(self, text: str) -> NDArray[np.float64]:
        return self._vec(text)

    def embed_batch(self, texts: list[str]) -> NDArray[np.float64]:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float64)
        return np.stack([self._vec(t) for t in texts])


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small / -large / older ada-002."""

    def __init__(self, config: EmbeddingConfig | None = None):
        from openai import OpenAI

        self._config = config or EmbeddingConfig()
        api_key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set and no api_key in EmbeddingConfig"
            )
        self._client = OpenAI(api_key=api_key)

    @property
    def dim(self) -> int:
        return self._config.dim

    def embed(self, text: str) -> NDArray[np.float64]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> NDArray[np.float64]:
        if not texts:
            return np.zeros((0, self._config.dim), dtype=np.float64)
        response = self._client.embeddings.create(
            model=self._config.model, input=texts
        )
        vectors = [np.asarray(item.embedding, dtype=np.float64) for item in response.data]
        return np.stack(vectors)


PROVIDERS = {
    "openai": OpenAIEmbedder,
    "fake": FakeEmbedder,
}


def make_embedder(provider: str, config: EmbeddingConfig | None = None) -> Embedder:
    """Factory: build an embedder by provider name."""
    if provider == "fake":
        return FakeEmbedder()
    if provider == "openai":
        return OpenAIEmbedder(config)
    raise ValueError(
        f"unknown embedding provider {provider!r}; choose from {sorted(PROVIDERS)}"
    )
