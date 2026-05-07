"""Pydantic-validated YAML run config for the pilot.

A pilot run is fully described by one `RunConfig`. Source of truth for
parameters lives in YAML files under `configs/`; the CLI is a thin layer
that loads YAML, applies optional overrides, and hands a resolved config
to `cli.main`.

The resolved config is dumped to `output_dir/config.yaml` at run start so
every result directory is self-describing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from doomscroll.config import (
    AGENT_VARIANTS,
    ConsolidationConfig,
    DistortionConfig,
    MemoryConfig,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMSettings(_Frozen):
    provider: Literal["anthropic", "openai", "openrouter", "fake"] = "fake"
    model: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024

    @model_validator(mode="after")
    def _model_required_for_real_providers(self) -> LLMSettings:
        if self.provider != "fake" and not self.model:
            raise ValueError(
                f"llm.model is required for provider={self.provider!r}"
            )
        return self


class EmbedderSettings(_Frozen):
    provider: Literal["openai", "fake"] = "fake"
    model: str = "text-embedding-3-small"


class FeedSettings(_Frozen):
    snapshot: Path | None = None  # None -> synthetic feed


class PilotSettings(_Frozen):
    variants: list[str] = Field(default_factory=lambda: list(AGENT_VARIANTS))
    sessions: int = 3
    posts_per_session: int = 50
    seed: int = 42
    consolidate: bool = True

    @model_validator(mode="after")
    def _validate_variants(self) -> PilotSettings:
        if not self.variants:
            raise ValueError("pilot.variants must not be empty")
        unknown = [v for v in self.variants if v not in AGENT_VARIANTS]
        if unknown:
            valid = ", ".join(sorted(AGENT_VARIANTS))
            raise ValueError(f"unknown variants {unknown}; valid: {valid}")
        return self


class MemorySettings(_Frozen):
    """Mirrors MemoryConfig fields. Kept separate so YAML can override them."""

    trace_decay_lambda: float = 0.0289
    centroid_weight_decay_lambda: float = 0.00412
    centroid_alpha: float = 0.1
    centroid_max_count: int = 5
    new_centroid_similarity_threshold: float = 0.55
    target_engagement_rate: float = 0.15
    salience_warmup_posts: int = 50

    def to_dataclass(self) -> MemoryConfig:
        return MemoryConfig(**self.model_dump())


class ConsolidationSettings(_Frozen):
    belief_sample_size: int = 20
    fragment_max_count: int = 50
    confidence_weight_alpha: float = 0.7
    recency_weight_alpha: float = 0.3
    max_belief_updates_per_pass: int = 3

    def to_dataclass(self) -> ConsolidationConfig:
        return ConsolidationConfig(**self.model_dump())


class RunConfig(_Frozen):
    """Top-level resolved config for a pilot run."""

    output_dir: Path = Path("./pilot-output")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedder: EmbedderSettings = Field(default_factory=EmbedderSettings)
    feed: FeedSettings = Field(default_factory=FeedSettings)
    pilot: PilotSettings = Field(default_factory=PilotSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    consolidation: ConsolidationSettings = Field(
        default_factory=ConsolidationSettings
    )

    def variant_distortions(self) -> dict[str, DistortionConfig]:
        return {name: AGENT_VARIANTS[name] for name in self.pilot.variants}

    def to_yaml(self) -> str:
        return yaml.safe_dump(
            _to_yaml_safe(self.model_dump(mode="json")),
            sort_keys=False,
        )

    def write_to(self, path: Path) -> None:
        path.write_text(self.to_yaml())

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> RunConfig:
        """Load YAML (if any), deep-merge overrides, validate."""
        data: dict[str, Any] = {}
        if path is not None:
            if not path.exists():
                raise FileNotFoundError(f"config not found: {path}")
            loaded = yaml.safe_load(path.read_text()) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{path}: top-level YAML must be a mapping")
            data = loaded
        if overrides:
            data = _deep_merge(data, overrides)
        return cls.model_validate(data)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_yaml_safe(obj: Any) -> Any:
    """Recursively coerce Path -> str so yaml.safe_dump accepts it."""
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_yaml_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
