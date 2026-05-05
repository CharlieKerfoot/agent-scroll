"""All tunable parameters live here. No magic numbers in loop code."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DistortionConfig:
    """Per-agent cognitive distortion parameters."""

    name: str
    mood_volatility: float = 0.1
    mood_engagement_bias: float = 0.3
    novelty_hunger_initial: float = 0.7
    novelty_hunger_decay: float = 0.05
    confirmation_bias: float = 0.2
    emotional_contagion: float = 0.3
    reactivation_strength: float = 0.5


@dataclass(frozen=True)
class MemoryConfig:
    """Decay and update parameters for episodic + interest-vector memory."""

    trace_decay_lambda: float = 0.0289  # ln(2) / 24h, in 1/hour units; half-life = 1 day
    centroid_weight_decay_lambda: float = 0.00412  # half-life ~ 1 week
    centroid_alpha: float = 0.1  # EMA pull strength on engagement
    centroid_max_count: int = 5
    new_centroid_similarity_threshold: float = 0.55  # below this, spawn new centroid
    target_engagement_rate: float = 0.15  # adapt salience threshold to hit this
    salience_warmup_posts: int = 50  # posts before threshold adapts


@dataclass(frozen=True)
class ConsolidationConfig:
    """Sleep-consolidation pass parameters."""

    belief_sample_size: int = 20  # weighted sample of existing beliefs into prompt
    fragment_max_count: int = 50  # cap fragments per consolidation
    confidence_weight_alpha: float = 0.7  # confidence weight in sampling
    recency_weight_alpha: float = 0.3  # recency weight in sampling
    max_belief_updates_per_pass: int = 3


# Four pre-defined agent variants from the design doc.
AGENT_VARIANTS: dict[str, DistortionConfig] = {
    "balanced": DistortionConfig(name="balanced"),
    "high_confirmation_bias": DistortionConfig(
        name="high_confirmation_bias",
        confirmation_bias=0.6,
    ),
    "high_novelty_hunger": DistortionConfig(
        name="high_novelty_hunger",
        novelty_hunger_initial=0.95,
        novelty_hunger_decay=0.02,
    ),
    "high_mood_volatility": DistortionConfig(
        name="high_mood_volatility",
        mood_volatility=0.4,
        mood_engagement_bias=0.6,
    ),
}
