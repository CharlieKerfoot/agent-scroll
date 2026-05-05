"""Pure-function memory primitives.

Salience formula
    salience(post) = max_i [ cosine(post, centroid_i) * weight_i ]
                   + mood_engagement_bias  * sign_match(valence_est, mood)
                   + confirmation_bias     * max_belief_alignment(post)
                   + novelty_hunger        * (1 - max_centroid_similarity)
                   + emotional_contagion   * |valence_est|

Centroid update on engagement
    nearest_idx = argmax_i cosine(post, centroid_i)
    if best_similarity < new_centroid_similarity_threshold:
        spawn new centroid (evict least-active if at cap)
    else:
        c.embedding = (1 - alpha) * c.embedding + alpha * post_embedding
        c.weight   += 1
        c.last_active = now

Trace decay (ACT-R-flavored base activation)
    salience(now) = base * exp(-lambda * (now - last_reactivated_at))

Persistence layer is separate (sqlite); this module is pure numpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from doomscroll.config import DistortionConfig, MemoryConfig

EPSILON = 1e-12


@dataclass(frozen=True)
class Centroid:
    embedding: NDArray[np.float64]
    weight: float
    last_active: float


@dataclass(frozen=True)
class Belief:
    id: int
    text: str
    embedding: NDArray[np.float64]
    confidence: float
    last_updated_at: float


def _cosine(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < EPSILON or nb < EPSILON:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def max_centroid_similarity(
    post_embedding: NDArray[np.floating],
    centroids: list[Centroid],
) -> tuple[float, int]:
    """Return (best_similarity, best_index). (-1.0, -1) if no centroids."""
    if not centroids:
        return -1.0, -1
    best_sim = -1.0
    best_idx = -1
    for i, c in enumerate(centroids):
        sim = _cosine(post_embedding, c.embedding)
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_sim, best_idx


def max_belief_alignment(
    post_embedding: NDArray[np.floating],
    beliefs: list[Belief],
) -> float:
    """Confidence-weighted max alignment between post and any held belief.
    Returns 0.0 when there are no beliefs."""
    if not beliefs:
        return 0.0
    best = 0.0
    for b in beliefs:
        sim = _cosine(post_embedding, b.embedding)
        score = max(0.0, sim) * max(0.0, min(1.0, b.confidence))
        if score > best:
            best = score
    return best


def salience_score(
    post_embedding: NDArray[np.floating],
    centroids: list[Centroid],
    beliefs: list[Belief],
    mood: float,
    novelty_hunger: float,
    distortion: DistortionConfig,
    estimated_valence: float = 0.0,
) -> float:
    """Compute salience for an incoming post.

    Cold-start invariant: with empty centroids, max_centroid_similarity = -1
    is treated as 0 for the novelty term, so salience reduces to roughly
    novelty_hunger + (mood/confirmation/emotional terms). With novelty_hunger
    initialized high, the agent engages with most early posts and centroids
    form within the first ~10 engagements. See Issue 3 in eng review.
    """
    best_sim, _ = max_centroid_similarity(post_embedding, centroids)
    interest_term = 0.0
    if centroids:
        # Sum of weight * similarity, normalized by total weight.
        weights = np.array([c.weight for c in centroids], dtype=np.float64)
        sims = np.array(
            [_cosine(post_embedding, c.embedding) for c in centroids],
            dtype=np.float64,
        )
        total_weight = weights.sum()
        if total_weight > EPSILON:
            interest_term = float(np.max(sims * weights) / total_weight)

    # max_centroid_similarity returns -1 when empty; clamp for novelty math
    similarity_for_novelty = max(0.0, best_sim) if centroids else 0.0
    novelty_term = novelty_hunger * (1.0 - similarity_for_novelty)

    # Mood-aligned valence bumps salience (sign agreement, scaled by magnitudes)
    mood_term = (
        distortion.mood_engagement_bias
        * math.copysign(1.0, mood * estimated_valence)
        * min(abs(mood), 1.0)
        * min(abs(estimated_valence), 1.0)
        if abs(mood) > EPSILON and abs(estimated_valence) > EPSILON
        else 0.0
    )

    confirmation_term = distortion.confirmation_bias * max_belief_alignment(
        post_embedding, beliefs
    )

    contagion_term = distortion.emotional_contagion * min(abs(estimated_valence), 1.0)

    return interest_term + novelty_term + mood_term + confirmation_term + contagion_term


def update_centroids_on_engagement(
    centroids: list[Centroid],
    post_embedding: NDArray[np.floating],
    now: float,
    config: MemoryConfig,
) -> list[Centroid]:
    """Apply EMA pull to nearest centroid, or spawn new one.

    Pure function: returns a new list, does not mutate input.
    """
    post = np.asarray(post_embedding, dtype=np.float64)
    best_sim, best_idx = max_centroid_similarity(post, centroids)

    # Spawn case: no centroids yet, OR no centroid is similar enough
    if not centroids or best_sim < config.new_centroid_similarity_threshold:
        new_centroid = Centroid(embedding=post.copy(), weight=1.0, last_active=now)
        if len(centroids) < config.centroid_max_count:
            return list(centroids) + [new_centroid]
        # Evict least-active (oldest last_active timestamp)
        evict_idx = min(range(len(centroids)), key=lambda i: centroids[i].last_active)
        return [
            new_centroid if i == evict_idx else c for i, c in enumerate(centroids)
        ]

    # EMA pull on nearest centroid
    alpha = config.centroid_alpha
    target = centroids[best_idx]
    new_emb = (1.0 - alpha) * target.embedding + alpha * post
    updated = Centroid(
        embedding=new_emb,
        weight=target.weight + 1.0,
        last_active=now,
    )
    return [updated if i == best_idx else c for i, c in enumerate(centroids)]


def decayed_salience(
    base_salience: float,
    now: float,
    last_reactivated_at: float,
    lambda_per_unit: float,
) -> float:
    """ACT-R-flavored exponential decay.

    Time units must match the lambda's units (e.g., hours if lambda is per-hour).
    Half-life = ln(2) / lambda.
    """
    if now < last_reactivated_at:
        raise ValueError(
            f"now ({now}) cannot be earlier than last_reactivated_at "
            f"({last_reactivated_at})"
        )
    delta = now - last_reactivated_at
    return base_salience * math.exp(-lambda_per_unit * delta)


def reactivate_trace(
    base_salience: float,
    now: float,
    similarity: float,
    bump_strength: float,
) -> tuple[float, float]:
    """Apply a reactivation bump on retrieval.

    Returns (new_base_salience, new_last_reactivated_at).
    Bump magnitude scales with similarity to the triggering query.
    """
    bump = bump_strength * max(0.0, min(1.0, similarity))
    return base_salience + bump, now


def weighted_belief_sample(
    beliefs: list[Belief],
    n: int,
    now: float,
    config: "ConsolidationConfig",
    seed: int | None = None,
) -> list[Belief]:
    """Sample without replacement, weighted by confidence^a * recency^b.

    Recency = exp(-decay * (now - last_updated_at)), with decay tuned so
    7-day-old beliefs get ~0.5 recency. Sampling without replacement so
    no duplicates appear in the consolidation prompt.
    """
    from doomscroll.config import ConsolidationConfig  # local to avoid cycles

    assert isinstance(config, ConsolidationConfig)
    if not beliefs:
        return []
    if n >= len(beliefs):
        return list(beliefs)

    rng = np.random.default_rng(seed)
    confidences = np.array(
        [max(EPSILON, b.confidence) for b in beliefs], dtype=np.float64
    )
    # Recency decay: half-life of 7 days -> lambda = ln(2)/(7*24*3600)
    recency_lambda = math.log(2.0) / (7.0 * 24.0 * 3600.0)
    ages = np.array(
        [max(0.0, now - b.last_updated_at) for b in beliefs], dtype=np.float64
    )
    recencies = np.exp(-recency_lambda * ages)

    weights = (confidences**config.confidence_weight_alpha) * (
        recencies**config.recency_weight_alpha
    )
    weights = weights / weights.sum()

    indices = rng.choice(len(beliefs), size=n, replace=False, p=weights)
    return [beliefs[i] for i in indices]


def decay_centroid_weights(
    centroids: list[Centroid],
    now: float,
    config: MemoryConfig,
) -> list[Centroid]:
    """Decay centroid weights based on time since last_active.

    Used at session boundaries to let unused centroids fade.
    """
    out = []
    for c in centroids:
        delta = max(0.0, now - c.last_active)
        decayed_weight = c.weight * math.exp(
            -config.centroid_weight_decay_lambda * delta
        )
        out.append(replace(c, weight=decayed_weight))
    return out
