"""Tests for memory primitives.

Pure functions, deterministic. If these fail, the agent's perception layer
is broken and any downstream measurement is meaningless.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from doomscroll.config import (
    AGENT_VARIANTS,
    ConsolidationConfig,
    DistortionConfig,
    MemoryConfig,
)
from doomscroll.memory import (
    Belief,
    Centroid,
    decay_centroid_weights,
    decayed_salience,
    max_belief_alignment,
    max_centroid_similarity,
    reactivate_trace,
    salience_score,
    update_centroids_on_engagement,
    weighted_belief_sample,
)


def _c(emb, weight=1.0, last_active=0.0) -> Centroid:
    return Centroid(
        embedding=np.asarray(emb, dtype=np.float64),
        weight=weight,
        last_active=last_active,
    )


def _b(emb, confidence=0.5, id_=0, last_updated_at=0.0) -> Belief:
    return Belief(
        id=id_,
        text=f"belief-{id_}",
        embedding=np.asarray(emb, dtype=np.float64),
        confidence=confidence,
        last_updated_at=last_updated_at,
    )


class TestMaxCentroidSimilarity:
    def test_empty_returns_minus_one(self):
        sim, idx = max_centroid_similarity(np.array([1.0, 0.0]), [])
        assert sim == -1.0
        assert idx == -1

    def test_returns_best_match(self):
        post = np.array([1.0, 0.0])
        centroids = [
            _c([0.0, 1.0]),
            _c([0.9, 0.1]),
            _c([-1.0, 0.0]),
        ]
        sim, idx = max_centroid_similarity(post, centroids)
        assert idx == 1
        assert sim == pytest.approx(0.9 / math.sqrt(0.82), rel=1e-9)


class TestSalienceScore:
    def test_cold_start_high_novelty_engages(self):
        # Empty centroids, no beliefs, novelty_hunger=0.7 -> salience ~= 0.7
        post = np.array([1.0, 0.0])
        s = salience_score(
            post_embedding=post,
            centroids=[],
            beliefs=[],
            mood=0.0,
            novelty_hunger=0.7,
            distortion=AGENT_VARIANTS["balanced"],
            estimated_valence=0.0,
        )
        assert s == pytest.approx(0.7, abs=1e-9)

    def test_post_matches_centroid_lower_novelty(self):
        # When post matches an existing centroid, novelty term is small
        post = np.array([1.0, 0.0])
        centroids = [_c([1.0, 0.0], weight=10.0)]
        s = salience_score(
            post_embedding=post,
            centroids=centroids,
            beliefs=[],
            mood=0.0,
            novelty_hunger=0.7,
            distortion=AGENT_VARIANTS["balanced"],
        )
        # Novelty term ~ 0 (perfect match), interest term = 1.0
        assert s == pytest.approx(1.0, abs=1e-9)

    def test_confirmation_bias_boosts_belief_aligned_post(self):
        post = np.array([1.0, 0.0])
        belief = _b([1.0, 0.0], confidence=1.0)
        balanced = AGENT_VARIANTS["balanced"]
        biased = AGENT_VARIANTS["high_confirmation_bias"]

        s_balanced = salience_score(
            post_embedding=post,
            centroids=[],
            beliefs=[belief],
            mood=0.0,
            novelty_hunger=0.7,
            distortion=balanced,
        )
        s_biased = salience_score(
            post_embedding=post,
            centroids=[],
            beliefs=[belief],
            mood=0.0,
            novelty_hunger=0.7,
            distortion=biased,
        )
        assert s_biased > s_balanced
        # Difference should equal the confirmation_bias delta * alignment (1.0)
        assert s_biased - s_balanced == pytest.approx(
            biased.confirmation_bias - balanced.confirmation_bias, abs=1e-9
        )

    def test_novelty_term_zero_with_perfect_match(self):
        # interest term = 1, novelty term = 0
        post = np.array([0.0, 1.0])
        centroids = [_c([0.0, 1.0], weight=5.0)]
        s = salience_score(
            post_embedding=post,
            centroids=centroids,
            beliefs=[],
            mood=0.0,
            novelty_hunger=1.0,
            distortion=AGENT_VARIANTS["balanced"],
        )
        # interest = 1, novelty = 0, others = 0
        assert s == pytest.approx(1.0, abs=1e-9)

    def test_no_nan_with_zero_post_embedding(self):
        post = np.zeros(4)
        centroids = [_c([1.0, 0.0, 0.0, 0.0], weight=3.0)]
        beliefs = [_b([1.0, 0.0, 0.0, 0.0], confidence=0.8)]
        s = salience_score(
            post_embedding=post,
            centroids=centroids,
            beliefs=beliefs,
            mood=0.5,
            novelty_hunger=0.7,
            distortion=AGENT_VARIANTS["balanced"],
            estimated_valence=0.3,
        )
        assert math.isfinite(s)


class TestUpdateCentroids:
    def test_first_engagement_spawns_centroid(self):
        cfg = MemoryConfig()
        post = np.array([1.0, 0.0])
        result = update_centroids_on_engagement([], post, now=10.0, config=cfg)
        assert len(result) == 1
        assert result[0].weight == 1.0
        assert result[0].last_active == 10.0
        np.testing.assert_array_equal(result[0].embedding, post)

    def test_similar_post_pulls_existing_centroid(self):
        cfg = MemoryConfig(centroid_alpha=0.1, new_centroid_similarity_threshold=0.5)
        existing = [_c([1.0, 0.0], weight=5.0, last_active=0.0)]
        post = np.array([0.8, 0.6])  # similar enough
        result = update_centroids_on_engagement(existing, post, now=20.0, config=cfg)
        assert len(result) == 1
        # EMA pull: 0.9 * [1,0] + 0.1 * [0.8, 0.6] = [0.98, 0.06]
        np.testing.assert_allclose(result[0].embedding, [0.98, 0.06], atol=1e-9)
        assert result[0].weight == 6.0
        assert result[0].last_active == 20.0

    def test_dissimilar_post_spawns_new_centroid(self):
        cfg = MemoryConfig(new_centroid_similarity_threshold=0.55)
        existing = [_c([1.0, 0.0], weight=5.0, last_active=0.0)]
        post = np.array([0.0, 1.0])  # orthogonal -> spawn new
        result = update_centroids_on_engagement(existing, post, now=20.0, config=cfg)
        assert len(result) == 2
        np.testing.assert_array_equal(result[1].embedding, post)

    def test_eviction_at_max_count(self):
        cfg = MemoryConfig(
            centroid_max_count=3, new_centroid_similarity_threshold=0.55
        )
        existing = [
            _c([1.0, 0.0], weight=5.0, last_active=10.0),
            _c([0.0, 1.0], weight=3.0, last_active=5.0),  # least recent -> evicted
            _c([1.0, 1.0], weight=4.0, last_active=20.0),
        ]
        post = np.array([-1.0, 0.0])
        result = update_centroids_on_engagement(existing, post, now=30.0, config=cfg)
        assert len(result) == 3
        # Index 1 (the one with last_active=5.0) should be replaced
        np.testing.assert_array_equal(result[1].embedding, post)
        assert result[1].weight == 1.0

    def test_immutability(self):
        cfg = MemoryConfig()
        existing = [_c([1.0, 0.0], weight=5.0)]
        post = np.array([0.9, 0.1])
        result = update_centroids_on_engagement(existing, post, now=10.0, config=cfg)
        # Original list and centroid should be untouched
        assert existing[0].weight == 5.0
        assert result is not existing


class TestDecayedSalience:
    def test_no_time_no_decay(self):
        assert decayed_salience(1.0, now=10.0, last_reactivated_at=10.0,
                                lambda_per_unit=0.1) == pytest.approx(1.0)

    def test_one_half_life(self):
        # half-life = ln(2)/lambda; after 1 half-life, salience = 0.5 * base
        lam = math.log(2)
        assert decayed_salience(1.0, now=1.0, last_reactivated_at=0.0,
                                lambda_per_unit=lam) == pytest.approx(0.5, abs=1e-9)

    def test_negative_time_raises(self):
        with pytest.raises(ValueError, match="cannot be earlier"):
            decayed_salience(1.0, now=5.0, last_reactivated_at=10.0,
                             lambda_per_unit=0.1)

    def test_monotonic_decay(self):
        prev = 1.0
        for t in [1.0, 5.0, 10.0, 100.0]:
            cur = decayed_salience(1.0, now=t, last_reactivated_at=0.0,
                                   lambda_per_unit=0.1)
            assert cur < prev
            prev = cur


class TestReactivateTrace:
    def test_bump_proportional_to_similarity(self):
        new_sal, new_t = reactivate_trace(
            base_salience=0.5, now=10.0, similarity=1.0, bump_strength=0.5
        )
        assert new_sal == pytest.approx(1.0)
        assert new_t == 10.0

    def test_zero_similarity_no_bump(self):
        new_sal, _ = reactivate_trace(0.5, now=10.0, similarity=0.0, bump_strength=0.5)
        assert new_sal == pytest.approx(0.5)

    def test_negative_similarity_clamped(self):
        new_sal, _ = reactivate_trace(0.5, now=10.0, similarity=-0.5, bump_strength=0.5)
        assert new_sal == pytest.approx(0.5)


class TestWeightedBeliefSample:
    def test_empty_returns_empty(self):
        assert weighted_belief_sample([], n=5, now=0.0, config=ConsolidationConfig()) == []

    def test_n_greater_than_or_equal_returns_all(self):
        beliefs = [_b([1.0, 0.0], id_=i) for i in range(3)]
        result = weighted_belief_sample(beliefs, n=3, now=0.0,
                                        config=ConsolidationConfig())
        assert len(result) == 3
        result2 = weighted_belief_sample(beliefs, n=10, now=0.0,
                                         config=ConsolidationConfig())
        assert len(result2) == 3

    def test_no_duplicates(self):
        beliefs = [_b([float(i), 0.0], id_=i, confidence=0.5) for i in range(20)]
        result = weighted_belief_sample(beliefs, n=10, now=0.0,
                                        config=ConsolidationConfig(), seed=42)
        ids = [b.id for b in result]
        assert len(set(ids)) == len(ids)

    def test_high_confidence_more_likely(self):
        # Run many trials with one very-high-confidence belief; it should be
        # sampled in the vast majority of runs.
        beliefs = [_b([float(i), 0.0], id_=i, confidence=0.01) for i in range(10)]
        beliefs[5] = _b([5.0, 0.0], id_=5, confidence=1.0)
        cfg = ConsolidationConfig(confidence_weight_alpha=2.0, recency_weight_alpha=0.0)
        hits = 0
        trials = 100
        for s in range(trials):
            result = weighted_belief_sample(beliefs, n=2, now=0.0, config=cfg, seed=s)
            if any(b.id == 5 for b in result):
                hits += 1
        assert hits > 80, f"high-confidence belief sampled only {hits}/{trials} times"

    def test_low_confidence_belief_can_still_be_sampled(self):
        # Anti-entrenchment: low-confidence beliefs are not impossible.
        beliefs = [_b([float(i), 0.0], id_=i, confidence=0.01) for i in range(10)]
        beliefs[5] = _b([5.0, 0.0], id_=5, confidence=1.0)
        cfg = ConsolidationConfig()
        hits_low = 0
        for s in range(200):
            result = weighted_belief_sample(beliefs, n=3, now=0.0, config=cfg, seed=s)
            for b in result:
                if b.id != 5 and b.confidence < 0.05:
                    hits_low += 1
                    break
        # With weighted sampling (not top-N), low-confidence beliefs MUST appear sometimes.
        assert hits_low > 0


class TestDecayCentroidWeights:
    def test_no_time_no_decay(self):
        cfg = MemoryConfig(centroid_weight_decay_lambda=0.1)
        centroids = [_c([1.0, 0.0], weight=5.0, last_active=10.0)]
        result = decay_centroid_weights(centroids, now=10.0, config=cfg)
        assert result[0].weight == pytest.approx(5.0)

    def test_decay_reduces_weight(self):
        cfg = MemoryConfig(centroid_weight_decay_lambda=math.log(2))
        centroids = [_c([1.0, 0.0], weight=4.0, last_active=0.0)]
        # 1 half-life elapsed
        result = decay_centroid_weights(centroids, now=1.0, config=cfg)
        assert result[0].weight == pytest.approx(2.0, abs=1e-9)

    def test_immutable(self):
        cfg = MemoryConfig(centroid_weight_decay_lambda=0.1)
        centroids = [_c([1.0, 0.0], weight=5.0, last_active=0.0)]
        result = decay_centroid_weights(centroids, now=100.0, config=cfg)
        assert centroids[0].weight == 5.0  # original untouched
        assert result is not centroids
