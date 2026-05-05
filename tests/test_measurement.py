"""Tests for the measurement module.

If these pass, the experiment's numbers can be trusted.
If these fail or are missing, the experiment produces garbage.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from doomscroll.measurement import (
    belief_drift,
    cosine_distance,
    cross_agent_divergence,
    shannon_entropy,
    topic_entropy,
)


class TestShannonEntropy:
    def test_uniform_distribution_returns_log_n(self):
        for n in [2, 5, 10, 100]:
            probs = np.ones(n) / n
            assert shannon_entropy(probs) == pytest.approx(math.log(n), rel=1e-9)

    def test_single_outcome_returns_zero(self):
        assert shannon_entropy(np.array([1.0])) == 0.0
        assert shannon_entropy(np.array([1.0, 0.0, 0.0, 0.0])) == 0.0

    def test_empty_input_returns_zero(self):
        assert shannon_entropy(np.array([])) == 0.0

    def test_unnormalized_counts_are_normalized(self):
        # entropy is invariant to scale; counts should equal proportions
        assert shannon_entropy(np.array([10.0, 10.0])) == pytest.approx(math.log(2))
        assert shannon_entropy(np.array([1.0, 1.0])) == pytest.approx(math.log(2))

    def test_all_zeros_returns_zero(self):
        assert shannon_entropy(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_negative_values_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            shannon_entropy(np.array([0.5, -0.1, 0.6]))

    def test_monotonic_with_flattening(self):
        # As distribution flattens, entropy increases.
        peaked = np.array([0.9, 0.05, 0.025, 0.025])
        flatter = np.array([0.6, 0.2, 0.1, 0.1])
        flat = np.array([0.25, 0.25, 0.25, 0.25])
        assert shannon_entropy(peaked) < shannon_entropy(flatter) < shannon_entropy(flat)


class TestCosineDistance:
    def test_identical_vectors_distance_zero(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_vectors_distance_two(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(v, -v) == pytest.approx(2.0, abs=1e-9)

    def test_orthogonal_vectors_distance_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_symmetric(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 1.0, 2.0])
        assert cosine_distance(a, b) == cosine_distance(b, a)

    def test_zero_vector_returns_zero(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert cosine_distance(a, b) == 0.0

    def test_scale_invariant(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-9)


class TestTopicEntropy:
    def test_zero_embeddings_returns_zero(self):
        empty = np.zeros((0, 8))
        assert topic_entropy(empty, k=3) == 0.0

    def test_single_embedding_returns_zero(self):
        single = np.array([[1.0, 0.0, 0.0]])
        assert topic_entropy(single, k=3) == 0.0

    def test_n_less_than_k_returns_log_n(self):
        # If we have fewer beliefs than topics, each is its own topic.
        emb = np.eye(3)
        assert topic_entropy(emb, k=10) == pytest.approx(math.log(3), abs=1e-9)

    def test_perfectly_separated_clusters_return_log_k(self):
        # 3 tight clusters of 10 each, well-separated -> uniform cluster sizes -> log(3)
        rng = np.random.default_rng(42)
        cluster_centers = np.array(
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        )
        points = []
        for c in cluster_centers:
            for _ in range(10):
                points.append(c + rng.normal(0, 0.05, size=3))
        emb = np.array(points)
        result = topic_entropy(emb, k=3, seed=0)
        assert result == pytest.approx(math.log(3), abs=0.01)

    def test_concentrated_distribution_lower_entropy(self):
        # 25 points near one center, 5 near another -> peaked distribution
        rng = np.random.default_rng(7)
        center_a = np.array([10.0, 0.0])
        center_b = np.array([0.0, 10.0])
        points = [center_a + rng.normal(0, 0.05, size=2) for _ in range(25)]
        points += [center_b + rng.normal(0, 0.05, size=2) for _ in range(5)]
        peaked_emb = np.array(points)

        balanced_points = [center_a + rng.normal(0, 0.05, size=2) for _ in range(15)]
        balanced_points += [center_b + rng.normal(0, 0.05, size=2) for _ in range(15)]
        balanced_emb = np.array(balanced_points)

        peaked_entropy = topic_entropy(peaked_emb, k=2, seed=1)
        balanced_entropy = topic_entropy(balanced_emb, k=2, seed=1)
        assert peaked_entropy < balanced_entropy

    def test_invalid_k_raises(self):
        emb = np.array([[1.0, 0.0]])
        with pytest.raises(ValueError, match="k must be positive"):
            topic_entropy(emb, k=0)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="2D"):
            topic_entropy(np.array([1.0, 2.0, 3.0]), k=2)


class TestBeliefDrift:
    def test_identical_beliefs_zero_drift(self):
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        assert belief_drift(emb, emb) == pytest.approx(0.0, abs=1e-9)

    def test_symmetric(self):
        a = np.array([[1.0, 0.0], [0.5, 0.5]])
        b = np.array([[0.0, 1.0], [0.7, 0.3]])
        assert belief_drift(a, b) == pytest.approx(belief_drift(b, a), abs=1e-9)

    def test_opposite_centroids_max_drift(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        assert belief_drift(a, b) == pytest.approx(2.0, abs=1e-9)

    def test_orthogonal_centroids_drift_one(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        assert belief_drift(a, b) == pytest.approx(1.0, abs=1e-9)

    def test_empty_set_returns_zero(self):
        empty = np.zeros((0, 4))
        nonempty = np.array([[1.0, 2.0, 3.0, 4.0]])
        assert belief_drift(empty, nonempty) == 0.0
        assert belief_drift(nonempty, empty) == 0.0

    def test_dimension_mismatch_raises(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="dimensions must match"):
            belief_drift(a, b)

    def test_drift_in_zero_two_range(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(20, 16))
        b = rng.normal(size=(20, 16))
        d = belief_drift(a, b)
        assert 0.0 <= d <= 2.0


class TestCrossAgentDivergence:
    def test_symmetric_keys(self):
        agents = {
            "a": np.array([[1.0, 0.0]]),
            "b": np.array([[0.0, 1.0]]),
        }
        result = cross_agent_divergence(agents)
        assert result[("a", "b")] == result[("b", "a")]

    def test_self_pairs_omitted(self):
        agents = {
            "a": np.array([[1.0, 0.0]]),
            "b": np.array([[0.0, 1.0]]),
        }
        result = cross_agent_divergence(agents)
        assert ("a", "a") not in result
        assert ("b", "b") not in result

    def test_three_agents_six_entries(self):
        # 3 agents -> 3 unordered pairs -> 6 directed entries
        agents = {
            "a": np.array([[1.0, 0.0]]),
            "b": np.array([[0.0, 1.0]]),
            "c": np.array([[-1.0, 0.0]]),
        }
        result = cross_agent_divergence(agents)
        assert len(result) == 6

    def test_identical_agents_zero_divergence(self):
        emb = np.array([[1.0, 2.0, 3.0]])
        agents = {"a": emb, "b": emb.copy()}
        result = cross_agent_divergence(agents)
        assert result[("a", "b")] == pytest.approx(0.0, abs=1e-9)
