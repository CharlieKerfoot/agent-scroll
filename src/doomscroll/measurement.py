"""Measurement primitives for the doomscroll experiment.

These functions are the load-bearing surface of the experiment: if they are
buggy, a week of agent runs produces confidently-wrong numbers. Every function
here is pure, deterministic, and exhaustively unit-tested.

Coverage:
- shannon_entropy(probs)            -> entropy in nats
- topic_entropy(embeddings, k)      -> Shannon entropy of cluster sizes
- belief_drift(emb_t0, emb_tn)      -> cosine distance between centroids
- cross_agent_divergence(beliefs)   -> pairwise drift matrix
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPSILON = 1e-12


def shannon_entropy(probs: NDArray[np.floating]) -> float:
    """Shannon entropy in nats.

    Uniform distribution over n outcomes returns ln(n).
    Single-outcome distribution returns 0.
    Empty input returns 0 (degenerate but well-defined).
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.size == 0:
        return 0.0
    if not np.all(p >= -EPSILON):
        raise ValueError(f"probabilities must be non-negative; got min={p.min()}")
    total = p.sum()
    if total < EPSILON:
        return 0.0
    p = p / total
    nonzero = p[p > EPSILON]
    return float(-np.sum(nonzero * np.log(nonzero)))


def _normalize_rows(matrix: NDArray[np.floating]) -> NDArray[np.float64]:
    m = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < EPSILON, 1.0, norms)
    return m / norms


def topic_entropy(
    embeddings: NDArray[np.floating],
    k: int,
    seed: int = 0,
    max_iter: int = 50,
) -> float:
    """Shannon entropy of cluster-size distribution after k-means clustering.

    Higher entropy = beliefs spread across many topics.
    Lower entropy = beliefs concentrated in few topics (the doomscroll prediction).

    Edge cases:
    - 0 embeddings -> 0
    - 1 embedding  -> 0 (single belief, single topic)
    - n < k        -> entropy treats each belief as its own topic (returns log(n))
    """
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 2D; got shape {emb.shape}")
    if k <= 0:
        raise ValueError(f"k must be positive; got {k}")
    n = emb.shape[0]
    if n == 0:
        return 0.0
    if n == 1:
        return 0.0
    if n <= k:
        return float(np.log(n))

    emb = _normalize_rows(emb)
    rng = np.random.default_rng(seed)
    initial_indices = rng.choice(n, size=k, replace=False)
    centers = emb[initial_indices].copy()

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        similarities = emb @ centers.T
        new_assignments = np.argmax(similarities, axis=1)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments
        for j in range(k):
            members = emb[assignments == j]
            if members.shape[0] > 0:
                centers[j] = members.mean(axis=0)
        centers = _normalize_rows(centers)

    counts = np.bincount(assignments, minlength=k)
    return shannon_entropy(counts.astype(np.float64))


def cosine_distance(a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
    """Cosine distance in [0, 2]. 0 = identical direction, 2 = opposite."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a64)
    nb = np.linalg.norm(b64)
    if na < EPSILON or nb < EPSILON:
        return 0.0
    sim = float(np.dot(a64, b64) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    return 1.0 - sim


def belief_drift(
    embeddings_t0: NDArray[np.floating],
    embeddings_tn: NDArray[np.floating],
) -> float:
    """Cosine distance between mean belief embeddings at two timepoints.

    Returns 0.0 when belief sets have identical centroids.
    Symmetric: belief_drift(a, b) == belief_drift(b, a).
    Returns 0.0 if either set is empty (no drift definable).
    """
    e0 = np.asarray(embeddings_t0, dtype=np.float64)
    en = np.asarray(embeddings_tn, dtype=np.float64)
    if e0.size == 0 or en.size == 0:
        return 0.0
    if e0.ndim != 2 or en.ndim != 2:
        raise ValueError("embeddings must be 2D arrays")
    if e0.shape[1] != en.shape[1]:
        raise ValueError(
            f"embedding dimensions must match; got {e0.shape[1]} vs {en.shape[1]}"
        )
    centroid_0 = e0.mean(axis=0)
    centroid_n = en.mean(axis=0)
    return cosine_distance(centroid_0, centroid_n)


def cross_agent_divergence(
    agent_belief_sets: dict[str, NDArray[np.floating]],
) -> dict[tuple[str, str], float]:
    """Pairwise belief_drift between every pair of agents.

    Returns symmetric dict: result[(a, b)] == result[(b, a)].
    Self-pairs omitted.
    """
    names = sorted(agent_belief_sets.keys())
    result: dict[tuple[str, str], float] = {}
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            d = belief_drift(agent_belief_sets[name_a], agent_belief_sets[name_b])
            result[(name_a, name_b)] = d
            result[(name_b, name_a)] = d
    return result
