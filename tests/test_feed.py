"""Tests for the feed adapter layer.

Verifies the experimental invariant: same snapshot file -> same posts in
the same order for every agent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from doomscroll.embedding import FakeEmbedder
from doomscroll.feed import (
    SNAPSHOT_SCHEMA_VERSION,
    JSONSnapshotFeed,
    RawPost,
    SyntheticFeed,
    capture_moltbook_to_raw_json,
    deduplicate_posts,
    prepare_snapshot,
)


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=16)


# ---------------- deduplicate_posts ----------------

class TestDeduplicate:
    def test_no_duplicates_unchanged(self):
        posts = [
            RawPost("a", "u1", 0.0, "first text"),
            RawPost("b", "u2", 1.0, "second text"),
        ]
        assert len(deduplicate_posts(posts)) == 2

    def test_exact_duplicates_collapsed(self):
        posts = [
            RawPost("a", "u1", 0.0, "same content"),
            RawPost("b", "u2", 1.0, "same content"),  # different id, same text
            RawPost("c", "u3", 2.0, "different"),
        ]
        result = deduplicate_posts(posts)
        assert len(result) == 2
        # First occurrence wins
        assert result[0].id == "a"

    def test_whitespace_normalized(self):
        # Leading/trailing whitespace should not produce a "different" hash
        posts = [
            RawPost("a", "u", 0.0, "hello"),
            RawPost("b", "u", 1.0, "  hello  "),
        ]
        assert len(deduplicate_posts(posts)) == 1

    def test_case_sensitive(self):
        # Capitalization is meaningful for social-media text
        posts = [
            RawPost("a", "u", 0.0, "hello"),
            RawPost("b", "u", 1.0, "HELLO"),
        ]
        assert len(deduplicate_posts(posts)) == 2

    def test_empty_list(self):
        assert deduplicate_posts([]) == []


# ---------------- prepare_snapshot + JSONSnapshotFeed roundtrip ----------------

class TestSnapshotRoundtrip:
    def test_round_trip_basic(self, tmp_path, embedder):
        raw = [
            RawPost("p1", "alice", 100.0, "first thing"),
            RawPost("p2", "bob", 101.0, "second thing"),
            RawPost("p3", "carol", 102.0, "third thing"),
        ]
        path = prepare_snapshot(raw, embedder, tmp_path / "snap.json",
                                source="test", embedding_model="fake")
        assert path.exists()

        feed = JSONSnapshotFeed(path)
        assert len(feed) == 3
        posts = list(feed.iter_posts())
        assert posts[0].id == "p1"
        assert posts[0].text == "first thing"
        assert posts[0].author == "alice"
        assert posts[0].ts == 100.0
        # Embedding should match the embedder's output for that text
        np.testing.assert_allclose(
            posts[0].embedding, embedder.embed("first thing"), atol=1e-5
        )

    def test_dedupe_runs_during_prepare(self, tmp_path, embedder):
        raw = [
            RawPost("p1", "u1", 0.0, "duplicate"),
            RawPost("p2", "u2", 1.0, "duplicate"),  # should be dropped
            RawPost("p3", "u3", 2.0, "unique"),
        ]
        path = prepare_snapshot(raw, embedder, tmp_path / "s.json")
        feed = JSONSnapshotFeed(path)
        assert len(feed) == 2

    def test_metadata_preserved(self, tmp_path, embedder):
        raw = [RawPost("p1", "u", 0.0, "x")]
        path = prepare_snapshot(raw, embedder, tmp_path / "s.json",
                                source="moltbook", embedding_model="fake-16")
        with open(path) as f:
            blob = json.load(f)
        assert blob["metadata"]["source"] == "moltbook"
        assert blob["metadata"]["embedding_model"] == "fake-16"
        assert blob["metadata"]["post_count"] == 1
        assert blob["metadata"]["raw_count_before_dedup"] == 1

    def test_schema_version_mismatch_raises(self, tmp_path, embedder):
        raw = [RawPost("p1", "u", 0.0, "x")]
        path = prepare_snapshot(raw, embedder, tmp_path / "s.json")
        # Corrupt the version
        with open(path) as f:
            blob = json.load(f)
        blob["metadata"]["schema_version"] = 999
        with open(path, "w") as f:
            json.dump(blob, f)
        with pytest.raises(ValueError, match="schema version mismatch"):
            JSONSnapshotFeed(path)

    def test_missing_embedding_raises(self, tmp_path):
        # Hand-craft a snapshot with no embedding field
        path = tmp_path / "broken.json"
        blob = {
            "metadata": {
                "schema_version": SNAPSHOT_SCHEMA_VERSION,
                "source": "test",
                "embedding_model": "none",
                "post_count": 1,
                "raw_count_before_dedup": 1,
            },
            "posts": [{"id": "p1", "author": "u", "ts": 0.0, "text": "x"}],
        }
        with open(path, "w") as f:
            json.dump(blob, f)
        feed = JSONSnapshotFeed(path)
        with pytest.raises(ValueError, match="missing embedding"):
            list(feed.iter_posts())


# ---------------- Determinism invariant ----------------

class TestExperimentalInvariant:
    def test_same_snapshot_same_posts_for_every_agent(self, tmp_path, embedder):
        # The load-bearing invariant: cross-agent comparison requires
        # identical post sequences.
        raw = [RawPost(f"p{i}", "u", float(i), f"text {i}") for i in range(50)]
        path = prepare_snapshot(raw, embedder, tmp_path / "shared.json")

        # Three "agents" each open the same snapshot
        agent_a = list(JSONSnapshotFeed(path).iter_posts())
        agent_b = list(JSONSnapshotFeed(path).iter_posts())
        agent_c = list(JSONSnapshotFeed(path).iter_posts())

        assert len(agent_a) == len(agent_b) == len(agent_c) == 50
        for a, b, c in zip(agent_a, agent_b, agent_c):
            assert a.id == b.id == c.id
            assert a.text == b.text == c.text
            np.testing.assert_array_equal(a.embedding, b.embedding)
            np.testing.assert_array_equal(a.embedding, c.embedding)


# ---------------- SyntheticFeed ----------------

class TestSyntheticFeed:
    def test_produces_requested_count(self, embedder):
        feed = SyntheticFeed(embedder, n_posts=42, seed=0)
        assert len(feed) == 42
        posts = list(feed.iter_posts())
        assert len(posts) == 42

    def test_seed_makes_deterministic(self, embedder):
        a = list(SyntheticFeed(embedder, n_posts=10, seed=7).iter_posts())
        b = list(SyntheticFeed(embedder, n_posts=10, seed=7).iter_posts())
        for pa, pb in zip(a, b):
            assert pa.text == pb.text
            assert pa.author == pb.author

    def test_different_seeds_diverge(self, embedder):
        a = list(SyntheticFeed(embedder, n_posts=20, seed=1).iter_posts())
        b = list(SyntheticFeed(embedder, n_posts=20, seed=2).iter_posts())
        # At least some posts should differ
        differing = sum(1 for pa, pb in zip(a, b) if pa.text != pb.text)
        assert differing > 0

    def test_timestamps_are_monotonic(self, embedder):
        feed = SyntheticFeed(embedder, n_posts=20, seed=0,
                             start_ts=1000.0, post_interval_seconds=10.0)
        posts = list(feed.iter_posts())
        for prev, cur in zip(posts, posts[1:]):
            assert cur.ts > prev.ts
        assert posts[0].ts == 1000.0
        assert posts[-1].ts == 1000.0 + 19 * 10.0

    def test_embeddings_match_embedder(self, embedder):
        feed = SyntheticFeed(embedder, n_posts=5, seed=0)
        posts = list(feed.iter_posts())
        for p in posts:
            np.testing.assert_array_equal(p.embedding, embedder.embed(p.text))


# ---------------- MoltBook stub ----------------

class TestMoltBookStub:
    def test_capture_raises_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError, match="MoltBook crawler"):
            capture_moltbook_to_raw_json(tmp_path / "raw.json")
