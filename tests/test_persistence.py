"""Tests for the sqlite persistence layer.

Round-trip every entity through an in-memory database. If embeddings come
back wrong, the salience gate is broken on every subsequent session.
"""

from __future__ import annotations

import numpy as np
import pytest

from doomscroll.consolidation import BeliefAction, BeliefUpdate
from doomscroll.memory import Belief, Centroid
from doomscroll.persistence import Store


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


class TestSessions:
    def test_start_returns_id(self, store):
        sid = store.start_session(started_at=100.0)
        assert sid > 0
        assert store.session_count() == 1

    def test_end_session_updates_counts(self, store):
        sid = store.start_session(100.0)
        store.end_session(sid, ended_at=200.0, posts_seen=50,
                          posts_engaged=10, consolidation_run=True)
        cur = store._conn.execute("SELECT * FROM sessions WHERE id = ?", (sid,))
        row = cur.fetchone()
        assert row["posts_seen"] == 50
        assert row["posts_engaged"] == 10
        assert row["consolidation_run"] == 1
        assert row["ended_at"] == 200.0


class TestPostsSeen:
    def test_record_and_count(self, store):
        sid = store.start_session(0.0)
        emb = np.array([1.0, 0.0, 0.0])
        store.record_post_seen(sid, "p1", 1.0, 0.8, "engaged", emb)
        store.record_post_seen(sid, "p2", 2.0, 0.1, "skipped", emb)
        store.record_post_seen(sid, "p3", 3.0, 0.7, "engaged", emb)
        seen, engaged = store.post_seen_count(sid)
        assert seen == 3
        assert engaged == 2

    def test_invalid_decision_raises(self, store):
        sid = store.start_session(0.0)
        with pytest.raises(ValueError, match="decision"):
            store.record_post_seen(sid, "p1", 1.0, 0.5, "maybe", None)

    def test_null_embedding_allowed(self, store):
        sid = store.start_session(0.0)
        store.record_post_seen(sid, "p1", 1.0, 0.0, "skipped", None)
        seen, _ = store.post_seen_count(sid)
        assert seen == 1


class TestTraces:
    def test_round_trip(self, store):
        sid = store.start_session(0.0)
        emb = np.array([0.1, 0.2, 0.3, 0.4])
        tid = store.insert_trace(
            session_id=sid,
            post_id="p1",
            created_at=10.0,
            last_reactivated_at=10.0,
            base_salience=0.7,
            valence=0.3,
            novelty=0.5,
            fragment="this lingered",
            embedding=emb,
        )
        assert tid > 0
        traces = store.load_traces()
        assert len(traces) == 1
        t = traces[0]
        assert t.fragment == "this lingered"
        assert t.base_salience == 0.7
        np.testing.assert_allclose(t.embedding, emb, atol=1e-6)

    def test_reactivation_update(self, store):
        sid = store.start_session(0.0)
        tid = store.insert_trace(sid, "p1", 0.0, 0.0, 0.5, 0.0, 0.5, "x",
                                 np.array([1.0]))
        store.update_trace_reactivation(tid, new_base_salience=0.9,
                                        new_last_reactivated_at=100.0)
        t = store.load_traces()[0]
        assert t.base_salience == 0.9
        assert t.last_reactivated_at == 100.0

    def test_load_traces_since(self, store):
        sid = store.start_session(0.0)
        for ts in [10.0, 50.0, 100.0]:
            store.insert_trace(sid, f"p{ts}", ts, ts, 0.5, 0.0, 0.5, "x",
                               np.array([1.0]))
        recent = store.load_traces(since_ts=40.0)
        assert len(recent) == 2

    def test_session_filter(self, store):
        s1 = store.start_session(0.0)
        s2 = store.start_session(100.0)
        store.insert_trace(s1, "p1", 1.0, 1.0, 0.5, 0.0, 0.5, "a", np.array([1.0]))
        store.insert_trace(s2, "p2", 101.0, 101.0, 0.5, 0.0, 0.5, "b",
                           np.array([1.0]))
        assert len(store.load_session_traces(s1)) == 1
        assert len(store.load_session_traces(s2)) == 1


class TestBeliefs:
    def test_round_trip(self, store):
        emb = np.array([0.5, 0.5, 0.5])
        bid = store.insert_belief("things are fine", emb, 0.7, now=10.0)
        beliefs = store.load_active_beliefs()
        assert len(beliefs) == 1
        b = beliefs[0]
        assert b.id == bid
        assert b.text == "things are fine"
        assert b.confidence == 0.7
        np.testing.assert_allclose(b.embedding, emb, atol=1e-6)

    def test_update_confidence(self, store):
        bid = store.insert_belief("x", np.array([1.0]), 0.5, now=0.0)
        store.update_belief(bid, confidence=0.9, now=10.0)
        b = store.load_active_beliefs()[0]
        assert b.confidence == 0.9
        assert b.last_updated_at == 10.0

    def test_replace_text(self, store):
        bid = store.insert_belief("old", np.array([1.0]), 0.5, now=0.0)
        store.update_belief(bid, confidence=0.6, now=10.0, text="new")
        b = store.load_active_beliefs()[0]
        assert b.text == "new"

    def test_deactivate_excludes_from_load(self, store):
        bid = store.insert_belief("x", np.array([1.0]), 0.5, now=0.0)
        store.insert_belief("y", np.array([1.0]), 0.5, now=0.0)
        store.deactivate_belief(bid)
        beliefs = store.load_active_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].text == "y"

    def test_apply_belief_updates_new(self, store):
        embed = lambda txt: np.array([float(len(txt))])  # noqa: E731
        updates = [
            BeliefUpdate(action=BeliefAction.NEW, text="emergent",
                         target_id=None, new_confidence=0.6),
        ]
        store.apply_belief_updates(updates, embed_text=embed, now=10.0)
        beliefs = store.load_active_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].text == "emergent"

    def test_apply_belief_updates_drop(self, store):
        bid = store.insert_belief("doomed", np.array([1.0]), 0.5, now=0.0)
        embed = lambda txt: np.array([1.0])  # noqa: E731
        updates = [
            BeliefUpdate(action=BeliefAction.DROP, text=None,
                         target_id=bid, new_confidence=None)
        ]
        store.apply_belief_updates(updates, embed_text=embed, now=10.0)
        assert store.load_active_beliefs() == []

    def test_apply_belief_updates_strengthen(self, store):
        bid = store.insert_belief("x", np.array([1.0]), 0.3, now=0.0)
        embed = lambda txt: np.array([1.0])  # noqa: E731
        updates = [
            BeliefUpdate(action=BeliefAction.STRENGTHEN, text=None,
                         target_id=bid, new_confidence=0.85)
        ]
        store.apply_belief_updates(updates, embed_text=embed, now=10.0)
        b = store.load_active_beliefs()[0]
        assert b.confidence == 0.85


class TestInterestSnapshot:
    def test_save_and_load(self, store):
        sid = store.start_session(0.0)
        centroids = [
            Centroid(embedding=np.array([1.0, 0.0]), weight=3.0, last_active=10.0),
            Centroid(embedding=np.array([0.0, 1.0]), weight=5.0, last_active=20.0),
        ]
        store.save_interest_snapshot(sid, ts=20.0, centroids=centroids,
                                     mood=-0.3, novelty_hunger=0.65)
        result = store.load_latest_interest_snapshot()
        assert result is not None
        loaded_cs, mood, nh = result
        assert len(loaded_cs) == 2
        np.testing.assert_allclose(loaded_cs[0].embedding, [1.0, 0.0])
        assert loaded_cs[0].weight == 3.0
        assert mood == -0.3
        assert nh == 0.65

    def test_empty_returns_none(self, store):
        assert store.load_latest_interest_snapshot() is None

    def test_latest_wins(self, store):
        sid = store.start_session(0.0)
        store.save_interest_snapshot(sid, 1.0, [], mood=0.1, novelty_hunger=0.7)
        store.save_interest_snapshot(sid, 2.0, [], mood=0.5, novelty_hunger=0.4)
        _, mood, nh = store.load_latest_interest_snapshot()
        assert mood == 0.5
        assert nh == 0.4


class TestPersistenceAcrossConnections:
    def test_data_survives_close_and_reopen(self, tmp_path):
        path = tmp_path / "agent.db"
        s1 = Store(path)
        sid = s1.start_session(0.0)
        s1.insert_trace(sid, "p1", 0.0, 0.0, 0.5, 0.0, 0.5, "fragment",
                        np.array([1.0, 2.0, 3.0]))
        s1.insert_belief("belief", np.array([0.0, 1.0]), 0.7, now=0.0)
        s1.close()

        s2 = Store(path)
        traces = s2.load_traces()
        beliefs = s2.load_active_beliefs()
        assert len(traces) == 1
        assert traces[0].fragment == "fragment"
        assert len(beliefs) == 1
        np.testing.assert_allclose(beliefs[0].embedding, [0.0, 1.0], atol=1e-6)
        s2.close()
