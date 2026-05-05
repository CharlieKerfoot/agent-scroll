"""Tests for the agent loop.

Heavy use of FakeLLM + FakeEmbedder. The point is to verify the
orchestration: state transitions, store writes, and end-to-end flow.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from doomscroll.agent_loop import (
    AgentState,
    Post,
    ReactionResult,
    _parse_reaction,
    begin_session,
    end_session,
    estimate_valence,
    nightly_consolidation,
    react_to_post,
    run_session,
    session_step,
)
from doomscroll.config import (
    AGENT_VARIANTS,
    ConsolidationConfig,
    MemoryConfig,
)
from doomscroll.embedding import FakeEmbedder
from doomscroll.llm import FakeLLM
from doomscroll.persistence import Store


# ---------------- Fixtures ----------------

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "agent.db")
    yield s
    s.close()


@pytest.fixture
def embedder():
    return FakeEmbedder(dim=32)


def _make_post(id_: str, text: str, embedder: FakeEmbedder, ts: float = 0.0) -> Post:
    return Post(
        id=id_, author="someone", ts=ts, text=text, embedding=embedder.embed(text)
    )


# ---------------- estimate_valence ----------------

class TestEstimateValence:
    def test_neutral_text_zero(self):
        assert estimate_valence("hello world") == 0.0

    def test_excitement_positive(self):
        assert estimate_valence("amazing news!!!") > 0

    def test_negative_words_flip_sign(self):
        assert estimate_valence("this is terrible!!!") < 0

    def test_empty_string_zero(self):
        assert estimate_valence("") == 0.0

    def test_all_caps_intensifies(self):
        plain = estimate_valence("this is bad")
        shouty = estimate_valence("THIS IS BAD")
        assert abs(shouty) >= abs(plain)


# ---------------- _parse_reaction ----------------

class TestParseReaction:
    def test_clean_json(self):
        raw = json.dumps({"fragment": "huh", "valence": 0.4, "novelty": 0.7})
        r = _parse_reaction(raw)
        assert r.fragment == "huh"
        assert r.valence == 0.4
        assert r.novelty == 0.7

    def test_strips_code_fences(self):
        raw = '```json\n{"fragment": "ok", "valence": 0, "novelty": 0.5}\n```'
        r = _parse_reaction(raw)
        assert r.fragment == "ok"

    def test_malformed_returns_default(self):
        r = _parse_reaction("not json at all")
        assert r.fragment == "(unreadable reaction)"
        assert r.valence == 0.0
        assert r.novelty == 0.5

    def test_clamps_out_of_range(self):
        raw = json.dumps({"fragment": "x", "valence": 99.0, "novelty": -1.0})
        r = _parse_reaction(raw)
        assert r.valence == 1.0
        assert r.novelty == 0.0

    def test_caps_fragment_length(self):
        long_text = "a" * 500
        raw = json.dumps({"fragment": long_text, "valence": 0, "novelty": 0.5})
        r = _parse_reaction(raw)
        assert len(r.fragment) == 200

    def test_empty_fragment_replaced(self):
        raw = json.dumps({"fragment": "", "valence": 0, "novelty": 0.5})
        r = _parse_reaction(raw)
        assert r.fragment == "(empty reaction)"

    def test_missing_fields_defaults(self):
        raw = json.dumps({"fragment": "hi"})
        r = _parse_reaction(raw)
        assert r.fragment == "hi"
        assert r.valence == 0.0
        assert r.novelty == 0.5


# ---------------- session_step ----------------

class TestSessionStep:
    def test_low_salience_skips(self, store, embedder):
        sid = store.start_session(0.0)
        state = AgentState(
            centroids=[],
            mood=0.0,
            novelty_hunger=0.05,  # low novelty -> low salience for unknown posts
            salience_threshold=0.5,
        )
        post = _make_post("p1", "boring post", embedder)
        llm = FakeLLM()  # should not be called

        new_state = session_step(
            state=state,
            post=post,
            store=store,
            session_id=sid,
            distortion=AGENT_VARIANTS["balanced"],
            mem_config=MemoryConfig(),
            llm=llm,
            beliefs=[],
            now=10.0,
        )
        assert new_state.posts_engaged == 0
        assert llm.calls == []
        seen, engaged = store.post_seen_count(sid)
        assert seen == 1
        assert engaged == 0

    def test_high_salience_engages_and_writes_trace(self, store, embedder):
        sid = store.start_session(0.0)
        state = AgentState(
            centroids=[],
            mood=0.0,
            novelty_hunger=0.95,  # almost everything engages cold-start
            salience_threshold=0.3,
        )
        post = _make_post("p1", "interesting post", embedder)
        llm = FakeLLM([
            json.dumps({"fragment": "hm", "valence": 0.3, "novelty": 0.7})
        ])

        new_state = session_step(
            state=state,
            post=post,
            store=store,
            session_id=sid,
            distortion=AGENT_VARIANTS["balanced"],
            mem_config=MemoryConfig(),
            llm=llm,
            beliefs=[],
            now=10.0,
        )
        assert new_state.posts_engaged == 1
        assert len(llm.calls) == 1
        assert len(new_state.centroids) == 1  # spawned from cold start
        traces = store.load_traces()
        assert len(traces) == 1
        assert traces[0].fragment == "hm"
        assert traces[0].valence == 0.3

    def test_engagement_decays_novelty_hunger(self, store, embedder):
        sid = store.start_session(0.0)
        state = AgentState(
            centroids=[],
            mood=0.0,
            novelty_hunger=0.95,
            salience_threshold=0.1,
        )
        distortion = AGENT_VARIANTS["balanced"]
        initial = state.novelty_hunger
        llm = FakeLLM([json.dumps(
            {"fragment": "x", "valence": 0.0, "novelty": 0.5}
        )])
        post = _make_post("p1", "post", embedder)
        new_state = session_step(state, post, store, sid, distortion,
                                 MemoryConfig(), llm, [], now=1.0)
        assert new_state.novelty_hunger < initial
        assert new_state.novelty_hunger == pytest.approx(
            initial - distortion.novelty_hunger_decay
        )

    def test_mood_drifts_toward_post_valence(self, store, embedder):
        sid = store.start_session(0.0)
        state = AgentState(centroids=[], mood=0.0, novelty_hunger=0.95,
                           salience_threshold=0.1)
        distortion = AGENT_VARIANTS["high_mood_volatility"]
        llm = FakeLLM([json.dumps(
            {"fragment": "x", "valence": 1.0, "novelty": 0.5}
        )])
        post = _make_post("p1", "post", embedder)
        new_state = session_step(state, post, store, sid, distortion,
                                 MemoryConfig(), llm, [], now=1.0)
        # high_mood_volatility = 0.4 -> new_mood = 0 + 0.4 * (1.0 - 0) = 0.4
        assert new_state.mood == pytest.approx(0.4)

    def test_mood_clamped_to_unit(self, store, embedder):
        sid = store.start_session(0.0)
        state = AgentState(centroids=[], mood=0.95, novelty_hunger=0.95,
                           salience_threshold=0.1)
        distortion = AGENT_VARIANTS["high_mood_volatility"]
        llm = FakeLLM([json.dumps(
            {"fragment": "x", "valence": 1.0, "novelty": 0.5}
        )])
        post = _make_post("p1", "post", embedder)
        new_state = session_step(state, post, store, sid, distortion,
                                 MemoryConfig(), llm, [], now=1.0)
        assert -1.0 <= new_state.mood <= 1.0


# ---------------- nightly_consolidation ----------------

class TestNightlyConsolidation:
    def test_no_traces_returns_zero(self, store, embedder):
        sid = store.start_session(0.0)
        llm = FakeLLM(["[]"])  # not even called, no traces to consolidate
        n = nightly_consolidation(
            store=store, llm=llm, embedder=embedder,
            consolidation_config=ConsolidationConfig(),
            session_id=sid, mood=0.0, now=10.0, seed=0,
        )
        assert n == 0
        assert llm.calls == []  # short-circuited

    def test_applies_belief_updates(self, store, embedder):
        sid = store.start_session(0.0)
        store.insert_trace(sid, "p1", 0.0, 0.0, 0.7, 0.3, 0.5,
                           "lingering thought", embedder.embed("p1"))
        store.insert_trace(sid, "p2", 1.0, 1.0, 0.5, -0.2, 0.6,
                           "another thought", embedder.embed("p2"))
        llm = FakeLLM([json.dumps([
            {"action": "new", "text": "discovered pattern", "new_confidence": 0.6}
        ])])
        n = nightly_consolidation(
            store=store, llm=llm, embedder=embedder,
            consolidation_config=ConsolidationConfig(),
            session_id=sid, mood=0.1, now=10.0, seed=42,
        )
        assert n == 1
        beliefs = store.load_active_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].text == "discovered pattern"

    def test_passes_fragments_to_llm(self, store, embedder):
        sid = store.start_session(0.0)
        store.insert_trace(sid, "p1", 0.0, 0.0, 0.9, 0.0, 0.5,
                           "uniquely identifiable fragment", embedder.embed("p1"))
        llm = FakeLLM(["[]"])
        nightly_consolidation(
            store=store, llm=llm, embedder=embedder,
            consolidation_config=ConsolidationConfig(),
            session_id=sid, mood=0.0, now=10.0, seed=0,
        )
        assert "uniquely identifiable fragment" in llm.calls[0]


# ---------------- session boundaries ----------------

class TestSessionBoundaries:
    def test_begin_with_no_prior_state(self, store):
        sid, state = begin_session(store, AGENT_VARIANTS["balanced"], started_at=0.0)
        assert sid > 0
        assert state.centroids == []
        assert state.novelty_hunger == AGENT_VARIANTS["balanced"].novelty_hunger_initial

    def test_begin_rehydrates_from_snapshot(self, store):
        # Seed a snapshot from a prior session
        prior_sid = store.start_session(0.0)
        from doomscroll.memory import Centroid
        store.save_interest_snapshot(
            prior_sid, ts=10.0,
            centroids=[Centroid(np.array([1.0, 0.0]), 5.0, 10.0)],
            mood=0.4, novelty_hunger=0.2,
        )
        store.end_session(prior_sid, 10.0, 5, 1, True)

        sid, state = begin_session(store, AGENT_VARIANTS["balanced"], started_at=20.0)
        assert len(state.centroids) == 1
        assert state.mood == 0.4
        # Novelty hunger RESETS at session boundary (within-session decay only)
        assert state.novelty_hunger == AGENT_VARIANTS["balanced"].novelty_hunger_initial

    def test_end_session_persists_snapshot(self, store):
        from doomscroll.memory import Centroid
        sid, state = begin_session(store, AGENT_VARIANTS["balanced"], 0.0)
        state.centroids = [Centroid(np.array([1.0, 0.0]), 3.0, 5.0)]
        state.mood = -0.2
        state.posts_processed = 100
        state.posts_engaged = 15
        end_session(store, sid, state, MemoryConfig(), ended_at=100.0,
                    consolidation_run=True)
        snap = store.load_latest_interest_snapshot()
        assert snap is not None
        cs, mood, _ = snap
        assert len(cs) == 1
        assert mood == -0.2


# ---------------- E2E with FakeLLM ----------------

class _RoutingFakeLLM:
    """Routes responses by prompt content. Robust to any engagement count."""

    def __init__(self, reaction_response: str, consolidation_response: str):
        self.reaction_response = reaction_response
        self.consolidation_response = consolidation_response
        self.reaction_calls = 0
        self.consolidation_calls = 0

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if "FRAGMENTS FROM TODAY" in prompt:
            self.consolidation_calls += 1
            return self.consolidation_response
        self.reaction_calls += 1
        return self.reaction_response


class TestEndToEnd:
    def test_full_session_runs(self, store, embedder):
        # 30 posts. Configure for fast warmup + engage-all so the test is
        # deterministic regardless of FakeEmbedder vector geometry.
        posts = [
            _make_post(f"p{i}", f"some content {i}", embedder, ts=float(i))
            for i in range(30)
        ]
        reaction = json.dumps({"fragment": "noted", "valence": 0.1, "novelty": 0.6})
        consolidation_response = json.dumps([
            {"action": "new", "text": "weekly mood: mild", "new_confidence": 0.5}
        ])
        llm = _RoutingFakeLLM(reaction, consolidation_response)

        mem_config = MemoryConfig(
            target_engagement_rate=1.0,  # engage everything once warmup hits
            salience_warmup_posts=5,     # adapt threshold quickly
        )
        final_state = run_session(
            store=store,
            posts=posts,
            distortion=AGENT_VARIANTS["balanced"],
            mem_config=mem_config,
            consolidation_config=ConsolidationConfig(),
            llm=llm,
            embedder=embedder,
            started_at=0.0,
            ended_at=3600.0,
            seed=42,
        )
        assert final_state.posts_processed == 30
        # After warmup with target_rate=1.0, threshold = min(recent), so all
        # post-warmup posts engage. Pre-warmup engagement varies.
        assert final_state.posts_engaged >= 25
        assert len(final_state.centroids) >= 1
        # Consolidation must have run and created the queued belief.
        beliefs = store.load_active_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].text == "weekly mood: mild"
        # All posts are recorded in posts_seen.
        seen, _ = store.post_seen_count(1)
        assert seen == 30

    def test_two_sessions_persist_state(self, store, embedder):
        reaction = json.dumps({"fragment": "hm", "valence": 0.2, "novelty": 0.6})
        llm = _RoutingFakeLLM(reaction, "[]")

        mem_config = MemoryConfig(target_engagement_rate=1.0, salience_warmup_posts=3)
        cfg = ConsolidationConfig()

        # Session 1
        posts1 = [_make_post(f"p{i}", f"text {i}", embedder, ts=float(i))
                  for i in range(10)]
        run_session(
            store=store, posts=posts1, distortion=AGENT_VARIANTS["balanced"],
            mem_config=mem_config, consolidation_config=cfg,
            llm=llm, embedder=embedder, started_at=0.0, ended_at=100.0,
        )
        snap_after_1 = store.load_latest_interest_snapshot()
        assert snap_after_1 is not None
        centroids_1 = snap_after_1[0]
        assert len(centroids_1) >= 1

        # Session 2 should rehydrate centroids from snapshot
        posts2 = [_make_post(f"p{i+10}", f"more text {i}", embedder,
                             ts=float(i + 100)) for i in range(10)]
        run_session(
            store=store, posts=posts2, distortion=AGENT_VARIANTS["balanced"],
            mem_config=mem_config, consolidation_config=cfg,
            llm=llm, embedder=embedder, started_at=200.0, ended_at=300.0,
        )
        assert store.session_count() == 2
        snap_after_2 = store.load_latest_interest_snapshot()
        assert snap_after_2 is not None
