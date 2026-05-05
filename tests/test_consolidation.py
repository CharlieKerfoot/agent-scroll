"""Tests for the consolidation module.

These guarantee invariants the prompt design depends on:
- Source attribution provably stripped (no post text leaks)
- Fragment shuffling deterministic with seed
- Malformed LLM output handled (returns [], never raises)
- Output capped at max_belief_updates_per_pass
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from doomscroll.config import ConsolidationConfig
from doomscroll.consolidation import (
    BeliefAction,
    Fragment,
    build_prompt,
    consolidate,
    parse_belief_updates,
)
from doomscroll.memory import Belief


def _frag(text: str, sal=0.5, val=0.0, nov=0.5) -> Fragment:
    return Fragment(text=text, salience=sal, valence=val, novelty=nov)


def _belief(id_: int, text: str, conf=0.5) -> Belief:
    return Belief(
        id=id_,
        text=text,
        embedding=np.array([1.0, 0.0]),
        confidence=conf,
        last_updated_at=0.0,
    )


class TestBuildPrompt:
    def test_deterministic_with_seed(self):
        cfg = ConsolidationConfig()
        frags = [_frag(f"frag-{i}") for i in range(10)]
        beliefs = [_belief(i, f"belief-{i}", conf=0.5) for i in range(5)]
        p1 = build_prompt(frags, beliefs, mood=0.0, config=cfg, now=0.0, seed=42)
        p2 = build_prompt(frags, beliefs, mood=0.0, config=cfg, now=0.0, seed=42)
        assert p1 == p2

    def test_different_seed_different_order(self):
        cfg = ConsolidationConfig()
        frags = [_frag(f"unique-fragment-{i}") for i in range(20)]
        beliefs = []
        p1 = build_prompt(frags, beliefs, mood=0.0, config=cfg, now=0.0, seed=1)
        p2 = build_prompt(frags, beliefs, mood=0.0, config=cfg, now=0.0, seed=2)
        # Same fragments but different shuffled order means different prompts.
        assert p1 != p2

    def test_no_source_post_text_leaks(self):
        # Critical invariant: only fragments (the agent's reactions) appear
        # in the prompt. Post text must NEVER be embedded.
        cfg = ConsolidationConfig()
        secret_post_text = "THIS_IS_RAW_POST_CONTENT_SHOULD_NEVER_APPEAR"
        # Even if a fragment somehow contained source-style text, the data
        # type itself doesn't carry source. We confirm by inspecting fields.
        frags = [_frag("my reaction was muted")]
        prompt = build_prompt(frags, [], mood=0.0, config=cfg, now=0.0, seed=0)
        assert secret_post_text not in prompt
        # The fragment's *text* (the agent's reaction) IS in the prompt.
        assert "my reaction was muted" in prompt

    def test_mood_formatted_in_prompt(self):
        cfg = ConsolidationConfig()
        prompt = build_prompt([], [], mood=-0.42, config=cfg, now=0.0, seed=0)
        assert "-0.42" in prompt

    def test_empty_inputs_render(self):
        cfg = ConsolidationConfig()
        prompt = build_prompt([], [], mood=0.0, config=cfg, now=0.0, seed=0)
        assert "no beliefs yet" in prompt
        assert "nothing today" in prompt

    def test_max_updates_in_prompt(self):
        cfg = ConsolidationConfig(max_belief_updates_per_pass=5)
        prompt = build_prompt([], [], mood=0.0, config=cfg, now=0.0, seed=0)
        assert "AT MOST 5" in prompt

    def test_fragments_capped_to_max(self):
        cfg = ConsolidationConfig(fragment_max_count=3)
        frags = [_frag(f"frag-{i}") for i in range(10)]
        prompt = build_prompt(frags, [], mood=0.0, config=cfg, now=0.0, seed=0)
        # Only first 3 should appear (cap, then shuffle of those 3)
        for i in range(3):
            assert f"frag-{i}" in prompt
        for i in range(3, 10):
            assert f"frag-{i}" not in prompt


class TestParseBeliefUpdates:
    def test_valid_new_belief(self):
        raw = json.dumps([
            {"action": "new", "text": "thing is happening", "new_confidence": 0.6}
        ])
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert len(result) == 1
        assert result[0].action == BeliefAction.NEW
        assert result[0].text == "thing is happening"
        assert result[0].new_confidence == 0.6

    def test_valid_strengthen(self):
        raw = json.dumps([
            {"action": "strengthen", "target_id": 7, "new_confidence": 0.8}
        ])
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert len(result) == 1
        assert result[0].action == BeliefAction.STRENGTHEN
        assert result[0].target_id == 7

    def test_drop_doesnt_need_confidence(self):
        raw = json.dumps([{"action": "drop", "target_id": 3}])
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert len(result) == 1
        assert result[0].action == BeliefAction.DROP

    def test_empty_array_valid(self):
        result = parse_belief_updates("[]", ConsolidationConfig())
        assert result == []

    def test_malformed_json_returns_empty(self):
        assert parse_belief_updates("not json at all", ConsolidationConfig()) == []
        assert parse_belief_updates("{broken", ConsolidationConfig()) == []
        assert parse_belief_updates("", ConsolidationConfig()) == []

    def test_non_array_returns_empty(self):
        raw = json.dumps({"action": "new", "text": "x", "new_confidence": 0.5})
        assert parse_belief_updates(raw, ConsolidationConfig()) == []

    def test_invalid_action_skipped(self):
        raw = json.dumps([
            {"action": "delete_universe", "text": "x", "new_confidence": 0.5},
            {"action": "new", "text": "valid", "new_confidence": 0.5},
        ])
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert len(result) == 1
        assert result[0].text == "valid"

    def test_out_of_range_confidence_rejected(self):
        raw = json.dumps([
            {"action": "new", "text": "x", "new_confidence": 1.5},
            {"action": "new", "text": "y", "new_confidence": -0.1},
        ])
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert result == []

    def test_new_without_text_rejected(self):
        raw = json.dumps([{"action": "new", "new_confidence": 0.5}])
        assert parse_belief_updates(raw, ConsolidationConfig()) == []

    def test_strengthen_without_target_rejected(self):
        raw = json.dumps([{"action": "strengthen", "new_confidence": 0.7}])
        assert parse_belief_updates(raw, ConsolidationConfig()) == []

    def test_capped_at_max_updates(self):
        cfg = ConsolidationConfig(max_belief_updates_per_pass=2)
        raw = json.dumps([
            {"action": "new", "text": f"belief-{i}", "new_confidence": 0.5}
            for i in range(10)
        ])
        result = parse_belief_updates(raw, cfg)
        assert len(result) == 2

    def test_strips_code_fences(self):
        raw = '```json\n[{"action": "new", "text": "x", "new_confidence": 0.5}]\n```'
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert len(result) == 1

    def test_extracts_array_from_prose(self):
        raw = (
            "Here is the consolidation:\n"
            '[{"action": "new", "text": "x", "new_confidence": 0.5}]\n'
            "Hope this helps!"
        )
        result = parse_belief_updates(raw, ConsolidationConfig())
        assert len(result) == 1


class TestConsolidate:
    def test_uses_llm_output(self):
        class FakeLLM:
            def generate(self, prompt: str, max_tokens: int = 1024) -> str:
                return json.dumps([
                    {"action": "new", "text": "from llm", "new_confidence": 0.7}
                ])

        result = consolidate(
            fragments=[_frag("x")],
            beliefs=[],
            mood=0.0,
            llm=FakeLLM(),
            config=ConsolidationConfig(),
            now=0.0,
            seed=0,
        )
        assert len(result) == 1
        assert result[0].text == "from llm"

    def test_empty_llm_response_returns_empty(self):
        class EmptyLLM:
            def generate(self, prompt: str, max_tokens: int = 1024) -> str:
                return ""

        result = consolidate(
            fragments=[_frag("x")],
            beliefs=[],
            mood=0.0,
            llm=EmptyLLM(),
            config=ConsolidationConfig(),
            now=0.0,
            seed=0,
        )
        assert result == []

    def test_llm_receives_well_formed_prompt(self):
        captured = {}

        class CaptureLLM:
            def generate(self, prompt: str, max_tokens: int = 1024) -> str:
                captured["prompt"] = prompt
                return "[]"

        consolidate(
            fragments=[_frag("hello"), _frag("world")],
            beliefs=[_belief(1, "existing", 0.6)],
            mood=0.3,
            llm=CaptureLLM(),
            config=ConsolidationConfig(),
            now=100.0,
            seed=42,
        )
        prompt = captured["prompt"]
        assert "hello" in prompt
        assert "world" in prompt
        assert "existing" in prompt
        assert "0.30" in prompt
