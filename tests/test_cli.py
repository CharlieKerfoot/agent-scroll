"""Tests for the pilot CLI runner.

Invokes main() in fake mode against a tmp output dir. Validates that the
end-to-end pipeline produces a results.json with the expected structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doomscroll.cli import _RoutingFakeLLM, _resolve_variants, main
from doomscroll.config import AGENT_VARIANTS


class TestResolveVariants:
    def test_all_default(self):
        result = _resolve_variants(",".join(AGENT_VARIANTS.keys()))
        assert set(result.keys()) == set(AGENT_VARIANTS.keys())

    def test_subset(self):
        result = _resolve_variants("balanced,high_confirmation_bias")
        assert set(result.keys()) == {"balanced", "high_confirmation_bias"}

    def test_unknown_variant_exits(self):
        with pytest.raises(SystemExit, match="unknown variant"):
            _resolve_variants("balanced,not_a_real_variant")

    def test_empty_exits(self):
        with pytest.raises(SystemExit, match="no variants"):
            _resolve_variants(",,")


class TestRoutingFakeLLM:
    def test_consolidation_returns_empty_array(self):
        llm = _RoutingFakeLLM()
        out = llm.generate("... FRAGMENTS FROM TODAY ... whatever")
        assert json.loads(out) == []

    def test_reaction_returns_object(self):
        llm = _RoutingFakeLLM()
        out = llm.generate("... POST: hello ...")
        data = json.loads(out)
        assert "fragment" in data
        assert "valence" in data
        assert "novelty" in data


class TestPilotRun:
    def test_minimal_pilot_runs(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        rc = main([
            "--provider", "fake",
            "--embedder", "fake",
            "--output-dir", str(out_dir),
            "--variants", "balanced,high_confirmation_bias",
            "--sessions", "2",
            "--posts-per-session", "10",
            "--seed", "0",
        ])
        assert rc == 0
        # Per-agent DBs created
        assert (out_dir / "agent_balanced.db").exists()
        assert (out_dir / "agent_high_confirmation_bias.db").exists()
        # results.json structure
        results_path = out_dir / "results.json"
        assert results_path.exists()
        with open(results_path) as f:
            results = json.load(f)
        assert results["metadata"]["provider"] == "fake"
        assert results["metadata"]["sessions"] == 2
        assert len(results["per_agent"]) == 2
        for agent_summary in results["per_agent"]:
            assert agent_summary["sessions_run"] == 2
            assert agent_summary["posts_processed"] == 20
            # Fake LLM returns empty consolidation -> no beliefs accumulate
            assert agent_summary["belief_count"] == 0

    def test_summary_printed(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        main([
            "--provider", "fake",
            "--embedder", "fake",
            "--output-dir", str(out_dir),
            "--variants", "balanced",
            "--sessions", "1",
            "--posts-per-session", "5",
        ])
        captured = capsys.readouterr().out
        assert "PILOT RESULTS" in captured
        assert "balanced" in captured

    def test_idempotent_reruns(self, tmp_path):
        out_dir = tmp_path / "out"
        argv = [
            "--provider", "fake", "--embedder", "fake",
            "--output-dir", str(out_dir),
            "--variants", "balanced",
            "--sessions", "1", "--posts-per-session", "5",
            "--seed", "1",
        ]
        main(argv)
        first = json.load(open(out_dir / "results.json"))
        main(argv)  # rerun should clobber, not accumulate
        second = json.load(open(out_dir / "results.json"))
        assert (
            first["per_agent"][0]["posts_processed"]
            == second["per_agent"][0]["posts_processed"]
        )

    def test_missing_model_for_real_provider_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="--model is required"):
            main([
                "--provider", "anthropic",
                "--output-dir", str(tmp_path / "out"),
                "--sessions", "1", "--posts-per-session", "1",
            ])

    def test_missing_snapshot_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="snapshot not found"):
            main([
                "--provider", "fake", "--embedder", "fake",
                "--output-dir", str(tmp_path / "out"),
                "--snapshot", str(tmp_path / "nope.json"),
            ])
