"""Tests for the pilot CLI runner.

The CLI is a thin wrapper around RunConfig + the agent loop. Tests drive
it through `--set` overrides against a tmp output dir in fake mode.
"""

from __future__ import annotations

import json

import pytest
import yaml

from doomscroll.cli import _RoutingFakeLLM, _parse_overrides, main
from doomscroll.run_config import RunConfig


class TestParseOverrides:
    def test_nested_dotted_path(self):
        out = _parse_overrides(["pilot.sessions=5", "llm.provider=anthropic"])
        assert out == {
            "pilot": {"sessions": 5},
            "llm": {"provider": "anthropic"},
        }

    def test_yaml_typed_values(self):
        out = _parse_overrides(
            ["pilot.consolidate=false", "feed.snapshot=null", "llm.temperature=0.3"]
        )
        assert out["pilot"]["consolidate"] is False
        assert out["feed"]["snapshot"] is None
        assert out["llm"]["temperature"] == 0.3

    def test_list_value(self):
        out = _parse_overrides(["pilot.variants=[balanced, high_novelty_hunger]"])
        assert out["pilot"]["variants"] == ["balanced", "high_novelty_hunger"]

    def test_missing_equals_exits(self):
        with pytest.raises(SystemExit, match="KEY=VALUE"):
            _parse_overrides(["pilot.sessions"])


class TestRunConfigLoad:
    def test_defaults(self):
        cfg = RunConfig.load()
        assert cfg.llm.provider == "fake"
        assert cfg.pilot.sessions == 3
        assert "balanced" in cfg.pilot.variants

    def test_unknown_variant_rejected(self):
        with pytest.raises(Exception, match="unknown variants"):
            RunConfig.load(overrides={"pilot": {"variants": ["not_a_real"]}})

    def test_real_provider_requires_model(self):
        with pytest.raises(Exception, match="llm.model is required"):
            RunConfig.load(overrides={"llm": {"provider": "anthropic"}})

    def test_yaml_round_trip(self, tmp_path):
        cfg = RunConfig.load(
            overrides={"pilot": {"sessions": 7}, "llm": {"temperature": 0.1}}
        )
        out = tmp_path / "cfg.yaml"
        cfg.write_to(out)
        reloaded = RunConfig.load(out)
        assert reloaded.pilot.sessions == 7
        assert reloaded.llm.temperature == 0.1


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
    def test_minimal_pilot_runs(self, tmp_path):
        out_dir = tmp_path / "out"
        rc = main([
            "--set", f"output_dir={out_dir}",
            "--set", "pilot.variants=[balanced, high_confirmation_bias]",
            "--set", "pilot.sessions=2",
            "--set", "pilot.posts_per_session=10",
            "--set", "pilot.seed=0",
        ])
        assert rc == 0
        assert (out_dir / "agent_balanced.db").exists()
        assert (out_dir / "agent_high_confirmation_bias.db").exists()
        assert (out_dir / "config.yaml").exists()

        results = json.loads((out_dir / "results.json").read_text())
        assert results["metadata"]["provider"] == "fake"
        assert results["metadata"]["sessions"] == 2
        assert len(results["per_agent"]) == 2
        for agent_summary in results["per_agent"]:
            assert agent_summary["sessions_run"] == 2
            assert agent_summary["posts_processed"] == 20
            assert agent_summary["belief_count"] == 0

    def test_resolved_config_dumped(self, tmp_path):
        out_dir = tmp_path / "out"
        main([
            "--set", f"output_dir={out_dir}",
            "--set", "pilot.variants=[balanced]",
            "--set", "pilot.sessions=1",
            "--set", "pilot.posts_per_session=5",
        ])
        dumped = yaml.safe_load((out_dir / "config.yaml").read_text())
        assert dumped["pilot"]["sessions"] == 1
        assert dumped["pilot"]["variants"] == ["balanced"]

    def test_summary_printed(self, tmp_path, capsys):
        out_dir = tmp_path / "out"
        main([
            "--set", f"output_dir={out_dir}",
            "--set", "pilot.variants=[balanced]",
            "--set", "pilot.sessions=1",
            "--set", "pilot.posts_per_session=5",
        ])
        captured = capsys.readouterr().out
        assert "PILOT RESULTS" in captured
        assert "balanced" in captured

    def test_idempotent_reruns(self, tmp_path):
        out_dir = tmp_path / "out"
        argv = [
            "--set", f"output_dir={out_dir}",
            "--set", "pilot.variants=[balanced]",
            "--set", "pilot.sessions=1",
            "--set", "pilot.posts_per_session=5",
            "--set", "pilot.seed=1",
        ]
        main(argv)
        first = json.loads((out_dir / "results.json").read_text())
        main(argv)
        second = json.loads((out_dir / "results.json").read_text())
        assert (
            first["per_agent"][0]["posts_processed"]
            == second["per_agent"][0]["posts_processed"]
        )

    def test_missing_model_for_real_provider_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="llm.model is required"):
            main([
                "--set", f"output_dir={tmp_path / 'out'}",
                "--set", "llm.provider=anthropic",
            ])

    def test_missing_snapshot_exits(self, tmp_path):
        with pytest.raises(SystemExit, match="snapshot not found"):
            main([
                "--set", f"output_dir={tmp_path / 'out'}",
                "--set", f"feed.snapshot={tmp_path / 'nope.json'}",
            ])

    def test_print_config_exits_without_running(self, tmp_path, capsys):
        rc = main([
            "--set", f"output_dir={tmp_path / 'out'}",
            "--print-config",
        ])
        assert rc == 0
        captured = capsys.readouterr().out
        assert "pilot:" in captured
        # No run actually happened
        assert not (tmp_path / "out").exists()

    def test_loads_yaml_file(self, tmp_path):
        cfg_path = tmp_path / "run.yaml"
        cfg_path.write_text(yaml.safe_dump({
            "output_dir": str(tmp_path / "out"),
            "pilot": {
                "variants": ["balanced"],
                "sessions": 1,
                "posts_per_session": 5,
            },
        }))
        rc = main(["--config", str(cfg_path)])
        assert rc == 0
        assert (tmp_path / "out" / "results.json").exists()
