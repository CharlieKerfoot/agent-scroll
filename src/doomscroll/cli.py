"""Command-line pilot runner.

Configuration source of truth is YAML, validated by Pydantic. The CLI is
a thin layer:

    1. Load --config <path> (or built-in defaults if omitted)
    2. Apply --set key.path=value overrides (typed via YAML scalar parsing)
    3. Snapshot the resolved config to <output_dir>/config.yaml
    4. Run sessions, write per-agent DBs + results.json

Usage examples:

    # Smoke test, all defaults (fake LLM, synthetic feed)
    uv run doomscroll-pilot

    # Real experiment from a versioned config file
    uv run doomscroll-pilot --config configs/experiments/anthropic_pilot.yaml

    # Quick override without editing YAML
    uv run doomscroll-pilot \\
        --config configs/default.yaml \\
        --set pilot.sessions=5 \\
        --set llm.provider=anthropic \\
        --set llm.model=claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv

from doomscroll.agent_loop import run_session
from doomscroll.config import DistortionConfig
from doomscroll.embedding import (
    Embedder,
    EmbeddingConfig,
    FakeEmbedder,
    make_embedder,
)
from doomscroll.feed import (
    JSONSnapshotFeed,
    SyntheticFeed,
)
from doomscroll.llm import LLMConfig, make_llm
from doomscroll.measurement import (
    cross_agent_divergence,
    topic_entropy,
)
from doomscroll.persistence import Store
from doomscroll.run_config import RunConfig


SESSION_DURATION_SECONDS = 30 * 60  # 30-min compressed session
DAY_SECONDS = 24 * 3600


class _RejectionCounter(logging.Handler):
    """Counts WARNING records by their logger module. Used for the end-of-pilot
    summary so silent drops aren't silent."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.counts: dict[str, int] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            self.counts[record.name] = self.counts.get(record.name, 0) + 1


def _configure_logging(output_dir: Path) -> _RejectionCounter:
    """File log captures everything; stderr captures warnings+; counter
    handler tracks warning counts for the final summary."""
    log_path = output_dir / "pilot.log"
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    file_h = logging.FileHandler(log_path, mode="w")
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(logging.Formatter(fmt))
    stderr_h = logging.StreamHandler(sys.stderr)
    stderr_h.setLevel(logging.WARNING)
    stderr_h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
    counter = _RejectionCounter()
    root = logging.getLogger()
    # Reset to avoid duplicate handlers across repeated main() invocations
    # (tests, --print-config dry runs, etc.)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.INFO)
    root.addHandler(file_h)
    root.addHandler(stderr_h)
    root.addHandler(counter)
    return counter


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doomscroll-pilot",
        description=(
            "Run a doomscroll cognitive simulation pilot. "
            "All parameters live in a YAML config; pass --config to point at "
            "one, or use --set for ad-hoc overrides."
        ),
    )
    p.add_argument(
        "--config",
        type=Path,
        help="Path to a pilot YAML config. Omit to use built-in defaults.",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a config field, e.g. --set pilot.sessions=5. "
            "Dotted paths target nested fields. Repeatable."
        ),
    )
    p.add_argument(
        "--print-config",
        action="store_true",
        help="Resolve config, print it, and exit without running.",
    )
    return p


def _parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse `--set a.b=c` into a nested dict. Values go through YAML scalar
    parsing so `5`, `true`, `null`, `[a, b]` all DTRT."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise SystemExit(f"--set has empty key: {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as e:
            raise SystemExit(f"--set {item!r}: invalid value ({e})") from e
        cursor = out
        parts = key.split(".")
        for p in parts[:-1]:
            cursor = cursor.setdefault(p, {})
            if not isinstance(cursor, dict):
                raise SystemExit(
                    f"--set {item!r}: {p} conflicts with non-mapping override"
                )
        cursor[parts[-1]] = value
    return out


def _build_llm(cfg: RunConfig):
    if cfg.llm.provider == "fake":
        return _RoutingFakeLLM()
    llm_cfg = LLMConfig(
        model=cfg.llm.model or "",
        temperature=cfg.llm.temperature,
        max_tokens=cfg.llm.max_tokens,
    )
    return make_llm(cfg.llm.provider, llm_cfg)


def _build_embedder(cfg: RunConfig) -> Embedder:
    if cfg.embedder.provider == "fake":
        return FakeEmbedder(dim=64)
    return make_embedder(
        "openai", EmbeddingConfig(model=cfg.embedder.model)
    )


class _RoutingFakeLLM:
    """Used when llm.provider == fake. Returns canned reactions and empty
    consolidation. Lets the pipeline run end-to-end without API access."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if "FRAGMENTS FROM TODAY" in prompt:
            return "[]"
        return json.dumps(
            {"fragment": "noted", "valence": 0.0, "novelty": 0.5}
        )


def _load_or_make_feed(cfg: RunConfig, embedder: Embedder):
    if cfg.feed.snapshot:
        path = Path(cfg.feed.snapshot)
        if not path.exists():
            raise SystemExit(f"snapshot not found: {path}")
        feed = JSONSnapshotFeed(path)
        print(f"[feed] loaded snapshot: {path} ({len(feed)} posts)")
        return feed
    n = cfg.pilot.sessions * cfg.pilot.posts_per_session
    print(f"[feed] no snapshot configured, generating synthetic ({n} posts)")
    return SyntheticFeed(embedder=embedder, n_posts=n, seed=cfg.pilot.seed)


def _run_one_agent(
    variant_name: str,
    distortion: DistortionConfig,
    posts: list,
    cfg: RunConfig,
    llm,
    embedder: Embedder,
    db_path: Path,
) -> dict:
    """Run one agent end-to-end. Returns per-agent summary dict."""
    print(f"[agent {variant_name}] starting on {db_path.name}")
    mem_config = cfg.memory.to_dataclass()
    cons_config = cfg.consolidation.to_dataclass()
    posts_per = cfg.pilot.posts_per_session
    base_ts = 0.0
    final_state = None
    sessions_run = 0
    total_processed = 0
    total_engaged = 0
    with Store(db_path) as store:
        for s in range(cfg.pilot.sessions):
            session_start = base_ts + s * DAY_SECONDS
            session_end = session_start + SESSION_DURATION_SECONDS
            session_posts = posts[s * posts_per : (s + 1) * posts_per]
            if not session_posts:
                break
            final_state = run_session(
                store=store,
                posts=session_posts,
                distortion=distortion,
                mem_config=mem_config,
                consolidation_config=cons_config,
                llm=llm,
                embedder=embedder,
                started_at=session_start,
                ended_at=session_end,
                consolidate_at_end=cfg.pilot.consolidate,
                seed=cfg.pilot.seed,
            )
            sessions_run += 1
            total_processed += final_state.posts_processed
            total_engaged += final_state.posts_engaged
        beliefs = store.load_active_beliefs()

    summary = {
        "variant": variant_name,
        "db_path": str(db_path),
        "sessions_run": sessions_run,
        "posts_processed": total_processed,
        "posts_engaged": total_engaged,
        "centroid_count": len(final_state.centroids) if final_state else 0,
        "final_mood": float(final_state.mood) if final_state else 0.0,
        "final_novelty_hunger": (
            float(final_state.novelty_hunger) if final_state else 0.0
        ),
        "belief_count": len(beliefs),
        "beliefs": [
            {"id": b.id, "text": b.text, "confidence": b.confidence}
            for b in beliefs
        ],
        "_belief_embeddings": [b.embedding.tolist() for b in beliefs],
    }
    print(
        f"[agent {variant_name}] done: "
        f"engaged {summary['posts_engaged']}/{summary['posts_processed']}, "
        f"{summary['belief_count']} beliefs, mood={summary['final_mood']:+.2f}"
    )
    return summary


def _compute_cross_agent_metrics(per_agent: list[dict]) -> dict:
    belief_sets: dict[str, np.ndarray] = {}
    for a in per_agent:
        embs = a["_belief_embeddings"]
        if embs:
            belief_sets[a["variant"]] = np.asarray(embs, dtype=np.float64)
    if len(belief_sets) < 2:
        divergence = {}
    else:
        raw = cross_agent_divergence(belief_sets)
        divergence = {
            f"{a}__{b}": float(d) for (a, b), d in raw.items() if a < b
        }

    entropy_k = 3
    topic_entropies = {
        name: float(topic_entropy(emb, k=entropy_k))
        for name, emb in belief_sets.items()
    }
    return {
        "cross_agent_divergence": divergence,
        "topic_entropy_k": entropy_k,
        "topic_entropy": topic_entropies,
    }


def _print_results_summary(per_agent: list[dict], cross: dict) -> None:
    print()
    print("=" * 60)
    print("PILOT RESULTS")
    print("=" * 60)
    print(f"{'agent':<28} {'engaged':>10} {'beliefs':>10} {'mood':>8}")
    print("-" * 60)
    for a in per_agent:
        ratio = (
            a["posts_engaged"] / a["posts_processed"]
            if a["posts_processed"]
            else 0
        )
        print(
            f"{a['variant']:<28} "
            f"{a['posts_engaged']:>4}/{a['posts_processed']:<4} ({ratio:>4.0%}) "
            f"{a['belief_count']:>10} "
            f"{a['final_mood']:>+7.2f}"
        )
    if cross["cross_agent_divergence"]:
        print()
        print("CROSS-AGENT BELIEF DIVERGENCE (cosine distance):")
        for pair, d in sorted(cross["cross_agent_divergence"].items()):
            print(f"  {pair:<50} {d:>6.3f}")
    if cross["topic_entropy"]:
        print()
        print(f"TOPIC ENTROPY per agent (k={cross['topic_entropy_k']}):")
        for name, e in sorted(cross["topic_entropy"].items()):
            print(f"  {name:<28} {e:>6.3f}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    # Load .env from CWD if present so API keys reach the SDKs without
    # the user having to `source` or use `uv run --env-file`. Existing
    # environment variables win (override=False).
    load_dotenv(override=False)

    args = _build_arg_parser().parse_args(argv)

    overrides = _parse_overrides(args.set)
    try:
        cfg = RunConfig.load(args.config, overrides)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from e
    except Exception as e:
        # pydantic ValidationError stringifies usefully
        raise SystemExit(f"invalid config: {e}") from e

    if args.print_config:
        print(cfg.to_yaml())
        return 0

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.write_to(cfg.output_dir / "config.yaml")
    rejection_counter = _configure_logging(cfg.output_dir)

    embedder = _build_embedder(cfg)
    llm = _build_llm(cfg)
    feed = _load_or_make_feed(cfg, embedder)
    posts = list(feed.iter_posts())
    needed = cfg.pilot.sessions * cfg.pilot.posts_per_session
    if len(posts) < needed:
        print(
            f"[warn] feed has {len(posts)} posts but pilot wants {needed}; "
            "later sessions will be short."
        )

    variants = cfg.variant_distortions()
    started_at = time.time()
    per_agent: list[dict] = []
    for name, distortion in variants.items():
        db_path = cfg.output_dir / f"agent_{name}.db"
        if db_path.exists():
            db_path.unlink()
        summary = _run_one_agent(
            name, distortion, posts, cfg, llm, embedder, db_path
        )
        per_agent.append(summary)

    cross = _compute_cross_agent_metrics(per_agent)
    elapsed = time.time() - started_at

    results = {
        "metadata": {
            "started_at": started_at,
            "elapsed_seconds": elapsed,
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "embedder": cfg.embedder.provider,
            "sessions": cfg.pilot.sessions,
            "posts_per_session": cfg.pilot.posts_per_session,
            "seed": cfg.pilot.seed,
        },
        "per_agent": [
            {k: v for k, v in a.items() if not k.startswith("_")}
            for a in per_agent
        ],
        "cross_agent": cross,
    }
    results_path = cfg.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_results_summary(per_agent, cross)
    print(f"\nresults written to: {results_path}")
    print(f"resolved config:    {cfg.output_dir / 'config.yaml'}")
    print(f"elapsed: {elapsed:.1f}s")

    if rejection_counter.counts:
        print()
        print("WARNINGS (silent drops -- check pilot.log for details):")
        for module, n in sorted(rejection_counter.counts.items()):
            print(f"  {module:<32} {n}")
        print(f"  full log:                        {cfg.output_dir / 'pilot.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
