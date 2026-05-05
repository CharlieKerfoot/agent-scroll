"""Command-line pilot runner.

Single-command entry point that ties the whole architecture together:
    1. Loads or generates a snapshot
    2. For each requested agent variant, runs N sessions on a per-agent DB
    3. Computes measurement comparisons (drift, entropy, divergence)
    4. Writes results to disk and prints summary

Usage examples:
    # Smoke test with everything fake (no API keys needed)
    uv run doomscroll-pilot --provider fake --output-dir ./out

    # Real run with Claude + OpenAI embeddings
    uv run doomscroll-pilot \\
        --provider anthropic --model claude-sonnet-4-6 \\
        --embedder openai --embedder-model text-embedding-3-small \\
        --sessions 3 --posts-per-session 50 \\
        --output-dir ./pilot-2026-05

The output directory contains: agent_<variant>.db per agent, plus a
results.json with measurement metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from doomscroll.agent_loop import run_session
from doomscroll.config import (
    AGENT_VARIANTS,
    ConsolidationConfig,
    DistortionConfig,
    MemoryConfig,
)
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
    belief_drift,
    cross_agent_divergence,
    topic_entropy,
)
from doomscroll.persistence import Store


SESSION_DURATION_SECONDS = 30 * 60  # 30-min compressed session
DAY_SECONDS = 24 * 3600


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="doomscroll-pilot",
        description="Run a doomscroll cognitive simulation pilot.",
    )
    p.add_argument(
        "--provider",
        choices=["anthropic", "openai", "openrouter", "fake"],
        default="fake",
        help="LLM provider for reactions + consolidation (default: fake)",
    )
    p.add_argument(
        "--model",
        help="LLM model name (required for non-fake providers)",
    )
    p.add_argument(
        "--embedder",
        choices=["openai", "fake"],
        default="fake",
        help="Embedding provider (default: fake)",
    )
    p.add_argument(
        "--embedder-model",
        default="text-embedding-3-small",
        help="Embedding model (only used when --embedder openai)",
    )
    p.add_argument(
        "--snapshot",
        type=Path,
        help="Path to a prepared snapshot JSON. If omitted, synthetic feed is used.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./pilot-output"),
        help="Directory for per-agent DBs and results.json",
    )
    p.add_argument(
        "--variants",
        default=",".join(AGENT_VARIANTS.keys()),
        help="Comma-separated agent variant names (default: all 4)",
    )
    p.add_argument(
        "--sessions",
        type=int,
        default=3,
        help="Number of sessions per agent (default: 3)",
    )
    p.add_argument(
        "--posts-per-session",
        type=int,
        default=50,
        help="Posts to scroll per session (default: 50)",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    p.add_argument(
        "--no-consolidation",
        action="store_true",
        help="Skip the nightly consolidation pass (debug)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.7, help="LLM temperature"
    )
    return p


def _resolve_variants(spec: str) -> dict[str, DistortionConfig]:
    names = [n.strip() for n in spec.split(",") if n.strip()]
    out = {}
    for name in names:
        if name not in AGENT_VARIANTS:
            valid = ", ".join(sorted(AGENT_VARIANTS))
            raise SystemExit(f"unknown variant {name!r}; valid: {valid}")
        out[name] = AGENT_VARIANTS[name]
    if not out:
        raise SystemExit("no variants selected")
    return out


def _build_llm(args: argparse.Namespace):
    if args.provider == "fake":
        from doomscroll.llm import FakeLLM
        # Pre-seed fake with a generic reaction + empty consolidation,
        # repeated indefinitely via routing
        return _RoutingFakeLLM()
    if not args.model:
        raise SystemExit(f"--model is required for provider={args.provider}")
    cfg = LLMConfig(model=args.model, temperature=args.temperature)
    return make_llm(args.provider, cfg)


def _build_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "fake":
        return FakeEmbedder(dim=64)
    return make_embedder("openai", EmbeddingConfig(model=args.embedder_model))


class _RoutingFakeLLM:
    """Used in --provider fake mode. Returns canned reactions and empty
    consolidation. Lets the pipeline run end-to-end without API access."""

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        if "FRAGMENTS FROM TODAY" in prompt:
            return "[]"
        return json.dumps(
            {"fragment": "noted", "valence": 0.0, "novelty": 0.5}
        )


def _load_or_make_feed(args: argparse.Namespace, embedder: Embedder):
    if args.snapshot:
        if not args.snapshot.exists():
            raise SystemExit(f"snapshot not found: {args.snapshot}")
        feed = JSONSnapshotFeed(args.snapshot)
        print(f"[feed] loaded snapshot: {args.snapshot} ({len(feed)} posts)")
        return feed
    n = args.sessions * args.posts_per_session
    print(f"[feed] no snapshot provided, generating synthetic ({n} posts)")
    return SyntheticFeed(
        embedder=embedder, n_posts=n, seed=args.seed,
    )


def _run_one_agent(
    variant_name: str,
    distortion: DistortionConfig,
    posts: list,
    args: argparse.Namespace,
    llm,
    embedder: Embedder,
    db_path: Path,
) -> dict:
    """Run one agent end-to-end. Returns per-agent summary dict."""
    print(f"[agent {variant_name}] starting on {db_path.name}")
    mem_config = MemoryConfig()
    cfg = ConsolidationConfig()
    posts_per = args.posts_per_session
    base_ts = 0.0
    final_state = None
    sessions_run = 0
    total_processed = 0
    total_engaged = 0
    with Store(db_path) as store:
        for s in range(args.sessions):
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
                consolidation_config=cfg,
                llm=llm,
                embedder=embedder,
                started_at=session_start,
                ended_at=session_end,
                consolidate_at_end=not args.no_consolidation,
                seed=args.seed,
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
    """Compute cross-agent divergence + per-agent topic entropy."""
    belief_sets: dict[str, np.ndarray] = {}
    for a in per_agent:
        embs = a["_belief_embeddings"]
        if embs:
            belief_sets[a["variant"]] = np.asarray(embs, dtype=np.float64)
    if len(belief_sets) < 2:
        divergence = {}
    else:
        raw = cross_agent_divergence(belief_sets)
        # Serialize tuple keys to "a__b" strings for JSON
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
    args = _build_arg_parser().parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    embedder = _build_embedder(args)
    llm = _build_llm(args)
    feed = _load_or_make_feed(args, embedder)
    posts = list(feed.iter_posts())
    needed = args.sessions * args.posts_per_session
    if len(posts) < needed:
        print(
            f"[warn] feed has {len(posts)} posts but pilot wants {needed}; "
            "later sessions will be short."
        )

    variants = _resolve_variants(args.variants)
    started_at = time.time()
    per_agent: list[dict] = []
    for name, distortion in variants.items():
        db_path = args.output_dir / f"agent_{name}.db"
        # Fresh runs only -- don't accumulate across pilot invocations
        if db_path.exists():
            db_path.unlink()
        summary = _run_one_agent(
            name, distortion, posts, args, llm, embedder, db_path
        )
        per_agent.append(summary)

    cross = _compute_cross_agent_metrics(per_agent)
    elapsed = time.time() - started_at

    # Strip large embedding arrays from the on-disk results
    results = {
        "metadata": {
            "started_at": started_at,
            "elapsed_seconds": elapsed,
            "provider": args.provider,
            "model": args.model,
            "embedder": args.embedder,
            "sessions": args.sessions,
            "posts_per_session": args.posts_per_session,
            "seed": args.seed,
        },
        "per_agent": [
            {k: v for k, v in a.items() if not k.startswith("_")}
            for a in per_agent
        ],
        "cross_agent": cross,
    }
    results_path = args.output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _print_results_summary(per_agent, cross)
    print(f"\nresults written to: {results_path}")
    print(f"elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
