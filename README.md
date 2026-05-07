# doomscroll

Hermes Doomscroll Agent — a cognitive simulation of feed consumption. Four agent variants with different cognitive distortions scroll the same captured feed; the pilot measures how their belief sets and topic distributions diverge.

## What it does

Each agent runs a per-post pipeline (salience → engagement → mood/novelty update → episodic trace) over a deterministic snapshot of posts, then performs a nightly consolidation pass that turns recent fragments into long-term beliefs. After all sessions, the pilot computes:

- **belief drift** — how an agent's beliefs move over time
- **topic entropy** — concentration of an agent's belief embeddings
- **cross-agent divergence** — cosine distance between agents' belief sets

The snapshot is the experimental control: every variant sees the same posts in the same order, with embeddings pre-computed once.

## Agent variants

Defined in `src/doomscroll/config.py`:

- `balanced`
- `high_confirmation_bias`
- `high_novelty_hunger`
- `high_mood_volatility`

## Install

```sh
uv sync
```

Requires Python ≥ 3.12.

## Configuration

Pilot runs are described by a YAML file validated by Pydantic. Source of truth lives under `configs/`. The CLI is a thin wrapper:

```sh
# Built-in defaults: fake LLM, fake embedder, synthetic feed
uv run doomscroll-pilot

# A versioned experiment config
uv run doomscroll-pilot --config configs/experiments/anthropic_pilot.yaml

# Ad-hoc overrides without editing YAML
uv run doomscroll-pilot \
    --config configs/default.yaml \
    --set pilot.sessions=5 \
    --set llm.provider=anthropic \
    --set llm.model=claude-sonnet-4-6

# Resolve and inspect a config without running
uv run doomscroll-pilot --config configs/experiments/anthropic_pilot.yaml --print-config
```

`--set` takes dotted paths; values are parsed as YAML scalars (so `5`, `true`, `null`, `[a, b]` all work).

The resolved config is written to `<output_dir>/config.yaml` at run start, alongside `results.json` and per-agent SQLite DBs — every output directory is self-describing.

### Config schema

See `configs/default.yaml` for the full schema. Top-level sections:

| Section | Purpose |
| --- | --- |
| `output_dir` | Where DBs, `results.json`, and resolved `config.yaml` land |
| `llm` | Provider, model, temperature, max_tokens |
| `embedder` | Embedding provider + model |
| `feed` | Snapshot path (null → synthetic feed) |
| `pilot` | Variants, sessions, posts-per-session, seed, consolidation toggle |
| `memory` | `MemoryConfig` tunables (decay, centroid params, salience) |
| `consolidation` | `ConsolidationConfig` tunables (sampling, update caps) |

### Secrets

Copy `.env.example` and export the keys for the providers you use:

```sh
export ANTHROPIC_API_KEY=...    # llm.provider: anthropic
export OPENAI_API_KEY=...       # llm.provider: openai  OR  embedder.provider: openai
export OPENROUTER_API_KEY=...   # llm.provider: openrouter
```

## Layout

```
src/doomscroll/
    agent_loop.py       per-post pipeline + session driver
    consolidation.py    nightly fragments → beliefs pass
    memory.py           episodic traces, interest centroids
    measurement.py      drift, entropy, cross-agent divergence
    feed.py             snapshot capture/replay + synthetic feed
    llm.py              Anthropic / OpenAI / OpenRouter / fake adapters
    embedding.py        OpenAI / fake embedders
    persistence.py      SQLite store
    config.py           all tunables; agent variants
    cli.py              doomscroll-pilot entry point
    run_config.py       Pydantic schema for the YAML run config
configs/                YAML run configs (default + experiments)
tests/                  pytest suite
data/snapshots/         captured feed snapshots
```

## Tests

```sh
uv run pytest
```
