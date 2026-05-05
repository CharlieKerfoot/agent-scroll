"""Per-post pipeline and per-session orchestration.

Per-post pipeline:
    pull_post -> embed (already done in snapshot) -> estimate_valence
        -> compute_salience -> adaptive_threshold -> engage_or_skip
        -> [if engaged] react_via_llm -> update_centroids -> update_mood
        -> decay_novelty_hunger -> reactivate_neighbors -> write_trace
        -> record_post_seen

Per-session boundary (called once at start, once at end):
    start: load_state_from_store (centroids, mood, novelty_hunger)
    end:   save_interest_snapshot, run nightly_consolidation

Nightly consolidation:
    load_session_fragments + load_active_beliefs -> build_prompt
        -> llm.generate -> parse_belief_updates -> apply_belief_updates

Embedding rule: all post embeddings are PRE-COMPUTED in the snapshot. The
agent loop never calls the embedder during a session. Only consolidation
does, and only for new beliefs (~3 calls per session at most).
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from doomscroll.config import ConsolidationConfig, DistortionConfig, MemoryConfig
from doomscroll.consolidation import (
    Fragment,
    LLMClient,
    consolidate as run_consolidation,
)
from doomscroll.embedding import Embedder
from doomscroll.memory import (
    Belief,
    Centroid,
    decay_centroid_weights,
    decayed_salience,
    reactivate_trace,
    salience_score,
    update_centroids_on_engagement,
)
from doomscroll.persistence import Store, Trace


# ---------------- Data ----------------

@dataclass(frozen=True)
class Post:
    """A pre-embedded post from the snapshot."""

    id: str
    author: str
    ts: float
    text: str
    embedding: NDArray[np.float64]


@dataclass(frozen=True)
class ReactionResult:
    fragment: str
    valence: float
    novelty: float


@dataclass
class AgentState:
    """Mutable per-agent runtime state. Persisted between sessions via Store."""

    centroids: list[Centroid] = field(default_factory=list)
    mood: float = 0.0
    novelty_hunger: float = 0.7
    salience_threshold: float = 0.5
    posts_processed: int = 0
    posts_engaged: int = 0
    recent_salience_scores: list[float] = field(default_factory=list)


# ---------------- Cheap valence heuristic ----------------

_EXCLAIM = re.compile(r"!+")
_CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")
_QUESTION = re.compile(r"\?+")


def estimate_valence(text: str) -> float:
    """Cheap pre-LLM valence estimate. Used by salience gate.

    Counts emphasis markers; sign comes from negative-word presence.
    Refined post-engagement by the LLM reaction call.
    """
    if not text:
        return 0.0
    excl = len(_EXCLAIM.findall(text))
    caps = len(_CAPS_WORD.findall(text))
    qmark = len(_QUESTION.findall(text))
    intensity = min(1.0, (excl * 0.3 + caps * 0.2 + qmark * 0.1))

    # Crude polarity from common negative words
    lowered = text.lower()
    negative_words = (
        "hate", "terrible", "awful", "broken", "fail", "wrong", "bad",
        "disgust", "stupid", "ridiculous", "outrage", "angry",
    )
    has_negative = any(w in lowered for w in negative_words)
    sign = -1.0 if has_negative else 1.0
    return sign * intensity


# ---------------- Reaction prompt ----------------

REACTION_PROMPT = """You are mid-scroll on a social feed. You just glanced at a post.

Don't summarize it. Don't analyze it. React in your own words, briefly,
the way a thought would actually form in your head. Lowercase fine,
fragments fine.

POST:
{post_text}

OUTPUT a JSON object with exactly these fields:
- "fragment": 1-2 sentences, your reaction (NOT the post). Max 200 chars.
- "valence": float in [-1, 1]. How did it land?
- "novelty": float in [0, 1]. How new/surprising was this to you?

Return ONLY the JSON object. No prose.
"""


def _parse_reaction(raw: str) -> ReactionResult:
    """Parse LLM reaction output. Returns neutral default on failure."""
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1)
    else:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            raw = m.group(0)
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return ReactionResult(fragment="(unreadable reaction)", valence=0.0, novelty=0.5)

    fragment = data.get("fragment")
    if not isinstance(fragment, str) or not fragment.strip():
        fragment = "(empty reaction)"
    fragment = fragment[:200]

    def _clamp(v, lo, hi, default):
        try:
            return max(lo, min(hi, float(v)))
        except (TypeError, ValueError):
            return default

    valence = _clamp(data.get("valence"), -1.0, 1.0, 0.0)
    novelty = _clamp(data.get("novelty"), 0.0, 1.0, 0.5)
    return ReactionResult(fragment=fragment, valence=valence, novelty=novelty)


def react_to_post(llm: LLMClient, post: Post, max_tokens: int = 256) -> ReactionResult:
    prompt = REACTION_PROMPT.format(post_text=post.text)
    raw = llm.generate(prompt, max_tokens=max_tokens)
    return _parse_reaction(raw)


# ---------------- Adaptive salience threshold ----------------

def maybe_adapt_threshold(
    state: AgentState,
    mem_config: MemoryConfig,
) -> AgentState:
    """Adapt salience threshold to hit target engagement rate.

    After warmup, set threshold to the (1 - target_engagement_rate) percentile
    of recent salience scores. So 15% target -> 85th percentile threshold.
    """
    if state.posts_processed < mem_config.salience_warmup_posts:
        return state
    if not state.recent_salience_scores:
        return state
    percentile = 100.0 * (1.0 - mem_config.target_engagement_rate)
    new_threshold = float(np.percentile(state.recent_salience_scores, percentile))
    return replace(state, salience_threshold=new_threshold)


# ---------------- The per-post step ----------------

def session_step(
    state: AgentState,
    post: Post,
    store: Store,
    session_id: int,
    distortion: DistortionConfig,
    mem_config: MemoryConfig,
    llm: LLMClient,
    beliefs: list[Belief],
    now: float,
    reactivation_neighbors: int = 3,
) -> AgentState:
    """Process a single post. Returns the new state.

    Pure-ish: state in -> state out. Side effects are confined to `store`
    writes (record_post_seen, insert_trace, update_trace_reactivation).
    """
    state.posts_processed += 1

    estimated_valence = estimate_valence(post.text)
    salience = salience_score(
        post_embedding=post.embedding,
        centroids=state.centroids,
        beliefs=beliefs,
        mood=state.mood,
        novelty_hunger=state.novelty_hunger,
        distortion=distortion,
        estimated_valence=estimated_valence,
    )
    # Track for threshold adaptation (cap recent window to last 200)
    state.recent_salience_scores.append(salience)
    if len(state.recent_salience_scores) > 200:
        state.recent_salience_scores = state.recent_salience_scores[-200:]

    state = maybe_adapt_threshold(state, mem_config)

    if salience < state.salience_threshold:
        store.record_post_seen(
            session_id, post.id, post.ts, salience, "skipped", post.embedding
        )
        return state

    # Engaged path: LLM reaction call
    reaction = react_to_post(llm, post)
    state.posts_engaged += 1

    # Update centroids (EMA pull or spawn-with-eviction)
    state.centroids = update_centroids_on_engagement(
        state.centroids, post.embedding, now=now, config=mem_config
    )

    # Update mood: drift toward post valence, scaled by mood_volatility
    new_mood = state.mood + distortion.mood_volatility * (
        reaction.valence - state.mood
    )
    state.mood = max(-1.0, min(1.0, new_mood))

    # Novelty hunger decays per engagement (resets at session start)
    state.novelty_hunger = max(
        0.0, state.novelty_hunger - distortion.novelty_hunger_decay
    )

    # Reactivate nearest existing traces (spacing effect)
    _reactivate_nearest_traces(
        store=store,
        post_embedding=post.embedding,
        now=now,
        bump_strength=distortion.reactivation_strength,
        k=reactivation_neighbors,
    )

    # Write the new episodic trace. Initial salience seeds from gate score
    # plus emotional contagion bump (using actual valence now).
    initial_trace_salience = salience + (
        distortion.emotional_contagion * abs(reaction.valence)
    )
    store.insert_trace(
        session_id=session_id,
        post_id=post.id,
        created_at=now,
        last_reactivated_at=now,
        base_salience=initial_trace_salience,
        valence=reaction.valence,
        novelty=reaction.novelty,
        fragment=reaction.fragment,
        embedding=post.embedding,
    )

    store.record_post_seen(
        session_id, post.id, post.ts, salience, "engaged", post.embedding
    )
    return state


def _reactivate_nearest_traces(
    store: Store,
    post_embedding: NDArray[np.float64],
    now: float,
    bump_strength: float,
    k: int,
) -> None:
    """Find the k nearest existing traces and apply reactivation bumps."""
    traces = store.load_traces()
    if not traces:
        return
    embeddings = np.stack([t.embedding for t in traces])
    norms = np.linalg.norm(embeddings, axis=1)
    pn = float(np.linalg.norm(post_embedding))
    if pn < 1e-12:
        return
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    sims = (embeddings @ post_embedding) / (safe_norms * pn)
    top = np.argsort(-sims)[:k]
    for idx in top:
        sim = float(sims[idx])
        if sim <= 0:
            continue
        trace = traces[idx]
        new_sal, new_t = reactivate_trace(
            trace.base_salience, now=now, similarity=sim, bump_strength=bump_strength
        )
        store.update_trace_reactivation(trace.id, new_sal, new_t)


# ---------------- Nightly consolidation ----------------

def nightly_consolidation(
    store: Store,
    llm: LLMClient,
    embedder: Embedder,
    consolidation_config: ConsolidationConfig,
    session_id: int,
    mood: float,
    now: float,
    seed: int = 0,
) -> int:
    """Run consolidation over the just-completed session.

    Loads session fragments (sorted by salience desc, capped), samples
    existing beliefs (weighted), calls the LLM, applies updates.
    Returns the count of belief updates applied.
    """
    raw_traces = store.load_session_traces(session_id)
    if not raw_traces:
        return 0

    raw_traces.sort(key=lambda t: t.base_salience, reverse=True)
    capped = raw_traces[: consolidation_config.fragment_max_count]
    fragments = [
        Fragment(
            text=t.fragment,
            salience=t.base_salience,
            valence=t.valence,
            novelty=t.novelty,
        )
        for t in capped
    ]
    beliefs = store.load_active_beliefs()

    updates = run_consolidation(
        fragments=fragments,
        beliefs=beliefs,
        mood=mood,
        llm=llm,
        config=consolidation_config,
        now=now,
        seed=seed,
    )
    store.apply_belief_updates(updates, embed_text=embedder.embed, now=now)
    return len(updates)


# ---------------- Session boundaries ----------------

def begin_session(
    store: Store,
    distortion: DistortionConfig,
    started_at: float,
) -> tuple[int, AgentState]:
    """Open a session and rehydrate state from the latest snapshot."""
    session_id = store.start_session(started_at=started_at)
    snapshot = store.load_latest_interest_snapshot()
    if snapshot is None:
        state = AgentState(
            centroids=[],
            mood=0.0,
            novelty_hunger=distortion.novelty_hunger_initial,
            salience_threshold=0.5,
        )
    else:
        centroids, mood, _ = snapshot
        # Novelty hunger resets at session start (within-session decay only)
        state = AgentState(
            centroids=centroids,
            mood=mood,
            novelty_hunger=distortion.novelty_hunger_initial,
            salience_threshold=0.5,
        )
    return session_id, state


def end_session(
    store: Store,
    session_id: int,
    state: AgentState,
    mem_config: MemoryConfig,
    ended_at: float,
    consolidation_run: bool,
) -> None:
    """Close a session, decay centroid weights, and snapshot interest vector."""
    decayed = decay_centroid_weights(state.centroids, now=ended_at, config=mem_config)
    store.save_interest_snapshot(
        session_id=session_id,
        ts=ended_at,
        centroids=decayed,
        mood=state.mood,
        novelty_hunger=state.novelty_hunger,
    )
    store.end_session(
        session_id=session_id,
        ended_at=ended_at,
        posts_seen=state.posts_processed,
        posts_engaged=state.posts_engaged,
        consolidation_run=consolidation_run,
    )


def run_session(
    store: Store,
    posts: list[Post],
    distortion: DistortionConfig,
    mem_config: MemoryConfig,
    consolidation_config: ConsolidationConfig,
    llm: LLMClient,
    embedder: Embedder,
    started_at: float,
    ended_at: float,
    consolidate_at_end: bool = True,
    seed: int = 0,
) -> AgentState:
    """Full session: open, process all posts, optionally consolidate, close.

    `started_at` and `ended_at` define the simulated time window. Per-post
    `now` is interpolated linearly across this window to give traces
    realistic timestamps for decay computation.
    """
    session_id, state = begin_session(store, distortion, started_at)

    n = max(1, len(posts))
    for i, post in enumerate(posts):
        # Interpolate per-post timestamp linearly across the session
        now = started_at + (ended_at - started_at) * (i / n)
        beliefs = store.load_active_beliefs()  # cheap; <200 beliefs
        state = session_step(
            state=state,
            post=post,
            store=store,
            session_id=session_id,
            distortion=distortion,
            mem_config=mem_config,
            llm=llm,
            beliefs=beliefs,
            now=now,
        )

    if consolidate_at_end:
        nightly_consolidation(
            store=store,
            llm=llm,
            embedder=embedder,
            consolidation_config=consolidation_config,
            session_id=session_id,
            mood=state.mood,
            now=ended_at,
            seed=seed,
        )

    end_session(
        store=store,
        session_id=session_id,
        state=state,
        mem_config=mem_config,
        ended_at=ended_at,
        consolidation_run=consolidate_at_end,
    )
    return state
