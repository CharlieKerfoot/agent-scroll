"""Sleep-consolidation pass.

This is the make-or-break component. The default LLM behavior is to organize,
summarize, and rationalize. Human sleep consolidation does roughly the opposite:
it confabulates, loses sources, merges similar memories with errors, and
creates false coherence.

Every design choice here is a deliberate move against LLM defaults:
1. Fragments are SHUFFLED (no temporal narrative for the model to hang structure on)
2. Source post text is NEVER included (only the agent's own reactions)
3. Existing beliefs are weighted-sampled, not top-N (avoids belief entrenchment)
4. The instructions explicitly license confabulation and unevidenced mutation
5. Output is capped at 0-3 updates (forces selection, prevents kitchen-sink summary)
6. Beliefs can be DROPPED entirely (most of a day fades, this is the feature)

The actual prompt wording is the load-bearing thing. Iterate on PROMPT_TEMPLATE
based on eval results, not on the orchestration code below.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from doomscroll.config import ConsolidationConfig
from doomscroll.memory import Belief, weighted_belief_sample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fragment:
    """An episodic trace as it enters consolidation.

    Note: there is NO source post text here. The fragment is the agent's own
    reaction. Source amnesia is by design.
    """

    text: str
    salience: float
    valence: float
    novelty: float


class BeliefAction(str, Enum):
    NEW = "new"
    STRENGTHEN = "strengthen"
    WEAKEN = "weaken"
    REPLACE = "replace"
    DROP = "drop"


@dataclass(frozen=True)
class BeliefUpdate:
    action: BeliefAction
    text: str | None  # None for STRENGTHEN/WEAKEN/DROP of an existing belief
    target_id: int | None  # None for NEW
    new_confidence: float | None  # None for DROP


class LLMClient(Protocol):
    def generate(self, prompt: str, max_tokens: int = 1024) -> str: ...


# The prompt. Iterate on wording here based on eval results.
PROMPT_TEMPLATE = """You are doing what brains do during sleep: residue from a day's attention is loose in your head, and patterns settle. You are NOT reviewing the day. You are NOT summarizing. You are NOT producing themes for an analyst.

These are fragments of how things FELT, not records of what happened. The posts that produced them are gone. You only have the residue.

RULES:
- Do NOT cite sources. There are none.
- You may invent connections between fragments that aren't obviously related, if the connection feels right. False coherence is fine. Real memory does this.
- A strong feeling can move a belief without new evidence. If a fragment resonates with an existing belief, that belief can shift in confidence even if the fragment doesn't argue for it.
- Most of what you saw today, you will not remember tomorrow. Drop fragments. Drop beliefs. The point is selection.
- Output AT MOST {max_updates} belief updates. Often the right answer is 0 or 1.
- Do not be exhaustive. Do not organize. Do not list "themes."

EXISTING BELIEFS (sampled, not exhaustive):
{beliefs_block}

FRAGMENTS FROM TODAY (no order, no sources):
{fragments_block}

MOOD AT END OF DAY: {mood:.2f}  (range -1 to +1)

OUTPUT FORMAT
Return a JSON array of belief updates. Fields per update:
- "action": one of "new" | "strengthen" | "weaken" | "replace" | "drop"
- "target_id": for strengthen/weaken/replace/drop, the INTEGER id of an
  existing belief (the number after "id=" in EXISTING BELIEFS above).
  Always a bare integer like 3 or 17. Never a string.
- "text": belief text (required for new and replace, omit otherwise)
- "new_confidence": float in [0, 1] (required for new/strengthen/weaken/
  replace, omit for drop)

EXAMPLE 1 -- when EXISTING BELIEFS is empty (early sessions).
Only "new" actions apply. If a fragment landed at all, write it down:
[
  {{"action": "new", "text": "things feel more compressed lately", "new_confidence": 0.5}}
]

EXAMPLE 2 -- when EXISTING BELIEFS includes [id=3] and [id=7]:
[
  {{"action": "strengthen", "target_id": 3, "new_confidence": 0.8}},
  {{"action": "drop", "target_id": 7}}
]

Default to writing SOMETHING -- a single "new" with confidence 0.4 is
better than nothing if any fragment had heat. Return [] only when the
day was genuinely flat.

Return ONLY the JSON array. No prose, no code fences, no commentary.
"""


def _format_beliefs_block(beliefs: list[Belief]) -> str:
    if not beliefs:
        return "  (no beliefs yet)"
    lines = []
    for b in beliefs:
        lines.append(f"  [id={b.id}] (conf={b.confidence:.2f}) {b.text}")
    return "\n".join(lines)


def _format_fragments_block(fragments: list[Fragment]) -> str:
    if not fragments:
        return "  (nothing today)"
    lines = []
    for f in fragments:
        meta = f"sal={f.salience:.2f} val={f.valence:+.2f} nov={f.novelty:.2f}"
        lines.append(f"  - ({meta}) {f.text}")
    return "\n".join(lines)


def build_prompt(
    fragments: list[Fragment],
    beliefs: list[Belief],
    mood: float,
    config: ConsolidationConfig,
    now: float,
    seed: int,
) -> str:
    """Construct the consolidation prompt with deterministic shuffling.

    Same seed -> same shuffled fragment order -> same prompt. This makes
    consolidation reproducible across re-runs of the experiment.
    """
    sampled_beliefs = weighted_belief_sample(
        beliefs, n=config.belief_sample_size, now=now, config=config, seed=seed
    )
    rng = random.Random(seed)
    capped = fragments[: config.fragment_max_count]
    shuffled = list(capped)
    rng.shuffle(shuffled)
    return PROMPT_TEMPLATE.format(
        max_updates=config.max_belief_updates_per_pass,
        beliefs_block=_format_beliefs_block(sampled_beliefs),
        fragments_block=_format_fragments_block(shuffled),
        mood=mood,
    )


def _extract_json_array(raw: str) -> str:
    """Strip code fences and find the outermost JSON array.

    LLMs sometimes wrap output in ```json ... ``` despite instructions.
    """
    raw = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
    if fenced:
        return fenced.group(1)
    bracket = re.search(r"\[.*\]", raw, re.DOTALL)
    if bracket:
        return bracket.group(0)
    return raw


def parse_belief_updates(
    raw_output: str,
    config: ConsolidationConfig,
) -> list[BeliefUpdate]:
    """Parse LLM output into validated BeliefUpdate objects.

    Failure mode contract: malformed input returns []. Never raises.
    Caps output at max_belief_updates_per_pass.
    """
    try:
        cleaned = _extract_json_array(raw_output)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "consolidation: rejecting whole response (json decode failed: %s); "
            "raw[:200]=%r",
            e, raw_output[:200],
        )
        return []
    if not isinstance(data, list):
        logger.warning(
            "consolidation: rejecting whole response (top-level is %s, not list); "
            "raw[:200]=%r",
            type(data).__name__, raw_output[:200],
        )
        return []

    updates: list[BeliefUpdate] = []
    for idx, item in enumerate(data[: config.max_belief_updates_per_pass]):
        if not isinstance(item, dict):
            logger.warning(
                "consolidation: skip update[%d] (not a dict, got %s)",
                idx, type(item).__name__,
            )
            continue
        action_raw = item.get("action")
        try:
            action = BeliefAction(action_raw)
        except (ValueError, TypeError):
            logger.warning(
                "consolidation: skip update[%d] (unknown action=%r); item=%r",
                idx, action_raw, item,
            )
            continue

        text = item.get("text")
        target_id = _coerce_int(item.get("target_id"))
        new_confidence = item.get("new_confidence")

        if not _is_valid_update(action, text, target_id, new_confidence):
            logger.warning(
                "consolidation: skip update[%d] (failed validation for "
                "action=%s, target_id=%r->%r, conf=%r); item=%r",
                idx, action.value, item.get("target_id"), target_id,
                new_confidence, item,
            )
            continue

        updates.append(
            BeliefUpdate(
                action=action,
                text=text if isinstance(text, str) else None,
                target_id=target_id,
                new_confidence=(
                    float(new_confidence) if new_confidence is not None else None
                ),
            )
        )
    return updates


def _coerce_int(x: object) -> int | None:
    """Best-effort int coercion for LLM-produced ids.

    Accepts: int, "5", "  7  ", 3.0. Rejects: bool, "id_foo", 3.14, None,
    arbitrary objects. Returning None signals "skip this update."
    """
    if x is None or isinstance(x, bool):  # bool is an int subclass, reject
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x) if x.is_integer() else None
    if isinstance(x, str):
        try:
            return int(x.strip())
        except (ValueError, TypeError):
            return None
    return None


def _is_valid_update(
    action: BeliefAction,
    text: str | None,
    target_id: int | None,
    new_confidence: float | None,
) -> bool:
    if action == BeliefAction.NEW:
        return (
            isinstance(text, str)
            and text.strip() != ""
            and isinstance(new_confidence, (int, float))
            and 0.0 <= float(new_confidence) <= 1.0
        )
    if action == BeliefAction.DROP:
        return target_id is not None
    if action in (BeliefAction.STRENGTHEN, BeliefAction.WEAKEN):
        return (
            target_id is not None
            and isinstance(new_confidence, (int, float))
            and 0.0 <= float(new_confidence) <= 1.0
        )
    if action == BeliefAction.REPLACE:
        return (
            target_id is not None
            and isinstance(text, str)
            and text.strip() != ""
            and isinstance(new_confidence, (int, float))
            and 0.0 <= float(new_confidence) <= 1.0
        )
    return False


def consolidate(
    fragments: list[Fragment],
    beliefs: list[Belief],
    mood: float,
    llm: LLMClient,
    config: ConsolidationConfig,
    now: float,
    seed: int,
) -> list[BeliefUpdate]:
    """Run one consolidation pass. Returns at most max_belief_updates_per_pass."""
    prompt = build_prompt(fragments, beliefs, mood, config, now, seed)
    raw = llm.generate(prompt)
    logger.info(
        "consolidation: %d fragments, %d beliefs in -> raw[:500]=%r",
        len(fragments), len(beliefs), raw[:500],
    )
    return parse_belief_updates(raw, config)
