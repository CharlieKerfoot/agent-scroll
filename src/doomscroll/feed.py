"""Feed adapters: capture, snapshot, replay.

Architecture (per Issue 1 of the eng review):
    capture_once  ->  raw posts JSON
                ->  prepare_snapshot (dedupe + pre-embed)
                ->  snapshot JSON  (the deterministic input to all 4 agents)
                ->  JSONSnapshotFeed.iter_posts()  (per-agent replay)

The snapshot is the experimental control: every agent sees the same posts
in the same order at the same simulated arrival rate. Embeddings are
pre-computed ONCE in the snapshot so per-agent runs do zero embedding
calls (per the performance section of the eng review).

The capture layer is intentionally a stub. MoltBook has no formal API;
the crawler will be browser-driven (Playwright or similar). Fill in
`capture_moltbook_to_raw_json` when MoltBook auth + page structure are
available. Until then, use SyntheticFeed for development.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np

from doomscroll.agent_loop import Post
from doomscroll.embedding import Embedder


@dataclass(frozen=True)
class RawPost:
    """A pre-embedding post pulled from a feed source.

    No embedding yet -- that happens once during prepare_snapshot.
    """

    id: str
    author: str
    ts: float
    text: str


# Snapshot file schema version. Bump when format changes.
SNAPSHOT_SCHEMA_VERSION = 1


class Feed(Protocol):
    """Anything an agent loop can iterate over to receive Posts in order."""

    def iter_posts(self) -> Iterator[Post]: ...

    def __len__(self) -> int: ...


# ---------------- JSON snapshot replay (the experiment's input) ----------------


class JSONSnapshotFeed:
    """Deterministic replay from a snapshot file.

    Same file -> same posts in the same order for every agent. This is
    the required substrate for cross-agent comparability.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, "r") as f:
            blob = json.load(f)
        version = blob.get("metadata", {}).get("schema_version")
        if version != SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"snapshot schema version mismatch: file has {version}, "
                f"expected {SNAPSHOT_SCHEMA_VERSION}"
            )
        self.metadata = blob["metadata"]
        self._raw_posts = blob["posts"]

    def __len__(self) -> int:
        return len(self._raw_posts)

    def iter_posts(self) -> Iterator[Post]:
        for raw in self._raw_posts:
            embedding = raw.get("embedding")
            if embedding is None:
                raise ValueError(
                    f"snapshot post {raw.get('id')} missing embedding; "
                    "prepare_snapshot was not run before saving"
                )
            yield Post(
                id=raw["id"],
                author=raw["author"],
                ts=float(raw["ts"]),
                text=raw["text"],
                embedding=np.asarray(embedding, dtype=np.float64),
            )


# ---------------- Snapshot preparation (dedupe + pre-embed) ----------------


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def deduplicate_posts(posts: list[RawPost]) -> list[RawPost]:
    """Remove posts with identical text content. Keeps first occurrence.

    Twitter-shaped feeds have RTs and quote-tweets that produce the same
    surface text. Dedup at this layer prevents artificial reinforcement
    later (see eng review architecture inline note).
    """
    seen: set[str] = set()
    deduped: list[RawPost] = []
    for p in posts:
        h = _content_hash(p.text)
        if h in seen:
            continue
        seen.add(h)
        deduped.append(p)
    return deduped


def prepare_snapshot(
    raw_posts: list[RawPost],
    embedder: Embedder,
    output_path: str | Path,
    source: str = "unknown",
    embedding_model: str = "unknown",
    batch_size: int = 100,
) -> Path:
    """Embed and write a snapshot file.

    Calls the embedder ONCE per unique post (after dedup). After this
    function returns, agent runs do zero embedding work for posts.
    """
    deduped = deduplicate_posts(raw_posts)

    posts_payload: list[dict] = []
    for start in range(0, len(deduped), batch_size):
        batch = deduped[start : start + batch_size]
        embeddings = embedder.embed_batch([p.text for p in batch])
        for p, emb in zip(batch, embeddings):
            posts_payload.append(
                {
                    "id": p.id,
                    "author": p.author,
                    "ts": p.ts,
                    "text": p.text,
                    "embedding": [float(x) for x in emb],
                }
            )

    blob = {
        "metadata": {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "captured_at": time.time(),
            "source": source,
            "embedding_model": embedding_model,
            "post_count": len(posts_payload),
            "raw_count_before_dedup": len(raw_posts),
        },
        "posts": posts_payload,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(blob, f)
    return output_path


# ---------------- Synthetic feed for development ----------------


class SyntheticFeed:
    """In-memory feed of plausible synthetic posts. For dev / smoke tests.

    NOT cognitively realistic -- text is templated, not human-like. Use only
    to shake down the pipeline. For real experimental runs use a snapshot
    captured from MoltBook.
    """

    _TOPICS = (
        ("ai", ["models keep getting cheaper", "another agent demo today",
                "claude {n} dropped", "new benchmark says X is best"]),
        ("politics", ["election thing happening", "policy take incoming",
                      "outrage of the day", "another scandal"]),
        ("tech", ["startup pivoted again", "framework {n} released",
                  "deploy broke prod", "rust crab go brrr"]),
        ("food", ["soup is good", "made bread today", "cant find good ramen",
                  "pizza place closed"]),
        ("personal", ["cant sleep", "third coffee of the day",
                      "kid said something funny", "feeling fine"]),
    )

    def __init__(
        self,
        embedder: Embedder,
        n_posts: int = 100,
        seed: int = 0,
        start_ts: float = 0.0,
        post_interval_seconds: float = 30.0,
    ):
        self.embedder = embedder
        self.n_posts = n_posts
        self._rng = np.random.default_rng(seed)
        self.start_ts = start_ts
        self.post_interval = post_interval_seconds
        self._posts: list[Post] = self._build()

    def _build(self) -> list[Post]:
        # Build metadata + texts first; embed in one batch. The single-call
        # path was fine for FakeEmbedder but cost N API roundtrips for
        # OpenAIEmbedder (~1s/post, painfully slow at pilot scale).
        meta: list[tuple[str, str, float, str]] = []
        for i in range(self.n_posts):
            topic, templates = self._TOPICS[self._rng.integers(len(self._TOPICS))]
            template = templates[self._rng.integers(len(templates))]
            text = template.format(n=self._rng.integers(2, 9))
            post_id = f"syn-{i:06d}"
            ts = self.start_ts + i * self.post_interval
            author = f"user-{topic}"
            meta.append((post_id, author, ts, text))

        texts = [m[3] for m in meta]
        embeddings = self.embedder.embed_batch(texts)
        return [
            Post(id=pid, author=author, ts=ts, text=text, embedding=emb)
            for (pid, author, ts, text), emb in zip(meta, embeddings)
        ]

    def __len__(self) -> int:
        return len(self._posts)

    def iter_posts(self) -> Iterator[Post]:
        return iter(self._posts)


# ---------------- MoltBook capture stub ----------------


def capture_moltbook_to_raw_json(
    output_path: str | Path,
    auth_session_path: str | Path | None = None,
    max_posts: int = 5000,
) -> Path:
    """Capture a snapshot of MoltBook posts to JSON. NOT YET IMPLEMENTED.

    Implementation plan once MoltBook details are available:
    1. Launch headless browser (Playwright recommended) with persistent
       context at `auth_session_path` for sign-in cookies.
    2. Navigate to MoltBook root / feed URL.
    3. Scroll/paginate, collecting visible posts. Capture id, author,
       timestamp, text. Stop at max_posts.
    4. Write list[RawPost] as JSON to output_path.

    The output is the input to prepare_snapshot, which dedupes and embeds.
    """
    raise NotImplementedError(
        "MoltBook crawler not implemented. Need: feed URL, auth flow, "
        "post DOM structure (selectors for id/author/timestamp/text), "
        "pagination mechanism. Until then use SyntheticFeed for dev."
    )
