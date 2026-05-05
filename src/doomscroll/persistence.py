"""SQLite persistence layer.

One DB file per agent. Pure stdlib sqlite3 -- no ORM, no sqlite-vec.
At week's end we expect ~5k traces and <200 beliefs per agent; in-memory
cosine over numpy arrays is faster than the dependency overhead of sqlite-vec
at this scale.

Schema invariants:
- sessions are per-agent runs. Every trace, post_seen, and snapshot belongs
  to exactly one session.
- consolidated_beliefs.active = 0 means dropped (kept for audit, not loaded
  into runtime).
- embeddings stored as float32 little-endian bytes. dim is fixed by the
  embedding provider; persistence does not enforce it.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from doomscroll.consolidation import BeliefAction, BeliefUpdate
from doomscroll.memory import Belief, Centroid

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL NOT NULL,
    ended_at REAL,
    posts_seen INTEGER NOT NULL DEFAULT 0,
    posts_engaged INTEGER NOT NULL DEFAULT 0,
    consolidation_run INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS posts_seen (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    post_id TEXT NOT NULL,
    ts REAL NOT NULL,
    salience_score REAL NOT NULL,
    decision TEXT NOT NULL,
    embedding BLOB,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS episodic_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    post_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    last_reactivated_at REAL NOT NULL,
    base_salience REAL NOT NULL,
    valence REAL NOT NULL,
    novelty REAL NOT NULL,
    fragment TEXT NOT NULL,
    embedding BLOB NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS consolidated_beliefs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    last_updated_at REAL NOT NULL,
    reinforcement_count INTEGER NOT NULL DEFAULT 1,
    text TEXT NOT NULL,
    confidence REAL NOT NULL,
    embedding BLOB NOT NULL,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS interest_vector_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    centroids_json TEXT NOT NULL,
    mood REAL NOT NULL,
    novelty_hunger REAL NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE INDEX IF NOT EXISTS idx_traces_active ON episodic_traces(last_reactivated_at);
CREATE INDEX IF NOT EXISTS idx_beliefs_active ON consolidated_beliefs(active);
"""


@dataclass(frozen=True)
class Trace:
    """A persisted episodic trace, with its DB-assigned id."""

    id: int
    session_id: int
    post_id: str
    created_at: float
    last_reactivated_at: float
    base_salience: float
    valence: float
    novelty: float
    fragment: str
    embedding: NDArray[np.float64]


def _embedding_to_bytes(emb: NDArray[np.floating]) -> bytes:
    return np.asarray(emb, dtype=np.float32).tobytes()


def _embedding_from_bytes(blob: bytes) -> NDArray[np.float64]:
    return np.frombuffer(blob, dtype=np.float32).astype(np.float64)


class Store:
    """Per-agent sqlite-backed store. Thread-safe via per-call connection use."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ----- sessions -----

    def start_session(self, started_at: float) -> int:
        cur = self._conn.execute(
            "INSERT INTO sessions (started_at) VALUES (?)", (started_at,)
        )
        return int(cur.lastrowid)

    def end_session(
        self,
        session_id: int,
        ended_at: float,
        posts_seen: int,
        posts_engaged: int,
        consolidation_run: bool,
    ) -> None:
        self._conn.execute(
            """UPDATE sessions
               SET ended_at = ?, posts_seen = ?, posts_engaged = ?,
                   consolidation_run = ?
               WHERE id = ?""",
            (
                ended_at,
                posts_seen,
                posts_engaged,
                1 if consolidation_run else 0,
                session_id,
            ),
        )

    def session_count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) AS n FROM sessions")
        return int(cur.fetchone()["n"])

    # ----- posts seen -----

    def record_post_seen(
        self,
        session_id: int,
        post_id: str,
        ts: float,
        salience_score: float,
        decision: str,
        embedding: NDArray[np.floating] | None,
    ) -> None:
        if decision not in ("engaged", "skipped"):
            raise ValueError(f"decision must be 'engaged' or 'skipped'; got {decision}")
        self._conn.execute(
            """INSERT INTO posts_seen
                  (session_id, post_id, ts, salience_score, decision, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                post_id,
                ts,
                salience_score,
                decision,
                _embedding_to_bytes(embedding) if embedding is not None else None,
            ),
        )

    def post_seen_count(self, session_id: int) -> tuple[int, int]:
        """(seen, engaged)."""
        cur = self._conn.execute(
            "SELECT decision, COUNT(*) AS n FROM posts_seen WHERE session_id = ? GROUP BY decision",
            (session_id,),
        )
        seen = 0
        engaged = 0
        for row in cur:
            seen += row["n"]
            if row["decision"] == "engaged":
                engaged += row["n"]
        return seen, engaged

    # ----- episodic traces -----

    def insert_trace(
        self,
        session_id: int,
        post_id: str,
        created_at: float,
        last_reactivated_at: float,
        base_salience: float,
        valence: float,
        novelty: float,
        fragment: str,
        embedding: NDArray[np.floating],
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO episodic_traces
                  (session_id, post_id, created_at, last_reactivated_at,
                   base_salience, valence, novelty, fragment, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                post_id,
                created_at,
                last_reactivated_at,
                base_salience,
                valence,
                novelty,
                fragment,
                _embedding_to_bytes(embedding),
            ),
        )
        return int(cur.lastrowid)

    def update_trace_reactivation(
        self, trace_id: int, new_base_salience: float, new_last_reactivated_at: float
    ) -> None:
        self._conn.execute(
            """UPDATE episodic_traces
               SET base_salience = ?, last_reactivated_at = ?
               WHERE id = ?""",
            (new_base_salience, new_last_reactivated_at, trace_id),
        )

    def load_traces(self, since_ts: float | None = None) -> list[Trace]:
        if since_ts is None:
            cur = self._conn.execute("SELECT * FROM episodic_traces")
        else:
            cur = self._conn.execute(
                "SELECT * FROM episodic_traces WHERE last_reactivated_at >= ?",
                (since_ts,),
            )
        return [_row_to_trace(row) for row in cur]

    def load_session_traces(self, session_id: int) -> list[Trace]:
        cur = self._conn.execute(
            "SELECT * FROM episodic_traces WHERE session_id = ?", (session_id,)
        )
        return [_row_to_trace(row) for row in cur]

    # ----- beliefs -----

    def insert_belief(
        self,
        text: str,
        embedding: NDArray[np.floating],
        confidence: float,
        now: float,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO consolidated_beliefs
                  (created_at, last_updated_at, text, confidence, embedding)
               VALUES (?, ?, ?, ?, ?)""",
            (now, now, text, confidence, _embedding_to_bytes(embedding)),
        )
        return int(cur.lastrowid)

    def update_belief(
        self,
        belief_id: int,
        confidence: float,
        now: float,
        text: str | None = None,
        increment_reinforcement: bool = True,
    ) -> None:
        if text is None:
            self._conn.execute(
                """UPDATE consolidated_beliefs
                   SET confidence = ?, last_updated_at = ?,
                       reinforcement_count = reinforcement_count + ?
                   WHERE id = ?""",
                (confidence, now, 1 if increment_reinforcement else 0, belief_id),
            )
        else:
            self._conn.execute(
                """UPDATE consolidated_beliefs
                   SET confidence = ?, last_updated_at = ?, text = ?,
                       reinforcement_count = reinforcement_count + ?
                   WHERE id = ?""",
                (
                    confidence,
                    now,
                    text,
                    1 if increment_reinforcement else 0,
                    belief_id,
                ),
            )

    def deactivate_belief(self, belief_id: int) -> None:
        self._conn.execute(
            "UPDATE consolidated_beliefs SET active = 0 WHERE id = ?", (belief_id,)
        )

    def load_active_beliefs(self) -> list[Belief]:
        cur = self._conn.execute(
            "SELECT * FROM consolidated_beliefs WHERE active = 1"
        )
        return [
            Belief(
                id=int(row["id"]),
                text=row["text"],
                embedding=_embedding_from_bytes(row["embedding"]),
                confidence=float(row["confidence"]),
                last_updated_at=float(row["last_updated_at"]),
            )
            for row in cur
        ]

    def apply_belief_updates(
        self,
        updates: Iterable[BeliefUpdate],
        embed_text: "callable[[str], NDArray[np.floating]]",
        now: float,
    ) -> None:
        """Apply parsed BeliefUpdate objects to the store.

        Caller provides an embedding function so persistence stays
        provider-agnostic. Embedding is only computed for new/replace.
        """
        for u in updates:
            if u.action == BeliefAction.NEW:
                assert u.text is not None and u.new_confidence is not None
                self.insert_belief(u.text, embed_text(u.text), u.new_confidence, now)
            elif u.action == BeliefAction.STRENGTHEN:
                assert u.target_id is not None and u.new_confidence is not None
                self.update_belief(u.target_id, u.new_confidence, now)
            elif u.action == BeliefAction.WEAKEN:
                assert u.target_id is not None and u.new_confidence is not None
                self.update_belief(u.target_id, u.new_confidence, now)
            elif u.action == BeliefAction.REPLACE:
                assert u.target_id is not None
                assert u.text is not None and u.new_confidence is not None
                self.update_belief(
                    u.target_id, u.new_confidence, now, text=u.text
                )
            elif u.action == BeliefAction.DROP:
                assert u.target_id is not None
                self.deactivate_belief(u.target_id)

    # ----- interest vector snapshots -----

    def save_interest_snapshot(
        self,
        session_id: int,
        ts: float,
        centroids: list[Centroid],
        mood: float,
        novelty_hunger: float,
    ) -> None:
        payload = json.dumps(
            [
                {
                    "embedding": c.embedding.astype(np.float32).tolist(),
                    "weight": float(c.weight),
                    "last_active": float(c.last_active),
                }
                for c in centroids
            ]
        )
        self._conn.execute(
            """INSERT INTO interest_vector_snapshots
                  (session_id, ts, centroids_json, mood, novelty_hunger)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, ts, payload, mood, novelty_hunger),
        )

    def load_latest_interest_snapshot(
        self,
    ) -> tuple[list[Centroid], float, float] | None:
        cur = self._conn.execute(
            "SELECT * FROM interest_vector_snapshots ORDER BY id DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        raw = json.loads(row["centroids_json"])
        centroids = [
            Centroid(
                embedding=np.asarray(item["embedding"], dtype=np.float64),
                weight=float(item["weight"]),
                last_active=float(item["last_active"]),
            )
            for item in raw
        ]
        return centroids, float(row["mood"]), float(row["novelty_hunger"])


def _row_to_trace(row: sqlite3.Row) -> Trace:
    return Trace(
        id=int(row["id"]),
        session_id=int(row["session_id"]),
        post_id=row["post_id"],
        created_at=float(row["created_at"]),
        last_reactivated_at=float(row["last_reactivated_at"]),
        base_salience=float(row["base_salience"]),
        valence=float(row["valence"]),
        novelty=float(row["novelty"]),
        fragment=row["fragment"],
        embedding=_embedding_from_bytes(row["embedding"]),
    )
