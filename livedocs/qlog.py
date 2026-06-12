"""Question log — every /ask and /ask/stream question recorded in SQLite.

Single table, one row per question. Connections are opened per call (SQLite
is cheap to open and this keeps the module safe across FastAPI worker threads).
"""
import json
import sqlite3
from datetime import datetime, timezone

from livedocs.config import QUESTIONS_DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    confidence REAL,
    low_confidence INTEGER NOT NULL DEFAULT 0,
    sub_queries TEXT,
    n_sources INTEGER,
    duration_ms INTEGER,
    error TEXT
);
"""


def _connect() -> sqlite3.Connection:
    QUESTIONS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(QUESTIONS_DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute(_SCHEMA)
    return conn


def log_question(question, answer=None, confidence=None, low_confidence=False,
                 sub_queries=None, n_sources=None, duration_ms=None, error=None):
    """Best-effort insert — logging must never break the query path."""
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO questions (ts, question, answer, confidence, low_confidence,"
                " sub_queries, n_sources, duration_ms, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    question,
                    answer,
                    confidence,
                    1 if low_confidence else 0,
                    json.dumps(sub_queries) if sub_queries else None,
                    n_sources,
                    duration_ms,
                    error,
                ),
            )
    except Exception as e:
        print(f"  (question log failed: {type(e).__name__}: {e})")


def recent(limit=50, offset=0) -> list[dict]:
    try:
        with _connect() as conn:
            rows = conn.execute(
                "SELECT * FROM questions ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["sub_queries"] = json.loads(d["sub_queries"]) if d["sub_queries"] else []
            out.append(d)
        return out
    except Exception as e:
        print(f"  (question log read failed: {type(e).__name__}: {e})")
        return []


def count() -> int:
    try:
        with _connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
    except Exception:
        return 0


def clear() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM questions")
