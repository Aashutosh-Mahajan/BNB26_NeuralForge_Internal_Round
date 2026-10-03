"""SQLite traces, checkpoints, and isolated content-addressed clean responses."""
from __future__ import annotations
import hashlib
import json
import re
import time
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

SECRET_KEYS = re.compile(r"(api[_-]?key|authorization|password|passwd|secret|token|cookie)", re.I)
SECRET_VALUES = re.compile(r"(sk-[A-Za-z0-9_\-]{16,}|hf_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|Bearer\s+[A-Za-z0-9._\-]{16,}|"
                           r"xox[baprs]-[A-Za-z0-9-]{10,}|ghp_[A-Za-z0-9]{20,})")


def redact(value):
    """Mask credentials in recorded payloads (keys that look secret, values that look like keys)."""
    if isinstance(value, dict):
        return {k: ("[REDACTED]" if isinstance(k, str) and SECRET_KEYS.search(k) and not isinstance(v, (dict, list, int, float, bool))
                    and v not in (None, "") and "tokens" not in k.lower() else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return SECRET_VALUES.sub("[REDACTED]", value)
    return value


def content_key(node, inputs, extra=None):
    """hash(node + inputs [+ provider/model/seed for LLM calls]) -> cached response."""
    payload = {"implementation": "agent-v2", "node": node, "inputs": inputs}
    if extra:
        payload["extra"] = extra
    return hashlib.sha256(canonical(payload).encode()).hexdigest()

class Store:
    def __init__(self, path="data/traces.db"):
        self.path = str(path)
        self._lock = threading.RLock()
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # One long-lived connection per Store; access is serialized by the lock.
        self._memory = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        if self.path != ":memory:":
            self._memory.execute("PRAGMA journal_mode=WAL")
            self._memory.execute("PRAGMA synchronous=NORMAL")
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints (checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step_id INTEGER NOT NULL, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS response_cache (cache_key TEXT PRIMARY KEY, output TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS runs_created_at ON runs(created_at DESC);
                CREATE TABLE IF NOT EXISTS experiments (experiment_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS experiments_run ON experiments(run_id, created_at DESC);
            """)

    @contextmanager
    def _connection(self):
        with self._lock:
            conn = self._memory
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def save_run(self, run):
        with self._connection() as conn:
            conn.execute("INSERT INTO runs VALUES (?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET created_at=excluded.created_at,data=excluded.data", (run["run_id"], run["created_at"], canonical(run)))
        return run

    def get_run(self, run_id):
        with self._connection() as conn:
            row = conn.execute("SELECT data FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Run {run_id!r} does not exist")
        return json.loads(row[0])

    def list_runs(self, limit=1000):
        with self._connection() as conn:
            rows = conn.execute("SELECT data FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (int(limit),)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def save_checkpoint(self, checkpoint_id, run_id, step_id, state):
        with self._connection() as conn:
            conn.execute("INSERT INTO checkpoints VALUES (?, ?, ?, ?)", (checkpoint_id, run_id, step_id, canonical(state)))

    def get_checkpoint(self, checkpoint_id):
        with self._connection() as conn:
            row = conn.execute("SELECT state FROM checkpoints WHERE checkpoint_id=?", (checkpoint_id,)).fetchone()
        if row is None:
            raise KeyError(checkpoint_id)
        return json.loads(row[0])

    def get_cached(self, key, max_age_s=None):
        """A cached response, or None. ``max_age_s`` expires entries for changing data."""
        with self._connection() as conn:
            row = conn.execute("SELECT output FROM response_cache WHERE cache_key=?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if max_age_s is not None and isinstance(value, dict) and time.time() - value.get("cached_at", 0) > max_age_s:
            return None
        return value

    def cache_output(self, key, output):
        with self._connection() as conn:
            if isinstance(output, dict):
                output = {**output, "cached_at": time.time()}
            conn.execute("INSERT OR REPLACE INTO response_cache VALUES (?, ?)", (key, canonical(output)))

    def save_experiment(self, experiment):
        with self._connection() as conn:
            conn.execute("INSERT OR REPLACE INTO experiments VALUES (?, ?, ?, ?)",
                         (experiment["experiment_id"], experiment["run_id"], experiment["created_at"], canonical(experiment)))
        return experiment

    def list_experiments(self, run_id):
        with self._connection() as conn:
            rows = conn.execute("SELECT data FROM experiments WHERE run_id=? ORDER BY created_at DESC", (run_id,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete_runs(self, run_ids):
        with self._connection() as conn:
            conn.executemany("DELETE FROM runs WHERE run_id=?", [(r,) for r in run_ids])
            conn.executemany("DELETE FROM checkpoints WHERE run_id=?", [(r,) for r in run_ids])

    def count(self):
        with self._connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    def close(self):
        self._memory.close()

SQLiteStore = Store
