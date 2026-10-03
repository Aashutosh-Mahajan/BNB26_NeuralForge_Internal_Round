"""SQLite traces, checkpoints, and isolated content-addressed clean responses."""
from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def content_key(node, inputs):
    return hashlib.sha256(canonical({"implementation": "sandbox-v1", "node": node, "inputs": inputs}).encode()).hexdigest()

class Store:
    def __init__(self, path="data/traces.db"):
        self.path = str(path)
        self._lock = threading.RLock()
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.path == ":memory:" else None
        if self._memory is None:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS checkpoints (checkpoint_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, step_id INTEGER NOT NULL, state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS response_cache (cache_key TEXT PRIMARY KEY, output TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS runs_created_at ON runs(created_at DESC);
            """)

    @contextmanager
    def _connection(self):
        with self._lock:
            conn = self._memory or sqlite3.connect(self.path, timeout=30)
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
            finally:
                if self._memory is None:
                    conn.close()

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

    def get_cached(self, key):
        with self._connection() as conn:
            row = conn.execute("SELECT output FROM response_cache WHERE cache_key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def cache_output(self, key, output):
        with self._connection() as conn:
            conn.execute("INSERT OR IGNORE INTO response_cache VALUES (?, ?)", (key, canonical(output)))

    def close(self):
        if self._memory is not None:
            self._memory.close()

SQLiteStore = Store
