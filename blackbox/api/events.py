"""Durable append-only events support live and late WebSocket subscribers."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._connection() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS api_events ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, "
                "payload TEXT NOT NULL)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_api_events_run ON api_events(run_id, id)"
            )

    @contextmanager
    def _connection(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def append(self, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
        payload = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(), **event}
        with self._connection() as connection:
            cursor = connection.execute(
                "INSERT INTO api_events (run_id, payload) VALUES (?, ?)",
                (run_id, json.dumps(payload, allow_nan=False)),
            )
            payload["event_id"] = cursor.lastrowid
        return payload

    def after(self, run_id: str, cursor: int = 0) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, payload FROM api_events WHERE run_id = ? AND id > ? ORDER BY id",
                (run_id, cursor),
            ).fetchall()
        return [{**json.loads(payload), "event_id": event_id} for event_id, payload in rows]
