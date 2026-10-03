"""Durable spend ledger and hard budget guard for billed LLM calls."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings


class BudgetExceeded(RuntimeError):
    pass


class UsageLedger:
    _lock = threading.RLock()

    def __init__(self, path: str | None = None):
        self.path = path or settings().usage_db
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS llm_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, provider TEXT, model TEXT,
                purpose TEXT, run_id TEXT, node TEXT, tokens_in INTEGER, tokens_out INTEGER,
                cached_in INTEGER, reasoning INTEGER, cost_usd REAL)""")

    @contextmanager
    def _connection(self):
        with self._lock:
            conn = sqlite3.connect(self.path, timeout=30)
            try:
                with conn:
                    yield conn
            finally:
                conn.close()

    def record(self, *, provider, model, purpose, run_id, node, tokens_in, tokens_out,
               cached_in, reasoning, cost_usd):
        with self._connection() as conn:
            conn.execute("INSERT INTO llm_usage (ts, provider, model, purpose, run_id, node, tokens_in, "
                         "tokens_out, cached_in, reasoning, cost_usd) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                         (datetime.now(timezone.utc).isoformat(), provider, model, purpose, run_id, node,
                          tokens_in, tokens_out, cached_in, reasoning, cost_usd))

    def spent(self, provider: str | None = None) -> float:
        with self._connection() as conn:
            query = "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_usage"
            row = conn.execute(query + " WHERE provider=?", (provider,)).fetchone() if provider else conn.execute(query).fetchone()
        return float(row[0])

    def summary(self) -> dict:
        with self._connection() as conn:
            rows = conn.execute("SELECT provider, model, purpose, COUNT(*), SUM(tokens_in), SUM(tokens_out), "
                                "SUM(cached_in), SUM(reasoning), SUM(cost_usd) FROM llm_usage "
                                "GROUP BY provider, model, purpose").fetchall()
        groups = [{"provider": p, "model": m, "purpose": u, "calls": c, "tokens_in": ti or 0,
                   "tokens_out": to or 0, "cached_in": ci or 0, "reasoning": r or 0, "cost_usd": round(cost or 0, 6)}
                  for p, m, u, c, ti, to, ci, r, cost in rows]
        return {"total_cost_usd": round(sum(g["cost_usd"] for g in groups), 6),
                "calls": sum(g["calls"] for g in groups), "budget_usd": settings().budget_usd, "by_purpose": groups}

    def check(self, provider: str) -> None:
        budget = settings().budget_usd
        if provider == "openai" and budget >= 0 and self.spent("openai") >= budget:
            raise BudgetExceeded(f"LLM budget of ${budget:.2f} is exhausted. Raise LLM_BUDGET_USD in .env to continue.")
