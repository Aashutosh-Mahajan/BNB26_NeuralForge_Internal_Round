"""Incident memory: what worked for similar, already-tested incidents.

Similarity is about task conditions, not node names alone: same task family, same
suspected node, and overlapping failed acceptance checks. Only tested branches count,
so a strategy is reused because it passed before, never by copying an old output.
"""
from __future__ import annotations

from collections import defaultdict


def _failed_checks(run: dict) -> set[str]:
    return {c["id"] for c in (run.get("acceptance") or {}).get("checks", []) if c["status"] == "fail"}


def similar_incidents(store, run: dict, step_id: int, limit: int = 500) -> dict:
    node = run["steps"][step_id - 1]["node_name"]
    family = run.get("task_family")
    checks = _failed_checks(run) - {"answer"}
    stats = defaultdict(lambda: {"tested": 0, "passed": 0, "incidents": set()})
    matched = []
    with store._connection() as conn:
        rows = conn.execute("SELECT data FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    import json
    for (data,) in rows:
        x = json.loads(data)
        if x["run_id"] == run["run_id"]:
            continue
        try:
            other = store.get_run(x["run_id"])
        except KeyError:
            continue
        if other.get("task_family") != family or other["steps"][x["step_id"] - 1]["node_name"] != node:
            continue
        overlap = _failed_checks(other) - {"answer"}
        if checks and overlap and not (checks & overlap):
            continue
        matched.append(x["run_id"])
        for branch in x["branches"]:
            if branch.get("tested"):
                row = stats[branch["id"]]
                row["tested"] += 1
                row["passed"] += branch["status"] == "passed"
                row["incidents"].add(x["run_id"])
    strategies = [{"strategy": k, "tested": v["tested"], "passed": v["passed"], "incidents": len(v["incidents"]),
                   "success_rate": v["passed"] / v["tested"]} for k, v in stats.items()]
    strategies.sort(key=lambda s: (-s["success_rate"], -s["tested"]))
    return {"run_id": run["run_id"], "step_id": step_id, "node": node, "family": family,
            "matched_incidents": len(set(matched)), "strategies": strategies,
            "matching": "same task family, same suspected step, overlapping failed checks"}
