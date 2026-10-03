"""Measure bounded live recovery (local, zero cost).

  python -m blackbox.live_eval --output data/metrics_live.json

- False interruptions: clean runs (incl. unusual-but-valid values) where recovery fired.
- Recovery: faults planted during execution; does the run still pass its checks?
Detection delay is 0 steps by construction: each step is checked when it finishes.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from .agent.templates import generate_tasks
from .config import ROOT
from .engine import Engine, FAULT_CATALOG
from .llm.client import SandboxLLM


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks-per-family", type=int, default=24)
    parser.add_argument("--output", default=str(ROOT / "data" / "metrics_live.json"))
    args = parser.parse_args()
    engine = Engine(":memory:", llm=SandboxLLM(0.0))
    rng = random.Random(5)
    tasks = generate_tasks(args.tasks_per_family, 99)
    clean_runs, interrupted = 0, 0
    per_fault = defaultdict(lambda: {"planted": 0, "failed_without": 0, "passed_with": 0, "escalated": 0})
    for task in tasks:
        params = {**task["params"], "template_id": task["template_id"], "frozen_at": task["frozen_at"]}
        run = engine.run(task["prompt"], task["family"], params, expose_params=False, live_recovery=True)
        clean_runs += 1
        interrupted += run["live_recovery"]["interruptions"] > 0
        spec = {i: kind for i, (_, kind, _) in enumerate(__import__("blackbox.agent.graph", fromlist=["x"]).specification(task["family"]), 1)}
        fault = rng.choice(FAULT_CATALOG)
        steps = [i for i, kind in spec.items() if kind in fault["node_types"]]
        if not steps:
            continue
        step = rng.choice(steps)
        try:
            without = engine.run(task["prompt"], task["family"], params, expose_params=False, faults={step: fault["id"]})
        except ValueError:
            continue
        if without["success"]:
            continue  # the fault did not change the outcome; nothing to recover
        with_live = engine.run(task["prompt"], task["family"], params, expose_params=False, faults={step: fault["id"]},
                               live_recovery=True)
        row = per_fault[fault["id"]]
        row["planted"] += 1
        row["failed_without"] += 1
        row["passed_with"] += bool(with_live["success"])
        row["escalated"] += with_live["live_recovery"]["escalated"]
        row["held_out"] = fault["held_out"]
    planted = sum(r["planted"] for r in per_fault.values())
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "cost_usd": 0.0,
           "clean_runs": clean_runs, "false_interruption_rate": interrupted / max(1, clean_runs),
           "planted_failures": planted, "recovered_rate": sum(r["passed_with"] for r in per_fault.values()) / max(1, planted),
           "escalated": sum(r["escalated"] for r in per_fault.values()), "detection_delay_steps": 0,
           "per_fault": {k: {**v, "recovered_rate": v["passed_with"] / max(1, v["planted"])} for k, v in sorted(per_fault.items())},
           "method": "rule-based prefix validators + catalogue repairs (no recorded-output substitution), max 2 attempts",
           "limitation": "Validators are hand-written; a learned prefix detector was not trained. Sandbox agent only."}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("clean_runs", "false_interruption_rate", "planted_failures", "recovered_rate", "escalated")}, indent=2))
    print({k: f"{v['passed_with']}/{v['planted']}" for k, v in out["per_fault"].items()})


if __name__ == "__main__":
    main()
