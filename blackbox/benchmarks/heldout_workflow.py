"""Held-out workflow: zero-shot diagnosis on an agent family never seen in training.

  python -m blackbox.benchmarks.heldout_workflow --output data/metrics_heldout_workflow.json

Uses the stock-LangGraph expense agent in examples/expense_agent.py (parallel branches,
reducers, its own acceptance check). One node is corrupted per run; the label is
that node's step. Local and free.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import random

from ..config import ROOT
from ..diagnosis import diagnose_many, heuristic
from ..models.train import ranking_metrics
from ..recorder.sdk import WrappedGraph
from ..storage import Store

FAULT_NODES = ["parse_request", "fetch_fx_rate", "fetch_hotel_price", "compute_total", "final_answer"]


def _agent_module():
    spec = importlib.util.spec_from_file_location("expense_agent", ROOT / "examples" / "expense_agent.py")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["expense_agent"] = module  # lets LangGraph resolve the module's type hints
    spec.loader.exec_module(module)
    return module


def main():
    from sklearn.metrics import roc_auc_score
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-fault", type=int, default=25)
    parser.add_argument("--output", default=str(ROOT / "data" / "metrics_heldout_workflow.json"))
    args = parser.parse_args()
    agent = _agent_module()
    rng = random.Random(11)
    store = Store(":memory:")
    runs, clean = [], []
    for i in range(args.per_fault * len(FAULT_NODES) + args.per_fault):
        city = rng.choice(list(agent.HOTEL_PER_NIGHT))
        request = f"Expense report: {rng.randint(1, 9)} nights in {city.title()} plus EUR {rng.randint(40, 900)} of meals. Total in USD?"
        gold = agent.expected_total(request)
        fault = FAULT_NODES[i % len(FAULT_NODES)] if i < args.per_fault * len(FAULT_NODES) else None
        agent.WORLD["fault"] = fault
        wrapped = WrappedGraph(agent.build(), store, name="expense-agent",
                               check=lambda s, gold=gold: abs(s.get("answer", -1) - gold) < 0.01)
        run = wrapped.invoke({"request": request, "log": []})
        agent.WORLD["fault"] = None
        if fault is None:
            clean.append(run)
            continue
        if run["success"] is False:
            run["label_step"] = next(s["step_id"] for s in run["steps"] if s["node_name"] == fault)
            run["fault_node"] = fault
            runs.append(run)
    diagnoses = diagnose_many(runs + clean)
    ranked = [[x["step"] for x in d["step_scores"]] for d in diagnoses[:len(runs)]]
    rules = [[x["step"] for x in heuristic(r)["step_scores"]] for r in runs]
    random_rank = [rng.sample([s["step_id"] for s in r["steps"]], len(r["steps"])) for r in runs]
    per_node = defaultdict(list)
    for r, rk in zip(runs, ranked):
        per_node[r["fault_node"]].append((r, rk))
    labels = [1] * len(runs) + [0] * len(clean)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "workflow": "expense-agent (never in training)",
           "failed_runs": len(runs), "clean_runs": len(clean), "cost_usd": 0.0,
           "black_box": ranking_metrics(runs, ranked), "rules": ranking_metrics(runs, rules),
           "random": ranking_metrics(runs, random_rank),
           "auroc": float(roc_auc_score(labels, [d["p_fail"] for d in diagnoses])),
           "per_fault_node": {k: ranking_metrics([p[0] for p in v], [p[1] for p in v]) for k, v in per_node.items()},
           "note": "Zero-shot: the models were trained only on the four built-in task families."}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("failed_runs", "black_box", "rules", "random", "auroc")}, indent=2))
    print({k: round(v["top1"], 2) for k, v in out["per_fault_node"].items()})


if __name__ == "__main__":
    main()
