"""Hard negatives: runs that PASS but look unusual. Measures false alarms (local, zero cost).

  python -m blackbox.hard_negatives --output data/metrics_hardneg.json

Categories:
  recovered_retry   a transient tool timeout recovered by a bounded retry
  unusual_values    extreme but valid task values (huge amounts, long terms, 0% discount)
  harmless_warning  a tool response carrying a harmless warning field, same value
Compared against the same number of planted failures, so AUROC says whether
Black Box separates "odd but fine" from "broken".
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random

from .agent.templates import TEMPLATES, render
from .config import ROOT
from .diagnosis import diagnose_many
from .engine import Engine, FAULT_CATALOG
from .llm.client import SandboxLLM

UNUSUAL = {
    "finance": [{"amount": 9500000, "months": 120, "annual_rate": 0.5}, {"amount": 2000, "months": 6, "annual_rate": 24},
                {"amount": 4750000, "months": 84, "annual_rate": 18}],
    "math": [{"quantity": 500, "unit_price": 3, "discount_pct": 0}, {"quantity": 1, "unit_price": 999, "discount_pct": 60},
             {"quantity": 250, "unit_price": 0.5, "discount_pct": 95}],
    "sql": [{"region": "east", "metric": "units", "quarter": "Q1"}, {"region": "west", "metric": "revenue", "quarter": "Q4"}],
    "doc_qa": [{"policy": "warranty", "category": "furniture", "product": "sofa"},
               {"policy": "price_match", "category": "groceries", "product": "grocery order"}],
}
# Training uses different extreme values and warning texts, so evaluation stays out of sample.
UNUSUAL_TRAIN = {
    "finance": [{"amount": 7000000, "months": 96, "annual_rate": 1}, {"amount": 3000, "months": 12, "annual_rate": 21}],
    "math": [{"quantity": 400, "unit_price": 2, "discount_pct": 0}, {"quantity": 2, "unit_price": 750, "discount_pct": 70}],
    "sql": [{"region": "south", "metric": "units", "quarter": "Q4"}, {"region": "north", "metric": "revenue", "quarter": "Q1"}],
    "doc_qa": [{"policy": "damage_claim", "category": "books", "product": "novel"},
               {"policy": "cancellation", "category": "toys", "product": "puzzle"}],
}
WARNINGS = {"eval": ["served from a read replica", "response took 2.1 s (slow upstream)", "1 field deprecated in API v2"],
            "train": ["cache warm-up in progress", "rate limit at 80% of quota", "using fallback region eu-west"]}
DEFAULTS = {"finance": {"base_currency": "INR", "target_currency": "USD"}, "math": {"item": "cable"}, "sql": {}, "doc_qa": {}}


def build(seed=7, per_category=40):
    rng = random.Random(seed)
    seed_offset = 0 if seed == 7 else 100000  # training sets use disjoint retry seeds
    cases_by_family = UNUSUAL if seed == 7 else UNUSUAL_TRAIN
    warnings = WARNINGS["eval" if seed == 7 else "train"]
    clean = Engine(":memory:", llm=SandboxLLM(0.0))
    runs = defaultdict(list)
    # Unusual but valid values, rendered through the normal task templates.
    for family, cases in cases_by_family.items():
        for case in cases:
            for index in range(len(TEMPLATES[family])):
                try:
                    prompt, params = render(family, index, {**DEFAULTS[family], **case})
                except AssertionError:
                    continue
                run = clean.run(prompt, family, {**params, "frozen_at": "2026-10-03T12:00:00+00:00"}, expose_params=False)
                if run["success"]:
                    runs["unusual_values"].append(run)
    # Recovered retries: noisy tools, keep runs that passed after at least one retry.
    noisy = Engine(":memory:", llm=SandboxLLM(0.0, tool_noise=0.6))
    for seed_i in range(3000):
        if len(runs["recovered_retry"]) >= per_category:
            break
        family = rng.choice(list(TEMPLATES))
        prompt, params = render(family, 0, {**DEFAULTS[family], **rng.choice(cases_by_family[family])})
        run = noisy.run(prompt, family, {**params, "frozen_at": "2026-10-03T12:00:00+00:00"}, expose_params=False,
                        seed=seed_i + seed_offset)
        if run["success"] and any(s["retries"] for s in run["steps"]):
            runs["recovered_retry"].append(run)
    # Harmless warnings: same value plus a warning field, recorded as a replay branch.
    base = runs["unusual_values"][:per_category]
    for run in base:
        step = next((s for s in run["steps"] if s["node_name"] in ("currency_rate", "doc_search", "schema_lookup", "line_items")), None)
        if step is None:
            continue
        output = {**deepcopy(step["output"]), "warning": rng.choice(warnings)}
        replay = clean.replay(run["run_id"], step["step_id"], {"output": output}, k=1)
        branch = clean.store.get_run(replay["run_id"])
        if branch["success"]:
            runs["harmless_warning"].append(branch)
    # Planted failures of the same tasks for comparison.
    failures = []
    for run in runs["unusual_values"]:
        fault = rng.choice([f for f in FAULT_CATALOG if not f["held_out"]])
        for step in run["steps"]:
            if step["node_type"] in fault["node_types"]:
                try:
                    failures.append(clean.inject(run["run_id"], step["step_id"], fault["id"]))
                    break
                except ValueError:
                    continue
    return runs, failures


def save_training_set(path: str, seed: int = 1234, per_category: int = 60):
    """Hard negatives for TRAINING (different seeds from the evaluation set), as passing runs."""
    import hashlib
    from .storage import Store
    runs, failures = build(seed=seed, per_category=per_category)
    store = Store(path)
    count = 0
    for category, items in runs.items():
        for run in items:
            split = "validation" if int(hashlib.sha256(run["run_id"].encode()).hexdigest()[:4], 16) % 5 == 0 else "train"
            store.save_run({**run, "split": split, "dataset_role": "hard_negative", "hard_negative": category})
            count += 1
    for run in failures:
        split = "validation" if int(hashlib.sha256(run["run_id"].encode()).hexdigest()[:4], 16) % 5 == 0 else "train"
        store.save_run({**run, "split": split, "dataset_role": "injected"})
    store.close()
    return {"hard_negatives": count, "failures": len(failures), "path": path}


def main():
    from sklearn.metrics import roc_auc_score
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default=str(ROOT / "data" / "metrics_hardneg.json"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--models", default=None, help="evaluate a different model folder (e.g. data/models_v2)")
    parser.add_argument("--save-training-set", default=None, help="write a training set (seed 1234) instead")
    args = parser.parse_args()
    if args.save_training_set:
        print(json.dumps(save_training_set(args.save_training_set), indent=2))
        return
    if args.models:
        from .models.ensemble import load_ensemble
        load_ensemble(args.models, reload=True)
    runs, failures = build()
    failure_scores = [d["p_fail"] for d in diagnose_many(failures)]
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "alarm_threshold": args.threshold, "cost_usd": 0.0,
           "models": args.models or "data/models (final)",
           "planted_failures": {"runs": len(failures),
                                "detected": sum(p >= args.threshold for p in failure_scores) / max(1, len(failures))},
           "categories": {}}
    all_neg = []
    for category, items in runs.items():
        scores = [d["p_fail"] for d in diagnose_many(items)]
        all_neg += scores
        out["categories"][category] = {"runs": len(items), "false_alarm_rate": sum(p >= args.threshold for p in scores) / max(1, len(scores)),
                                       "mean_p_fail": sum(scores) / max(1, len(scores)),
                                       "auroc_vs_failures": float(roc_auc_score([1] * len(failure_scores) + [0] * len(scores),
                                                                                failure_scores + scores)) if scores else None}
    out["overall"] = {"hard_negatives": len(all_neg),
                      "false_alarm_rate": sum(p >= args.threshold for p in all_neg) / max(1, len(all_neg)),
                      "auroc_vs_failures": float(roc_auc_score([1] * len(failure_scores) + [0] * len(all_neg), failure_scores + all_neg))}
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
