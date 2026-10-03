"""Reproducible synthetic evaluation with explicit held-out templates and faults.

Run: python -m blackbox.evaluation --output data/metrics.json --train-baseline
All examples are generated in an isolated in-memory Store. No API calls are made.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics

from .diagnosis import diagnose
from .models.linear import train_ranker

UNSEEN_FAULTS = {"premature_final", "memory_overwrite", "loop_repetition"}
FAMILIES = ("finance", "sql", "doc_qa", "math")
_TEMPLATES = {
    "train": ("Calculate the requested {family} result for case {case}.",
              "Solve this {family} task with the supplied constraints, case {case}."),
    "validation": ("Check the {family} scenario numbered {case} using its recorded inputs.",),
    "test": ("What result follows from the parameters in {family} case {case}?",
             "Please work out scenario {case} in the {family} workflow."),
    "unseen_fault": ("Resolve the pending {family} request with case reference {case}.",),
}


def _mean(values):
    return sum(values) / len(values) if values else None


def _auroc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label]
    negative = [score for label, score in zip(labels, scores) if not label]
    if not positive or not negative:
        return None
    return sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative) / (len(positive) * len(negative))


def _ranking_metrics(pairs):
    top1, top3, reciprocal, nearby = [], [], [], []
    for run, ranked in pairs:
        if run.get("success") is not False or run.get("label_step") is None:
            continue
        truth = run["label_step"]
        ids = [item["step"] if isinstance(item, dict) else item for item in ranked]
        rank = ids.index(truth) + 1 if truth in ids else None
        top1.append(float(rank == 1))
        top3.append(float(rank is not None and rank <= 3))
        reciprocal.append(1 / rank if rank else 0.0)
        step_positions = [step["step_id"] for step in run["steps"]]
        nearby.append(float(bool(ids) and ids[0] in step_positions and truth in step_positions and
                            abs(step_positions.index(ids[0]) - step_positions.index(truth)) <= 1))
    return {"count": len(top1), "top1": _mean(top1), "top3": _mean(top3),
            "mrr": _mean(reciprocal), "within_one": _mean(nearby)}


def evaluate_runs(runs: list[dict], seed=42) -> dict:
    """Labels are consumed here for scoring only, after predictions are computed."""
    predictions = [diagnose(run) for run in runs]  # Deliberately no clean paired references.
    report = _ranking_metrics([(run, pred["step_scores"]) for run, pred in zip(runs, predictions)])
    labels = [int(run.get("success") is False) for run in runs]
    scores = [prediction["p_fail"] for prediction in predictions]
    true_positive = sum(y == 1 and score >= 0.5 for y, score in zip(labels, scores))
    false_positive = sum(y == 0 and score >= 0.5 for y, score in zip(labels, scores))
    false_negative = sum(y == 1 and score < 0.5 for y, score in zip(labels, scores))
    denominator = 2 * true_positive + false_positive + false_negative
    report.update({"total_runs": len(runs), "failed_runs": sum(labels), "passed_runs": len(runs) - sum(labels),
                   "auroc": _auroc(labels, scores),
                   "f1": 2 * true_positive / denominator if denominator else 0.0,
                   "latency_ms": _mean([p["latency_ms"] for p in predictions]),
                   "latency_p95_ms": sorted(p["latency_ms"] for p in predictions)[max(0, int(len(predictions) * .95) - 1)]
                   if predictions else None})
    rng = random.Random(seed)
    baseline_pairs = defaultdict(list)
    fault_pairs, family_pairs, heatmap_pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    for run, prediction in zip(runs, predictions):
        ids = [step["step_id"] for step in run["steps"]]
        shuffled = ids[:]
        rng.shuffle(shuffled)
        first_error = next((step["step_id"] for step in run["steps"] if step.get("tool_error")), ids[-1] if ids else None)
        baseline_pairs["random"].append((run, shuffled))
        baseline_pairs["last_step"].append((run, list(reversed(ids))))
        baseline_pairs["first_tool_error"].append((run, ([first_error] if first_error is not None else []) +
                                                    [step for step in ids if step != first_error]))
        if run.get("success") is False:
            pair = (run, prediction["step_scores"])
            fault_pairs[run.get("fault_type", "unlabelled")].append(pair)
            family_pairs[run.get("task_family", "unknown")].append(pair)
            heatmap_pairs[(run.get("fault_type"), run.get("task_family"))].append(pair)
    report["baselines"] = {name: _ranking_metrics(pairs) for name, pairs in baseline_pairs.items()}
    report["baselines"].update(llm_all_at_once=None, llm_step_by_step=None)
    report["per_fault"] = {name: _ranking_metrics(pairs) for name, pairs in fault_pairs.items()}
    report["per_family"] = {name: _ranking_metrics(pairs) for name, pairs in family_pairs.items()}
    report["heatmap"] = [{"fault": fault, "family": family, **_ranking_metrics(pairs)}
                         for (fault, family), pairs in sorted(heatmap_pairs.items())]
    return report


def _params(family, rng, template_id, case):
    params = {"template_id": template_id, "frozen_at": f"2026-10-{3 + case % 20:02d}T12:00:00+00:00"}
    if family == "finance":
        params.update(amount=rng.randrange(10, 150) * 1000, months=rng.choice([6, 12, 18, 24, 36]),
                      annual_rate=rng.choice([4.5, 7.5, 9, 11, 13]))
    elif family == "sql":
        params.update(region=rng.choice(["north", "south", "east", "west"]), scale=rng.randint(1, 9))
    elif family == "doc_qa":
        params.update(policy="returns", days=rng.choice([14, 21, 30, 45, 60]), version="current")
    else:
        params.update(quantity=rng.randint(2, 30), unit_price=rng.randint(3, 150), discount_pct=rng.choice([5, 10, 15, 20]))
    return params


def generate_dataset(tasks_per_family=8, seed=42):
    from .engine import Engine, FAULT_CATALOG
    engine = Engine(":memory:")
    rng = random.Random(seed)
    dataset, skipped, template_sets = {}, [], {}
    counts = {"train": tasks_per_family, "validation": max(2, tasks_per_family // 4),
              "test": max(2, tasks_per_family // 4), "unseen_fault": max(2, tasks_per_family // 4)}
    for split, count in counts.items():
        dataset[split], templates = [], set()
        for family in FAMILIES:
            for case in range(count):
                template_index = case % len(_TEMPLATES[split])
                template_id = f"{family}-{split}-template-{template_index}"
                templates.add(template_id)
                params = _params(family, rng, template_id, case)
                prompt = _TEMPLATES[split][template_index].format(family=family, case=case)
                clean = engine.run(prompt, task_family=family, params=params)
                if not clean["success"]:
                    raise RuntimeError(f"Unexpected failed clean generator task: {family}/{split}/{case}")
                dataset[split].append(clean)
                faults = [fault for fault in FAULT_CATALOG
                          if (fault["id"] in UNSEEN_FAULTS) == (split == "unseen_fault")]
                for fault in faults:
                    candidates = [step for step in clean["steps"] if step["node_type"] in fault["node_types"]]
                    rng.shuffle(candidates)
                    last_error = "No compatible node"
                    for step in candidates:
                        try:
                            failed = engine.inject(clean["run_id"], step["step_id"], fault["id"])
                            dataset[split].append(failed)
                            break
                        except ValueError as error:
                            last_error = str(error)
                    else:
                        skipped.append({"split": split, "family": family, "fault": fault["id"], "reason": last_error})
        template_sets[split] = sorted(templates)
    sets = [set(values) for values in template_sets.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise AssertionError("Task templates leaked between dataset splits")
    identities = [{"split": split, "template": run["template_id"], "task": run["task_id"],
                   "fault": run.get("fault_type"), "step": run.get("label_step")}
                  for split, runs in dataset.items() for run in runs]
    fingerprint = hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest()
    return engine, dataset, {"seed": seed, "tasks_per_family": tasks_per_family,
                             "split_templates": template_sets, "dataset_sha256": fingerprint,
                             "skipped_incompatible_interventions": skipped}


def _evaluate_replay(engine, runs, seed):
    rng = random.Random(seed)
    failed = [run for run in runs if run.get("success") is False]
    rng.shuffle(failed)
    selected = failed[:32]
    outcomes, savings, unavailable = [], [], 0
    for run in selected:
        candidate = diagnose(run)["root_cause"]
        options = engine.suggest_fix(run["run_id"], candidate["step"])["options"]
        # Tool retry is available at inference; paired successful outputs are excluded.
        retry = next((option for option in options if option["id"] == "retry"), None)
        if retry is None:
            unavailable += 1
            outcomes.append(0.0)
            continue
        result = engine.replay(run["run_id"], candidate["step"], retry["patch"], k=1)
        outcomes.append(float(result["passed"] == 1))
        savings.append(result["avoided_steps_pct"])
    return {"attempted_runs": len(selected), "replayed_runs": len(savings),
            "no_retry_available": unavailable, "success_rate": _mean(outcomes),
            "avoided_steps_pct": _mean(savings), "tokens_saved_pct": None, "time_saved_pct": None,
            "protocol": "Top heuristic suspect, recompute that node at frozen time, one deterministic replay; no paired clean reference.",
            "note": "Step reuse is measured. Sandbox has no LLM tokens; token and latency savings are unmeasured."}


def run_evaluation(output="data/metrics.json", tasks_per_family=8, seed=42, train_baseline=False,
                   model_output="data/models/linear_ranker.json"):
    engine, dataset, provenance = generate_dataset(tasks_per_family, seed)
    splits = {split: evaluate_runs(runs, seed) for split, runs in dataset.items() if split != "train"}
    headline = splits["test"]
    metrics = {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
               "status": "measured_synthetic_sandbox", "method": "observable_evidence_heuristic",
               "model_status": "heuristic_not_trained",
               **{key: headline[key] for key in ("top1", "top3", "mrr", "within_one", "auroc", "f1", "latency_ms")},
               "unseen_top1": splits["unseen_fault"]["top1"], "natural_top1": None,
               "cross_provider_top1": None, "llm_improvement_ratio": None,
               "baselines": headline["baselines"], "splits": splits, "heatmap": headline["heatmap"],
               "replay": _evaluate_replay(engine, dataset["test"], seed),
               "trained_baseline": None,
               "provenance": {**provenance, "counts": {s: len(r) for s, r in dataset.items()},
                              "reference_runs_used_for_prediction": 0,
                              "split_strategy": "Disjoint prompt-template IDs and text forms; unseen fault types only in their separate split.",
                              "known_faults": sorted({r["fault_type"] for r in dataset["train"] if r.get("fault_type")}),
                              "unseen_faults": sorted(UNSEEN_FAULTS),
                              "baseline_fallback": "First-tool-error uses last step if there is no recorded error.",
                              "limitations": [
                                  "Small generated sandbox data, fixed ten-step DAG, and synthetic injections; not evidence of production generalization.",
                                  "No natural failures, cross-LLM test, public benchmark, LLM judges, or trained neural ensemble measured.",
                                  "Heuristic failure/blame scores are uncalibrated; AUROC measures ordering on this dataset only.",
                                  "Unseen faults are held out of logistic training; heuristic rules are authored and not learned."]}}
    if train_baseline:
        model = train_ranker(dataset["train"], seed=seed)
        model.save(model_output)
        metrics["trained_baseline"] = {"method": "supervised_logistic_step_ranker",
                                      "artifact": str(model_output), "provenance": model.artifact["provenance"],
                                      "splits": {split: _ranking_metrics([(run, model.predict(run)) for run in runs])
                                                 for split, runs in dataset.items() if split != "train"},
                                      "note": "Measured separately; live diagnosis still uses the transparent heuristic."}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/metrics.json")
    parser.add_argument("--tasks-per-family", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-baseline", action="store_true")
    parser.add_argument("--model-output", default="data/models/linear_ranker.json")
    args = parser.parse_args()
    if args.tasks_per_family < 2:
        parser.error("--tasks-per-family must be at least 2")
    metrics = run_evaluation(args.output, args.tasks_per_family, args.seed, args.train_baseline, args.model_output)
    print(json.dumps({"output": args.output, "status": metrics["status"], "top1": metrics["top1"],
                      "top3": metrics["top3"], "unseen_top1": metrics["unseen_top1"],
                      "auroc": metrics["auroc"], "counts": metrics["provenance"]["counts"]}, indent=2))


if __name__ == "__main__":
    main()
