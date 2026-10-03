"""Evaluation suite (FR-12) -> data/metrics.json.

  python -m blackbox.evaluation --dataset data/dataset_sandbox.db
  python -m blackbox.evaluation --dataset data/dataset_sandbox.db --cross-dataset data/dataset_openai.db \
      --judge openai --judge-runs 100 --ablations

Every split is reported separately; unseen-fault results never enter the headline.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time

import numpy as np

from .config import ROOT, settings
from .diagnosis import diagnose_many, heuristic
from .models.ensemble import blend, load_ensemble
from .models.train import auroc, by_split, has_label, load_runs, rank_steps, ranking_metrics

logger = logging.getLogger("blackbox.evaluation")
UNSEEN_FAULTS = {"premature_final", "memory_overwrite", "loop_repetition"}


def _f1(labels, scores, threshold=0.5):
    tp = sum(y and s >= threshold for y, s in zip(labels, scores))
    fp = sum((not y) and s >= threshold for y, s in zip(labels, scores))
    fn = sum(y and s < threshold for y, s in zip(labels, scores))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None


def simple_baselines(runs, seed):
    rng = random.Random(seed)
    out = defaultdict(list)
    for run in runs:
        ids = [s["step_id"] for s in run["steps"]]
        shuffled = ids[:]
        rng.shuffle(shuffled)
        first_error = next((s["step_id"] for s in run["steps"] if s.get("tool_error")), None)
        out["random"].append(shuffled)
        out["last_step"].append(list(reversed(ids)))
        out["first_tool_error"].append(([first_error] if first_error else []) + [i for i in reversed(ids) if i != first_error])
    return out


class Scorer:
    """Scores runs once with every model so splits, ablations and baselines share work."""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def items(self, runs):
        missing = [r for r in runs if r["run_id"] not in self.cache]
        for i in range(0, len(missing), 256):
            chunk = missing[i:i + 256]
            for run, item in zip(chunk, self.model.score(chunk)):
                self.cache[run["run_id"]] = item
        return [self.cache[r["run_id"]] for r in runs]


def evaluate_split(runs, scorer, seed, judges=None):
    items = scorer.items(runs)
    weights = scorer.model.weights
    labels = [int(r.get("success") is False) for r in runs]
    p_fail = [item["p_fail"] for item in items]
    report = ranking_metrics(runs, [rank_steps(r, item["blame"]) for r, item in zip(runs, items)])
    report.update(total_runs=len(runs), failed_runs=sum(labels), passed_runs=len(runs) - sum(labels),
                  auroc=auroc(labels, p_fail) if runs else None, f1=_f1(labels, p_fail))
    failed = [(r, item) for r, item in zip(runs, items) if has_label(r)]
    fruns = [r for r, _ in failed]
    models = {}
    for key in ("m1", "m2", "m3"):
        models[key] = ranking_metrics(fruns, [rank_steps(r, item[key]) for r, item in failed])
    models["ensemble"] = ranking_metrics(fruns, [rank_steps(r, blend(item, weights)) for r, item in failed])
    report["models"] = models
    heur = [heuristic(r) for r in fruns]
    baselines = {name: ranking_metrics(fruns, ranks) for name, ranks in simple_baselines(fruns, seed).items()}
    baselines["heuristic_rules"] = ranking_metrics(fruns, [[s["step"] for s in h["step_scores"]] for h in heur])
    for name, ranks in (judges or {}).items():
        subset = [r for r in fruns if r["run_id"] in ranks]
        baselines[name] = {**ranking_metrics(subset, [ranks[r["run_id"]] for r in subset]), "subset": True}
    report["baselines"] = baselines
    groups = defaultdict(list)
    for r, item in failed:
        groups[("fault", r.get("fault_type") or r.get("label_method"))].append((r, item))
        groups[("family", r.get("task_family"))].append((r, item))
        groups[("cell", r.get("fault_type") or "natural", r.get("task_family"))].append((r, item))
    calc = lambda pairs: ranking_metrics([p[0] for p in pairs], [rank_steps(p[0], p[1]["blame"]) for p in pairs])
    report["per_fault"] = {k[1]: calc(v) for k, v in groups.items() if k[0] == "fault"}
    report["per_family"] = {k[1]: calc(v) for k, v in groups.items() if k[0] == "family"}
    report["heatmap"] = [{"fault": k[1], "family": k[2], **calc(v)} for k, v in sorted(groups.items(), key=str) if k[0] == "cell"]
    return report


def evaluate_replay(dataset, runs, seed, limit=40, k=3):
    """Top suspect -> automatic fix (retry / backup / historical median) -> K-variant replay."""
    from .engine import Engine
    engine = Engine(dataset)
    rng = random.Random(seed)
    failed = [r for r in runs if has_label(r)]
    rng.shuffle(failed)
    outcomes, avoided, tokens, times, correct_target, probes, no_fix = [], [], [], [], [], [], 0
    tokens_no_cache = []
    successes = [r for r in runs if r.get("success")]
    diagnoses = diagnose_many(failed[:limit])
    for run, diag in zip(failed[:limit], diagnoses):
        step = diag["root_cause"]["step"]
        correct_target.append(step == run["label_step"])
        options = [o for o in engine.suggest_fix(run["run_id"], step, successful_runs=successes)["options"]
                   if o.get("kind") != "output_substitution"]
        if not options:
            no_fix += 1
            outcomes.append(0.0)
            continue
        result = engine.replay(run["run_id"], step, options[0]["patch"], k=k, purpose="evaluation")
        probes += result["replay_run_ids"]
        outcomes.append(result["passed"] / k)
        avoided.append(result["avoided_steps_pct"])
        if result["tokens_saved_pct"] is not None:
            tokens.append(result["tokens_saved_pct"])
            tokens_no_cache.append(result["tokens_saved_pct_without_cache"])
        if result["time_saved_pct"] is not None:
            times.append(result["time_saved_pct"])
    engine.store.delete_runs(probes)
    engine.store.close()
    mean = lambda v: float(np.mean(v)) if v else None
    return {"attempted_runs": min(limit, len(failed)), "k": k, "fix_success_rate": mean(outcomes),
            "diagnosis_correct_rate": mean(correct_target), "no_fix_available": no_fix,
            "avoided_steps_pct": mean(avoided), "tokens_saved_pct": mean(tokens),
            "tokens_saved_pct_without_cache": mean(tokens_no_cache), "time_saved_pct": mean(times),
            "tokens_estimated": runs[0].get("tokens_estimated", True) if runs else True,
            "protocol": "Ensemble top suspect; first automatic fix (retry, backup source or historical median; "
                        "never the paired clean output); K variant replay from that checkpoint."}


def measure_latency(runs, count=30):
    sample = runs[:count]
    times = []
    for run in sample:
        start = time.perf_counter()
        diagnose_many([run])
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)) if times else None,
            "p95_ms": float(np.percentile(times, 95)) if times else None, "runs": len(times)}


def run_judges(runs, provider, limit, seed):
    from .judge import judge_all_at_once, judge_step_by_step
    from .llm import get_llm
    llm = get_llm(provider)
    rng = random.Random(seed)
    sample = [r for r in runs if has_label(r)]
    rng.shuffle(sample)
    sample = sample[:limit]
    ranks = {"llm_all_at_once": {}, "llm_step_by_step": {}}
    for i, run in enumerate(sample, 1):
        ranks["llm_all_at_once"][run["run_id"]] = judge_all_at_once(run, llm)
        ranks["llm_step_by_step"][run["run_id"]] = judge_step_by_step(run, llm)
        if i % 10 == 0:
            logger.info("LLM judge %d/%d", i, len(sample))
    return ranks, f"{llm.provider}:{llm.model}"


def run_evaluation(datasets, output, cross=None, judge=None, judge_runs=100, ablations=False, seed=42,
                   replay_limit=40, train_datasets=None):
    model = load_ensemble(reload=True)
    if model is None:
        raise RuntimeError("Train models first: python -m blackbox.models.train")
    runs = load_runs(datasets)
    splits = by_split(runs)
    scorer = Scorer(model)
    test = splits["test"]
    natural_test = [r for r in test if r.get("dataset_role") == "natural"]
    judges, judge_model = None, None
    if judge:
        judges, judge_model = run_judges(test, judge, judge_runs, seed)
    report_splits = {"validation": evaluate_split(splits["validation"], scorer, seed),
                     "test": evaluate_split(test, scorer, seed, judges),
                     "unseen_fault": evaluate_split(splits["unseen_fault"] + [r for r in test if r.get("success")], scorer, seed),
                     "natural_failures": evaluate_split(natural_test, scorer, seed) if natural_test else None}
    cross_report = None
    if cross:
        cross_runs = load_runs(cross)
        cross_report = evaluate_split(cross_runs, scorer, seed)
        cross_report["provider"] = sorted({r.get("model", "?") for r in cross_runs})
    headline = report_splits["test"]
    judge_top1 = max([headline["baselines"][k]["top1"] or 0 for k in ("llm_all_at_once", "llm_step_by_step")
                      if k in headline["baselines"]] or [0]) or None
    ablation_report = None
    if ablations:
        from .features import FEATURE_GROUPS, FEATURE_NAMES
        from .features.semantic import INPUT_DIM, build_matrices, shared_encoder
        from .models.train import train_all
        offset = INPUT_DIM - len(FEATURE_NAMES)
        unseen = splits["unseen_fault"]
        ablation_report = {}
        for group, names in FEATURE_GROUPS.items():
            logger.info("Ablation: retraining without %s features", group)
            trained = train_all(train_datasets or datasets, None, epochs=20, seed=seed, drop_features=names, save=False, log=logger.info)
            entry = {}
            for label, subset in (("test", test), ("unseen_fault", unseen)):
                encoded = build_matrices(subset, shared_encoder(), trained["stats"])
                for e in encoded:
                    for name in names:
                        e["numeric"][:, FEATURE_NAMES.index(name)] = 0
                        e["matrix"][:, offset + FEATURE_NAMES.index(name)] = 0
                items = trained["outputs"].compute(encoded, [[s["node_name"] for s in r["steps"]] for r in subset])
                entry[label] = ranking_metrics(subset, [rank_steps(r, blend(i, trained["meta"]["weights"]))
                                                        for r, i in zip(subset, items)])
            ablation_report[f"without_{group}"] = {**entry["test"], "unseen_top1": entry["unseen_fault"]["top1"],
                                                   "features": names}
        ablation_report["full_ensemble"] = {**headline["models"]["ensemble"],
                                            "unseen_top1": report_splits["unseen_fault"]["models"]["ensemble"]["top1"]}
        for key in ("m1", "m2", "m3"):
            ablation_report[f"{key}_alone"] = {**headline["models"][key],
                                               "unseen_top1": report_splits["unseen_fault"]["models"][key]["top1"]}
    from .llm.usage import UsageLedger
    manifest = {}
    for path in datasets:
        file = Path(path).with_suffix(".manifest.json")
        if file.is_file():
            manifest[str(path)] = json.loads(file.read_text(encoding="utf-8"))
    metrics = {
        "schema_version": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "measured", "method": "ensemble", "model_status": "trained", "model_version": model.version,
        "weights": model.weights,
        **{k: headline[k] for k in ("top1", "top3", "mrr", "within_one", "auroc", "f1")},
        "unseen_top1": report_splits["unseen_fault"]["top1"],
        "unseen_top3": report_splits["unseen_fault"]["top3"],
        "natural_top1": report_splits["natural_failures"]["top1"] if report_splits["natural_failures"] else None,
        "cross_provider_top1": cross_report["top1"] if cross_report else None,
        "llm_judge_top1": judge_top1, "llm_judge_model": judge_model,
        "llm_improvement_ratio": (headline["top1"] / judge_top1) if judge_top1 else None,
        "baselines": headline["baselines"], "models": headline["models"], "splits": report_splits,
        "cross_provider": cross_report, "heatmap": headline["heatmap"] + report_splits["unseen_fault"]["heatmap"],
        "per_fault": {**headline["per_fault"], **report_splits["unseen_fault"]["per_fault"]},
        "ablations": ablation_report,
        "replay": evaluate_replay(datasets[0], test, seed, replay_limit),
        "latency": measure_latency(test),
        "llm_cost": UsageLedger().summary(),
        "provenance": {"datasets": datasets, "train_datasets": train_datasets or datasets, "cross_datasets": cross, "seed": seed,
                       "counts": {k: len(v) for k, v in splits.items()}, "manifests": manifest,
                       "model": model.meta.get("provenance"), "semantic": model.meta.get("semantic"),
                       "unseen_faults": sorted(UNSEEN_FAULTS),
                       "split_strategy": "Disjoint prompt templates per split (8/2/2 of 12 per family); "
                                         "held-out fault types 10-12 only in the unseen-fault split.",
                       "limitations": [
                           "Sandbox agent with a fixed ten-step DAG; production traces will be noisier.",
                           "Injected faults are realistic but synthetic; natural failures come from the configured LLM provider.",
                           "LLM-judge baselines run on a subset of test runs to limit cost."]},
    }
    metrics["latency_ms"] = metrics["latency"]["mean_ms"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    return metrics


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append")
    parser.add_argument("--cross-dataset", action="append", default=None)
    parser.add_argument("--output", default=str(ROOT / "data" / "metrics.json"))
    parser.add_argument("--judge", choices=["openai", "ollama"], default=None)
    parser.add_argument("--judge-runs", type=int, default=100)
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--replay-limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-dataset", action="append", default=None, help="datasets used for ablation retraining")
    args = parser.parse_args()
    datasets = args.dataset or [str(ROOT / "data" / "dataset_sandbox.db")]
    metrics = run_evaluation(datasets, args.output, args.cross_dataset, args.judge, args.judge_runs,
                             args.ablations, args.seed, args.replay_limit, args.train_dataset)
    print(json.dumps({k: metrics[k] for k in ("top1", "top3", "mrr", "auroc", "unseen_top1", "natural_top1",
                                              "cross_provider_top1", "llm_judge_top1", "llm_improvement_ratio",
                                              "latency_ms")}, indent=2))


if __name__ == "__main__":
    main()
