"""Extra evaluation (local, zero API cost): calibration, abstention, position baseline,
PR-AUC, bootstrap confidence intervals and accuracy by trace length.

  python -m blackbox.eval_extra --dataset data/dataset_openai.db --output data/metrics_extra.json

The abstention threshold is chosen on the validation split only and written to
data/calibration.json, which live diagnosis reads. Never writes data/metrics.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random

import numpy as np

from .config import ROOT
from .models.ensemble import load_ensemble
from .models.train import by_split, has_label, load_runs, rank_steps, ranking_metrics


def _scores(model, runs):
    items = []
    for i in range(0, len(runs), 256):
        items += model.score(runs[i:i + 256])
    return items


def reliability(labels, probs, bins=10):
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi if hi < 1 else probs <= hi)
        if mask.any():
            conf, acc = probs[mask].mean(), labels[mask].mean()
            ece += mask.mean() * abs(conf - acc)
            rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "count": int(mask.sum()), "mean_predicted": float(conf),
                         "observed_rate": float(acc)})
    return {"ece": float(ece), "brier": float(np.mean((probs - labels) ** 2)), "bins": rows}


def abstention_curve(correct, scores, thresholds):
    correct, scores = np.asarray(correct, float), np.asarray(scores, float)
    out = []
    for t in thresholds:
        answered = scores >= t
        out.append({"threshold": float(t), "coverage": float(answered.mean()),
                    "accuracy_when_answering": float(correct[answered].mean()) if answered.any() else None})
    return out


def choose_threshold(curve, target=0.95):
    ok = [c for c in curve if c["accuracy_when_answering"] is not None and c["accuracy_when_answering"] >= target]
    return min(ok, key=lambda c: c["threshold"])["threshold"] if ok else max(c["threshold"] for c in curve)


def bootstrap(fn, n, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = [fn(rng.integers(0, n, n)) for _ in range(reps)]
    values = [v for v in values if v is not None]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None


def main():
    from sklearn.metrics import average_precision_score, roc_auc_score
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "dataset_openai.db"))
    parser.add_argument("--output", default=str(ROOT / "data" / "metrics_extra.json"))
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--no-whowhen", action="store_true")
    args = parser.parse_args()
    if Path(args.output).resolve() == (ROOT / "data" / "metrics.json").resolve():
        parser.error("Refusing to overwrite data/metrics.json")
    model = load_ensemble()
    splits = by_split(load_runs([args.dataset]))
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "dataset": Path(args.dataset).name,
           "model_version": model.version, "cost_usd": 0.0}

    # Position-only baseline: rank steps by how often each position was the origin in training.
    prior = Counter(r["label_step"] for r in splits["train"] if has_label(r))
    order = [step for step, _ in prior.most_common()]
    test = splits["test"]
    labelled = [r for r in test if has_label(r)]
    out["position_only_baseline"] = {
        **ranking_metrics(labelled, [order + [s["step_id"] for s in r["steps"] if s["step_id"] not in order] for r in labelled]),
        "train_position_frequency": dict(prior.most_common())}

    # Calibration and abstention: thresholds from validation, measurement on test.
    val = splits["validation"]
    val_items, test_items = _scores(model, val), _scores(model, test)
    def top_score(item):
        return float(np.max(item["blame"]))
    val_l = [r for r in val if has_label(r)]
    val_li = [i for r, i in zip(val, val_items) if has_label(r)]
    val_correct = [rank_steps(r, i["blame"])[0] == r["label_step"] for r, i in zip(val_l, val_li)]
    thresholds = np.round(np.linspace(0.0, 0.9, 19), 3)
    val_curve = abstention_curve(val_correct, [top_score(i) for i in val_li], thresholds)
    tau = choose_threshold(val_curve, args.target_accuracy)
    test_li = [i for r, i in zip(test, test_items) if has_label(r)]
    test_correct = [rank_steps(r, i["blame"])[0] == r["label_step"] for r, i in zip(labelled, test_li)]
    test_curve = abstention_curve(test_correct, [top_score(i) for i in test_li], thresholds)
    at_tau = next(c for c in test_curve if c["threshold"] == tau)
    out["abstention"] = {"threshold_from_validation": tau, "target_accuracy": args.target_accuracy,
                         "test_at_threshold": at_tau, "validation_curve": val_curve, "test_curve": test_curve,
                         "meaning": "Below the threshold Black Box says 'not sure' and shows the top 3 suspects instead."}
    labels = [int(r.get("success") is False) for r in test]
    pfail = [i["p_fail"] for i in test_items]
    out["calibration_p_fail"] = {**reliability(labels, pfail), "pr_auc": float(average_precision_score(labels, pfail)),
                                 "auroc": float(roc_auc_score(labels, pfail))}
    out["ranking_score_reliability"] = reliability(test_correct, [top_score(i) for i in test_li])

    # Bootstrap 95% confidence intervals on the test split.
    ranks = [rank_steps(r, i["blame"]) for r, i in zip(labelled, test_li)]
    top1 = np.array([rk[0] == r["label_step"] for r, rk in zip(labelled, ranks)], float)
    top3 = np.array([r["label_step"] in rk[:3] for r, rk in zip(labelled, ranks)], float)
    lab, pf = np.array(labels), np.array(pfail)
    out["confidence_intervals_95"] = {
        "top1": {"value": float(top1.mean()), "ci": bootstrap(lambda idx: top1[idx].mean(), len(top1)), "n": len(top1)},
        "top3": {"value": float(top3.mean()), "ci": bootstrap(lambda idx: top3[idx].mean(), len(top3)), "n": len(top3)},
        "auroc": {"value": float(roc_auc_score(lab, pf)), "n": len(lab),
                  "ci": bootstrap(lambda idx: roc_auc_score(lab[idx], pf[idx]) if 0 < lab[idx].sum() < len(idx) else None, len(lab))},
    }
    Path(ROOT / "data" / "calibration.json").write_text(json.dumps(
        {"abstain_below_ranking_score": tau, "chosen_on": "validation split", "target_accuracy": args.target_accuracy,
         "model_version": model.version, "generated_at": out["generated_at"]}, indent=2), encoding="utf-8")

    if not args.no_whowhen:
        from .benchmarks.whowhen import load_cases
        from .features.semantic import build_matrices, shared_encoder
        from .models.ensemble import blend
        cases = load_cases()
        items = model.outputs.compute(build_matrices(cases, shared_encoder(), model.stats),
                                      [[s["node_name"] for s in c["steps"]] for c in cases])
        buckets = {"1-10": (1, 10), "11-24": (11, 24), "25-50": (25, 50), "51+": (51, 10 ** 6)}
        rng = random.Random(0)
        table = {}
        for name, (lo, hi) in buckets.items():
            idx = [i for i, c in enumerate(cases) if lo <= len(c["steps"]) <= hi]
            if not idx:
                continue
            sub = [cases[i] for i in idx]
            table[name] = {"cases": len(idx),
                           "ensemble": ranking_metrics(sub, [rank_steps(cases[i], blend(items[i], model.weights)) for i in idx])["top1"],
                           "transformer_windows": ranking_metrics(sub, [rank_steps(cases[i], items[i]["m1"]) for i in idx])["top1"],
                           "random": ranking_metrics(sub, [rng.sample([s["step_id"] for s in c["steps"]], len(c["steps"])) for c in sub])["top1"]}
        out["whowhen_by_length"] = table
    Path(args.output).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"position_only_top1": out["position_only_baseline"]["top1"], "abstain_threshold": tau,
                      "test_at_threshold": at_tau, "p_fail_ece": out["calibration_p_fail"]["ece"],
                      "pr_auc": out["calibration_p_fail"]["pr_auc"], "ci": out["confidence_intervals_95"],
                      "whowhen_by_length": out.get("whowhen_by_length")}, indent=2, default=float))


if __name__ == "__main__":
    main()
