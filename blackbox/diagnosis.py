"""Failure diagnosis: the trained ensemble (M1+M2+M3) with a transparent heuristic fallback.

No outcome, gold answer or injection label is ever an input.
"""
from __future__ import annotations

import math
from time import perf_counter

from .explain import compare_runs
from .explain.evidence import DESCRIPTIONS, describe, factors_from_shap, summary_sentence
from .features import extract_features

# Explicit expert weights for the fallback / baseline; not trained, not SHAP.
HEURISTIC_WEIGHTS = {
    "tool_error": 4.5, "empty_output": 2.8, "data_age_days": 4.3,
    "source_disagreement": 4.0, "constraint_loss": 3.4,
    "value_inconsistency": 3.5, "reference_divergence": 0.25,
    "first_divergence": 5.0, "repetition": 4.0, "state_overwrite": 3.8,
    "premature_answer": 4.0, "missing_parent": 2.2, "invalid_number": 4.5,
}
_DESCRIPTIONS = DESCRIPTIONS


def _scaled(feature: str, value: float) -> float:
    if feature == "data_age_days":
        return min(max(math.expm1(value) - 1.0, 0.0) / 30.0, 1.5)
    if feature in ("constraint_loss", "source_disagreement"):
        return min(value * 8.0, 1.5)
    return value


def _empty(run, start, method):
    return {"run_id": run.get("run_id"), "p_fail": 0.0, "root_cause": None, "top_suspects": [], "step_scores": [],
            "evidence": {"summary": "No recorded steps are available to diagnose.", "data_flow": [],
                         "nearest_success": None, "diverges_at": None, "factors": []},
            "method": method, "model_status": "heuristic_not_trained" if method != "ensemble" else "trained",
            "latency_ms": round((perf_counter() - start) * 1000, 3)}


def heuristic(run: dict, successful_runs: list[dict] | None = None) -> dict:
    """Rank observable anomalies with fixed weights (uncalibrated)."""
    start = perf_counter()
    features = extract_features(run, successful_runs)
    rows, details = features["rows"], features["details"]
    if not rows:
        return _empty(run, start, "observable_evidence_heuristic")
    per_step = [{name: weight * _scaled(name, row[name]) for name, weight in HEURISTIC_WEIGHTS.items()
                 if _scaled(name, row[name]) > 0} for row in rows]
    scores = [sum(contributions.values()) for contributions in per_step]
    for i, detail in enumerate(details):
        for j in range(i):
            if detail["step"] in details[j]["descendants"] and scores[j] > 1.0:
                scores[i] -= min(2.5, scores[j] * 0.5)
    largest = max(scores)
    exp = [math.exp(max(-50.0, score - largest)) for score in scores]
    total = sum(exp)
    blame = [value / total for value in exp]
    order = sorted(range(len(rows)), key=lambda index: (-blame[index], index))
    ranked = [{"step": details[i]["step"], "node": details[i]["node"], "blame": round(blame[i], 6),
               "score": round(scores[i], 5)} for i in order]
    root = order[0]
    factors = [{"feature": f, "value": round(rows[root][f], 6), "contribution": round(c, 6),
                "description": describe(f, rows[root][f])}
               for f, c in sorted(per_step[root].items(), key=lambda pair: -pair[1])]
    p_fail = 1.0 / (1.0 + math.exp(-(largest - 2.4)))
    summary = " ".join(item["description"] for item in factors[:2]) or \
        "No strong observable anomaly was found. The ranking is inconclusive; inspect or replay a candidate step."
    return {"run_id": run.get("run_id"), "p_fail": round(p_fail, 6),
            "root_cause": {"step": details[root]["step"], "node": details[root]["node"], "confidence": round(blame[root], 6)},
            "top_suspects": ranked[:3], "step_scores": ranked,
            "evidence": {"summary": summary, "data_flow": [details[root]["step"]] + details[root]["descendants"],
                         "nearest_success": features["reference"], "diverges_at": features["first_divergence"],
                         "factors": factors, "attribution_method": "explicit_heuristic_contributions",
                         "limitations": "Scores are uncalibrated. Anomaly evidence is a hypothesis until tested by replay."},
            "method": "observable_evidence_heuristic", "model_status": "heuristic_not_trained",
            "latency_ms": round((perf_counter() - start) * 1000, 3)}


def _semantic_fallback() -> bool:
    """True when MiniLM/NLI could not load and hashed embeddings stand in (shown in the UI)."""
    from .features.semantic import shared_encoder
    return bool(shared_encoder().fallback)


def _ensemble_result(run, item, model, successful_runs, start):
    details, rows = item["details"], item["rows"]
    blame = item["blame"]
    order = sorted(range(item["n"]), key=lambda i: (-blame[i], i))
    ranked = [{"step": details[i]["step"], "node": details[i]["node"], "blame": round(float(blame[i]), 6),
               "m1": round(float(item["m1"][i]), 6), "m2": round(float(item["m2"][i]), 6),
               "m3": round(float(item["m3"][i]), 6)} for i in order]
    root = order[0]
    shap = model.shap(item, root)
    factors = factors_from_shap(shap, rows[root])
    votes = {}
    for key, label in (("m1", "Step Blame Transformer"), ("m2", "LightGBM ranker"), ("m3", "Anomaly autoencoder")):
        best = int(item[key].argmax())
        votes[key] = {"model": label, "step": details[best]["step"], "node": details[best]["node"],
                      "blame": round(float(item[key][best]), 4), "weight": model.weights.get(key, 0)}
    reference = extract_features(run, successful_runs) if successful_runs else {"reference": None, "first_divergence": None}
    node = details[root]["node"]
    return {"run_id": run.get("run_id"), "p_fail": round(item["p_fail"], 6),
            "root_cause": {"step": details[root]["step"], "node": node, "confidence": round(float(blame[root]), 6)},
            "top_suspects": ranked[:3], "step_scores": ranked,
            "evidence": {"summary": summary_sentence(node, factors), "data_flow": [details[root]["step"]] + details[root]["descendants"],
                         "nearest_success": reference["reference"], "diverges_at": reference["first_divergence"],
                         "factors": factors, "shap": {k: round(v, 4) for k, v in sorted(shap.items(), key=lambda kv: -abs(kv[1]))[:8]},
                         "model_votes": votes, "attribution_method": "lightgbm_treeshap",
                         "limitations": "The ranking is a learned hypothesis; replay from the blamed step tests it."},
            "method": "ensemble", "model_status": "trained", "model_version": model.version,
            "semantic_fallback": _semantic_fallback(),
            "weights": model.weights, "latency_ms": round((perf_counter() - start) * 1000, 3)}


def _abstain_threshold():
    """Ranking-score threshold chosen on validation (data/calibration.json), if measured."""
    import json
    from pathlib import Path
    from .config import ROOT
    path = Path(ROOT / "data" / "calibration.json")
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["abstain_below_ranking_score"]) if path.is_file() else None
    except (OSError, ValueError, KeyError):
        return None


def _mark_abstention(result):
    tau = _abstain_threshold()
    root = result.get("root_cause")
    result["abstain"] = bool(tau is not None and root and root["confidence"] < tau)
    result["abstain_threshold"] = tau
    if result["abstain"]:
        result["abstain_reason"] = (f"The top ranking score ({root['confidence']:.0%}) is below the threshold "
                                    f"({tau:.0%}) chosen on validation data; treat the top 3 suspects as candidates.")
    return result


def diagnose_many(runs: list[dict], successful_runs: list[dict] | None = None, method: str = "auto") -> list[dict]:
    from .models.ensemble import load_ensemble
    model = load_ensemble() if method in ("auto", "ensemble") else None
    if model is None:
        if method == "ensemble":
            raise RuntimeError("No trained ensemble is available")
        return [heuristic(run, successful_runs) for run in runs]
    start = perf_counter()
    scorable = [r for r in runs if r.get("steps")]
    items = iter(model.score(scorable)) if scorable else iter(())
    results = []
    for run in runs:
        if not run.get("steps"):
            results.append(_empty(run, start, "ensemble"))
            continue
        results.append(_ensemble_result(run, next(items), model, successful_runs, start))
    per_run = (perf_counter() - start) * 1000 / max(1, len(runs))
    for result in results:
        result["latency_ms"] = round(per_run, 3)
        _mark_abstention(result)
    return results


def diagnose(run: dict, successful_runs: list[dict] | None = None, method: str = "auto") -> dict:
    return diagnose_many([run], successful_runs, method)[0]


__all__ = ["diagnose", "diagnose_many", "heuristic", "compare_runs"]
