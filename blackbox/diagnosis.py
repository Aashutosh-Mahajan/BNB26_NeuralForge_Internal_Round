"""Transparent diagnostic baseline. No outcomes or injection labels are inputs."""
from __future__ import annotations
import math
from time import perf_counter

from .explain import compare_runs
from .features import extract_features

# These explicit expert weights are a heuristic, not a trained model or SHAP.
HEURISTIC_WEIGHTS = {
    "tool_error": 4.5, "empty_output": 2.8, "data_age_days": 4.3,
    "source_disagreement": 4.0, "constraint_loss": 3.4,
    "value_inconsistency": 3.5, "reference_divergence": 0.25,
    "first_divergence": 5.0, "repetition": 4.0, "state_overwrite": 3.8,
    "premature_answer": 4.0, "missing_parent": 2.2, "invalid_number": 4.5,
}
_DESCRIPTIONS = {
    "tool_error": "The tool recorded an error or timeout.",
    "empty_output": "The step returned an empty result or a null required value.",
    "data_age_days": "The returned data is {value:.1f} days older than the frozen run time.",
    "source_disagreement": "The returned value differs from an independently recorded source by {pct:.1f}%.",
    "constraint_loss": "The plan omitted or changed {pct:.0f}% of the recorded task constraints.",
    "value_inconsistency": "The output conflicts with a recorded dependency, tool choice, or current policy constraint.",
    "reference_divergence": "This output differs from the matching historical successful trace.",
    "first_divergence": "This is the earliest output that differs from the matching historical successful trace.",
    "repetition": "The node repeated an identical call or exceeded its recorded attempt budget.",
    "state_overwrite": "A memory step changed an established state value.",
    "premature_answer": "The route points to a final answer while recorded work remains.",
    "missing_parent": "A declared parent step was not present before this execution.",
    "invalid_number": "The output contains a non-finite numeric value.",
}


def _scaled(feature: str, value: float) -> float:
    if feature == "data_age_days":
        return min(max(value - 1.0, 0.0) / 30.0, 1.5)
    if feature == "constraint_loss":
        return min(value * 8.0, 1.5)
    if feature == "source_disagreement":
        return min(value * 8.0, 1.5)
    return value


def diagnose(run: dict, successful_runs: list[dict] | None = None) -> dict:
    """Rank observable anomalies, without consuming target outcome or gold labels.

    Blame and p_fail are uncalibrated heuristic scores, not measured probabilities.
    A suspected cause only becomes supported causal evidence after an intervention.
    """
    start = perf_counter()
    features = extract_features(run, successful_runs)
    rows, details = features["rows"], features["details"]
    if not rows:
        return {"run_id": run.get("run_id"), "p_fail": 0.0, "root_cause": None,
                "top_suspects": [], "step_scores": [],
                "evidence": {"summary": "No recorded steps are available to diagnose.",
                             "data_flow": [], "nearest_success": None, "diverges_at": None, "factors": []},
                "method": "observable_evidence_heuristic", "model_status": "heuristic_not_trained",
                "latency_ms": round((perf_counter() - start) * 1000, 3)}
    per_step = [{name: weight * _scaled(name, row[name]) for name, weight in HEURISTIC_WEIGHTS.items()
                 if _scaled(name, row[name]) > 0} for row in rows]
    scores = [sum(contributions.values()) for contributions in per_step]
    # If an ancestor has direct evidence, avoid treating propagation as equally causal.
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
    factors = []
    for feature, contribution in sorted(per_step[root].items(), key=lambda pair: -pair[1]):
        value = rows[root][feature]
        description = _DESCRIPTIONS[feature].format(value=value, pct=value * 100)
        factors.append({"feature": feature, "value": round(value, 6),
                        "contribution": round(contribution, 6), "description": description})
    p_fail = 1.0 / (1.0 + math.exp(-(largest - 2.4)))
    summary = " ".join(item["description"] for item in factors[:2])
    if not summary:
        summary = "No strong observable anomaly was found. The ranking is inconclusive; inspect or replay a candidate step."
    return {"run_id": run.get("run_id"), "p_fail": round(p_fail, 6),
            "root_cause": {"step": details[root]["step"], "node": details[root]["node"],
                           "confidence": round(blame[root], 6)},
            "top_suspects": ranked[:3], "step_scores": ranked,
            "evidence": {"summary": summary, "data_flow": [details[root]["step"]] + details[root]["descendants"],
                         "nearest_success": features["reference"], "diverges_at": features["first_divergence"],
                         "factors": factors, "attribution_method": "explicit_heuristic_contributions",
                         "limitations": "Scores are uncalibrated. Anomaly evidence is a hypothesis until tested by replay."},
            "method": "observable_evidence_heuristic", "model_status": "heuristic_not_trained",
            "latency_ms": round((perf_counter() - start) * 1000, 3)}


__all__ = ["diagnose", "compare_runs"]
