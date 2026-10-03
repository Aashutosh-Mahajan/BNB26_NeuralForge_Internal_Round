"""Explanation builder (FR-8): SHAP reasons as sentences, data-flow chain,
nearest-success divergence, model votes and counterfactual result."""
from __future__ import annotations

import math

DESCRIPTIONS = {
    "tool_error": "The step recorded an error or timeout.",
    "empty_output": "The step returned an empty result or a null required value.",
    "data_age_days": "The returned data is {age:.0f} days older than the run time.",
    "source_disagreement": "The returned value differs from an independent second source by {pct:.1f}%.",
    "constraint_loss": "The output disagrees with the task's stated constraints ({pct:.0f}% of fields differ).",
    "value_inconsistency": "The output conflicts with a recorded dependency, tool choice or current policy.",
    "value_deviation": "The value is far outside the range seen in recent successful runs (robust z = {z:.0f}).",
    "contradiction": "An NLI model finds this step contradicted by the steps that use it (p = {value:.2f}).",
    "repetition": "The node repeated a call or exceeded its attempt budget.",
    "state_overwrite": "A memory step overwrote an established state value.",
    "premature_answer": "The route jumps to a final answer while work remains.",
    "missing_parent": "A declared parent step did not run before this step.",
    "invalid_number": "The output contains a non-finite number.",
    "fan_out": "{value:.0f} later steps consume this output directly.",
    "descendant_count": "{value:.0f} downstream steps depend on this output.",
    "position": "Its position in the run makes it a likely origin.",
    "output_size": "The output size is unusual for this node.",
    "latency": "The step latency is unusual.",
    "tokens": "The token usage is unusual for this node.",
    "retries": "The step needed retries.",
    "reference_divergence": "This output differs from the matching successful run.",
    "first_divergence": "This is the earliest output that differs from the matching successful run.",
}


def describe(feature: str, value: float) -> str:
    if feature.startswith("node_"):
        return f"Steps of type '{feature[5:]}' are frequent root causes for this pattern."
    template = DESCRIPTIONS.get(feature, feature.replace("_", " ").capitalize() + ".")
    age = math.expm1(value) if feature == "data_age_days" else value
    z = math.expm1(value) if feature == "value_deviation" else value
    return template.format(value=value, pct=value * 100, age=age, z=z)


def factors_from_shap(shap: dict, row: dict, limit: int = 6) -> list[dict]:
    ranked = sorted(((k, v) for k, v in shap.items() if v > 0.01), key=lambda kv: -kv[1])
    factors = []
    for feature, contribution in ranked[:limit]:
        value = float(row.get(feature, 1.0 if feature.startswith("node_") else 0.0))
        if not feature.startswith("node_") and value == 0 and feature not in ("position",):
            continue
        factors.append({"feature": feature, "value": round(value, 6), "contribution": round(contribution, 6),
                        "description": describe(feature, value)})
    return factors


def summary_sentence(node: str, factors: list[dict]) -> str:
    direct = [f["description"] for f in factors if f["feature"] not in ("position", "fan_out", "descendant_count")
              and not f["feature"].startswith("node_")]
    if not direct:
        return (f"{node} is the most likely origin, but no single strong anomaly stands out; "
                "replay it to confirm.")
    return f"{node}: " + " ".join(direct[:2])


def narrative(diagnosis: dict, run: dict) -> str:
    """Plain-English template narrative; an LLM or QLoRA explainer may replace it."""
    root = diagnosis.get("root_cause")
    if not root:
        return "No steps were recorded."
    evidence = diagnosis.get("evidence", {})
    flow = evidence.get("data_flow", [])
    text = (f"The run {'failed' if run.get('success') is False else 'finished'} with answer {run.get('final_answer')!r}. "
            f"Step {root['step']} ({root['node']}) is the most likely root cause "
            f"(blame {root['confidence']:.0%}, failure probability {diagnosis.get('p_fail', 0):.0%}). ")
    text += evidence.get("summary", "")
    if len(flow) > 1:
        text += f" Its output flows into steps {', '.join(map(str, flow[1:]))}, so those steps look wrong too but are symptoms."
    votes = evidence.get("model_votes", {})
    agree = [name for name, vote in votes.items() if vote.get("step") == root["step"]]
    if votes:
        text += f" {len(agree)} of {len(votes)} models agree."
    if evidence.get("diverges_at"):
        text += f" Compared with successful run {evidence.get('nearest_success')}, the trace first diverges at step {evidence['diverges_at']}."
    return text
