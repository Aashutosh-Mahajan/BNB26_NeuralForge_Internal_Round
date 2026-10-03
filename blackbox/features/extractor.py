"""Small inspectable observable feature set for the offline baseline.

This is not the PRD's MiniLM/NLI 412-dimensional pipeline. The dependency-free
baseline is deliberately labelled separately from the planned neural ensemble.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from typing import Any

FEATURE_NAMES = ["position", "fan_out", "latency", "tokens", "retries", "tool_error",
                 "empty_output", "data_age_days", "source_disagreement", "constraint_loss",
                 "value_inconsistency", "reference_divergence", "first_divergence", "repetition",
                 "state_overwrite", "premature_answer", "missing_parent", "output_size",
                 "descendant_count", "invalid_number"]
_RUN_FIELDS = ("run_id", "created_at", "frozen_at", "family", "task_family", "prompt", "task_id")
_STEP_FIELDS = ("step_id", "node_name", "node_type", "parent_step_ids", "input", "output",
                "state_before", "state_after", "tool_error", "as_of", "latency_ms",
                "tokens_in", "tokens_out", "retries")
_FORBIDDEN = {"fault_type", "label_step", "gold_answer", "success", "label_method",
              "expected_answer", "ground_truth", "injected_fault", "template_id",
              "task_id", "run_id", "parent_run_id", "checkpoint_id", "cache_key"}


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if k not in _FORBIDDEN}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def observable_run(run: dict) -> dict:
    """Explicit allowlists prevent outcome and injection metadata leaking in."""
    result = {key: _clean(run[key]) for key in _RUN_FIELDS if key in run}
    result["steps"] = [{key: _clean(step[key]) for key in _STEP_FIELDS if key in step}
                       for step in run.get("steps", [])]
    return result


def _date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp
    except (TypeError, ValueError):
        return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _all_values(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        result = [value[key]] if key in value else []
        for item in value.values():
            result.extend(_all_values(item, key))
        return result
    if isinstance(value, list):
        return [found for item in value for found in _all_values(item, key)]
    return []


def _descendants(steps: list[dict], start: Any) -> list[Any]:
    visited = {start}
    changed = True
    while changed:
        changed = False
        for step in steps:
            sid = step.get("step_id")
            if sid not in visited and any(parent in visited for parent in step.get("parent_step_ids", [])):
                visited.add(sid)
                changed = True
    return [s.get("step_id") for s in steps if s.get("step_id") in visited and s.get("step_id") != start]


def _constraints(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    constraints = value.get("constraints", value.get("requirements", {}))
    if isinstance(constraints, list):
        return {str(k): True for k in constraints}
    return constraints if isinstance(constraints, dict) else {}


def _nearest_reference(run: dict, references: list[dict]) -> dict | None:
    candidates = []
    at = _date(run.get("created_at"))
    family = run.get("task_family", run.get("family"))
    for candidate in references:
        ref = observable_run(candidate)
        if ref.get("run_id") == run.get("run_id"):
            continue
        if ref.get("task_family", ref.get("family")) != family:
            continue
        if not run.get("prompt") or ref.get("prompt") != run.get("prompt"):
            continue
        # Matching prompts alone are insufficient when parameters differ.
        a_input = run.get("steps", [{}])[0].get("input") if run.get("steps") else None
        b_input = ref.get("steps", [{}])[0].get("input") if ref.get("steps") else None
        if a_input != b_input:
            continue
        ref_at = _date(ref.get("created_at"))
        if at and ref_at and ref_at > at:
            continue
        candidates.append(ref)
    return max(candidates, key=lambda r: str(r.get("created_at", "")), default=None)


def extract_features(raw_run: dict, successful_runs: list[dict] | None = None) -> dict:
    run = observable_run(raw_run)
    steps = run["steps"]
    reference = _nearest_reference(run, successful_runs or [])
    ref_by_node: dict[str, list[dict]] = {}
    for ref_step in reference.get("steps", []) if reference else []:
        ref_by_node.setdefault(ref_step.get("node_name", ""), []).append(ref_step)
    occurrence: dict[str, int] = {}
    seen_calls: set[str] = set()
    first_divergence = None
    result, details = [], []
    stamp = _date(run.get("frozen_at", run.get("created_at")))
    seen_ids: set[Any] = set()
    for i, step in enumerate(steps):
        sid, name = step.get("step_id", i + 1), step.get("node_name", "unknown")
        node_type = str(step.get("node_type", name)).lower()
        output, inp = step.get("output"), step.get("input")
        before, after = step.get("state_before", {}), step.get("state_after", {})
        out = output if isinstance(output, dict) else {}
        inputs = inp if isinstance(inp, dict) else {}
        fan_out = sum(sid in child.get("parent_step_ids", []) for child in steps)
        descendants = _descendants(steps, sid)
        timestamps = ([step["as_of"]] if step.get("as_of") else []) + _all_values(output, "as_of")
        age = max([max(0.0, (stamp - date).total_seconds() / 86400)
                   for value in timestamps if stamp and (date := _date(value))] or [0.0])
        disagreement = 0.0
        for first, second in (("rate", "backup_rate"), ("value", "backup_value"),
                              ("rate", "reference_rate"), ("value", "source_value")):
            x, y = _number(out.get(first)), _number(out.get(second))
            if x is not None and y not in (None, 0):
                disagreement = max(disagreement, abs(x - y) / max(abs(y), 1e-9))
        requested = _constraints(inputs)
        if not requested and isinstance(inputs.get("params"), dict):
            requested = inputs["params"]
        produced = _constraints(out)
        constraint_loss = 0.0
        if requested and ("plan" in node_type or "plan" in name):
            ignored = {"template_id", "frozen_at"}
            keys = [k for k in requested if k not in ignored]
            constraint_loss = sum(k not in produced or produced[k] != requested[k] for k in keys) / max(1, len(keys))
        deps = inputs.get("dependencies", {})
        deps = deps if isinstance(deps, dict) else {}
        if "router" in node_type and isinstance(deps.get("planner"), dict):
            requested = _constraints(deps["planner"])
            produced = out.get("arguments", {})
            if isinstance(produced, dict) and requested:
                keys = [key for key in requested if key not in {"template_id", "frozen_at"}]
                constraint_loss = sum(produced.get(k) != requested[k] for k in keys) / max(1, len(keys))
        value_inconsistency = 0.0
        route = deps.get("router", {})
        arguments = route.get("arguments", {}) if isinstance(route, dict) else {}
        for key in ("quantity", "unit_price", "term_months"):
            expected_key = "months" if key == "term_months" else key
            if key in out and expected_key in arguments and out[key] != arguments[expected_key]:
                value_inconsistency = 1.0
        if "router" in node_type and out.get("tool"):
            children = [s for s in steps if sid in s.get("parent_step_ids", [])
                        and s.get("node_type") in ("tool", "retriever")]
            if children and out["tool"] != children[0].get("node_name"):
                value_inconsistency = 1.0
        if node_type in ("memory", "reasoner", "final") and len(deps) == 1:
            source = next(iter(deps.values()))
            if isinstance(source, dict) and "error" not in source:
                key = "value" if node_type == "memory" else "answer"
                expected = source.get("answer", source.get("value"))
                if key in out and expected is not None and out[key] != expected:
                    value_inconsistency = 1.0
        versions = _all_values(out, "version")
        if any(str(version).lower() in ("archived", "outdated", "expired") for version in versions):
            value_inconsistency = 1.0
        for key in ("order_id", "customer_id", "currency", "units", "document_id", "policy_version"):
            if key in inputs and key in out and inputs[key] != out[key]:
                value_inconsistency = 1.0
        tool_error = float(bool(step.get("tool_error")) or bool(out.get("error")) or out.get("status") in ("error", "timeout"))
        empty = float(output is None or output == [] or output == {} or output == "" or
                      any(k in out and out[k] is None for k in ("value", "rate", "answer", "result")))
        occurrence[name] = occurrence.get(name, 0) + 1
        corresponding = ref_by_node.get(name, [])
        ref = corresponding[occurrence[name] - 1] if len(corresponding) >= occurrence[name] else None
        diverged = float(reference is not None and (ref is None or ref.get("output") != output))
        if diverged and first_divergence is None:
            first_divergence = sid
        call = json.dumps([name, inp], sort_keys=True, default=str)
        repeated = float(call in seen_calls or (isinstance(out.get("attempts"), (int, float)) and isinstance(out.get("step_budget"), (int, float)) and out["attempts"] > out["step_budget"]))
        seen_calls.add(call)
        overwritten = 0.0
        if isinstance(before, dict) and isinstance(after, dict) and "memory" in node_type:
            overwritten = float(any(key in after and after[key] != value for key, value in before.items()
                                    if key not in ("memory", "history", "messages", "context")))
        premature = float(bool(out.get("skip_remaining")) or (node_type == "router" and out.get("tool") == "final_answer" and bool(descendants)) or
                          (out.get("next") in ("final", "final_answer") and
                           bool(inputs.get("pending_steps", inputs.get("remaining_steps")))))
        missing_parent = float(any(p not in seen_ids for p in step.get("parent_step_ids", [])))
        invalid = 0.0
        for key in ("value", "rate", "answer", "result", "amount"):
            value = _number(out.get(key))
            if value is not None and not math.isfinite(value):
                invalid = 1.0
        row = dict(zip(FEATURE_NAMES, [i / max(len(steps) - 1, 1), float(fan_out),
            math.log1p(max(0, _number(step.get("latency_ms")) or 0)),
            math.log1p(max(0, (_number(step.get("tokens_in")) or 0) + (_number(step.get("tokens_out")) or 0))),
            float(step.get("retries") or 0), tool_error, empty, age, min(disagreement, 10.0),
            constraint_loss, value_inconsistency, diverged, float(sid == first_divergence), repeated,
            overwritten, premature, missing_parent, math.log1p(len(json.dumps(output, default=str))),
            float(len(descendants)), invalid]))
        result.append(row)
        details.append({"step": sid, "node": name, "descendants": descendants, "age_days": age})
        seen_ids.add(sid)
    return {"rows": result, "matrix": [[r[name] for name in FEATURE_NAMES] for r in result],
            "details": details, "reference": reference.get("run_id") if reference else None,
            "first_divergence": first_divergence}
