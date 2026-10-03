"""Prompted LLM-judge baselines (Who&When protocols): all-at-once and step-by-step."""
from __future__ import annotations

import json

from .features import observable_run

ALL_AT_ONCE = ("You are an expert debugger of AI agent execution traces. The run below FAILED: its final answer is "
               "wrong and no error was raised. Identify the single step that CAUSED the failure — the earliest "
               "decisive mistake, not a later step that merely used a bad value. Respond with JSON only: "
               "{\"step\": <step_id integer>, \"reason\": \"<short>\"}")
STEP_BY_STEP = ("You are an expert debugger of AI agent execution traces. The run FAILED: its final answer is wrong. "
                "You are shown the trace up to the current step. Decide whether the CURRENT step is the decisive "
                "mistake that caused the failure (not a step that merely propagated an earlier error). "
                "Respond with JSON only: {\"is_error\": true|false, \"reason\": \"<short>\"}")


def serialize(run: dict, upto: int | None = None) -> str:
    observed = observable_run(run)
    lines = [f"Task: {run.get('prompt')}", f"Final answer: {json.dumps(run.get('final_answer'))}", "Steps:"]
    for step in observed["steps"][:upto]:
        inputs = step.get("input") if isinstance(step.get("input"), dict) else {}
        shown = {k: v for k, v in inputs.items() if k not in ("dependencies", "frozen_at", "task_family")}
        lines.append(json.dumps({"step": step["step_id"], "node": step["node_name"], "type": step["node_type"],
                                 "uses_steps": step.get("parent_step_ids", []), "input": shown,
                                 "output": step.get("output")}, default=str)[:1500])
    return "\n".join(lines)


def _ranking(run, first):
    ids = [s["step_id"] for s in run["steps"]]
    return ([first] if first in ids else []) + [i for i in ids if i != first]


def judge_all_at_once(run: dict, llm) -> list[int]:
    result = llm.complete_json(ALL_AT_ONCE, serialize(run), seed=3, purpose="judge_all_at_once", run_id=run["run_id"])
    step = (result.data or {}).get("step") if isinstance(result.data, dict) else None
    try:
        step = int(step)
    except (TypeError, ValueError):
        step = None
    return _ranking(run, step)


def judge_step_by_step(run: dict, llm) -> list[int]:
    for index, step in enumerate(run["steps"], start=1):
        user = serialize(run, index) + f"\nCurrent step: {step['step_id']} ({step['node_name']})"
        result = llm.complete_json(STEP_BY_STEP, user, seed=3, purpose="judge_step_by_step", run_id=run["run_id"])
        if isinstance(result.data, dict) and result.data.get("is_error") in (True, "true", "yes"):
            return _ranking(run, step["step_id"])
    return _ranking(run, run["steps"][-1]["step_id"])
