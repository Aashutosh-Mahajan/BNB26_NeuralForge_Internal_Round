"""Counterfactual labels for natural failures (FR-5).

For each step in order, a fixer proposes a corrected output; the run is replayed
from that step K times. The label is the earliest step whose fix passes in at
least ``threshold`` of K variants — the earliest decisive step, not a symptom.

Label hygiene: the default fixer is a careful re-run that never sees the expected
answer, and a step whose inputs are already broken upstream (an error output in a
dependency) is never labelled, because rewriting a downstream answer to the right
value can make a run pass without locating where it went wrong.
"""
from __future__ import annotations

import json
from copy import deepcopy

from ..agent.graph import LLM_NODES, NodeContext, execute
from ..llm.client import ChatLLM, SandboxLLM
from ..storage import canonical

FIXER_SYSTEM = ("You are a debugging assistant repairing one step of a failed agent run. You are given the task, "
                "the step's input, its recorded output and the correct final answer. Return a corrected output for "
                "THIS step only, in exactly the same JSON schema as the recorded output. Respond with JSON only.")


def _broken_inputs(step: dict) -> bool:
    deps = (step.get("input") or {}).get("dependencies", {}) if isinstance(step.get("input"), dict) else {}
    return any(isinstance(v, dict) and v.get("error") for v in deps.values())


def _fix_output(engine, run: dict, step: dict, fixer, allow_gold: bool = False):
    """A candidate corrected output for one step, or None if no different output exists."""
    family = run["task_family"]
    if step["node_name"] in LLM_NODES and isinstance(fixer, ChatLLM) and allow_gold:
        user = (f"Task: {run['prompt']}\nCorrect final answer: {json.dumps(run['gold_answer'])}\n"
                f"Step: {step['node_name']}\nStep input: {json.dumps(step['input'], default=str)[:6000]}\n"
                f"Recorded output: {json.dumps(step['output'], default=str)[:3000]}")
        result = fixer.complete_json(FIXER_SYSTEM, user, seed=11, purpose="counterfactual_fixer",
                                     run_id=run["run_id"], node=step["node_name"])
        output = result.data if result.error is None else None
    elif step["node_name"] in LLM_NODES and isinstance(fixer, ChatLLM):
        # Careful re-run by the same real model with a stricter instruction; never the expected answer.
        from ..strategies import STRICT_INSTRUCTIONS
        try:
            output, _ = execute(step["step_id"], family, step["input"],
                                NodeContext(llm=fixer, seed=run.get("seed", 7), variant=97, run_id=run["run_id"],
                                            purpose="counterfactual",
                                            overrides={"prompt": STRICT_INSTRUCTIONS.get(step["node_name"])}))
        except Exception:
            return None
    else:
        # A careful re-execution: the noise-free sandbox policy for LLM nodes, a retry for tools.
        llm = SandboxLLM(0.0) if step["node_name"] in LLM_NODES else engine._llm_for(run)
        try:
            output, _ = execute(step["step_id"], family, step["input"],
                                NodeContext(llm=llm, seed=run.get("seed", 7), variant=97, run_id=run["run_id"],
                                            purpose="counterfactual"))
        except Exception:  # A node that cannot run cleanly offers no candidate fix.
            return None
    if output is None or canonical(output) == canonical(step["output"]):
        return None
    return output


def label_run(engine, run_id: str, k: int = 3, threshold: int = 2, fixer=None, cleanup: bool = True,
              allow_gold: bool = False, find_all: bool = False) -> dict:
    run = engine.store.get_run(run_id)
    if run.get("success") is not False:
        raise ValueError("Counterfactual labeling applies to failed runs")
    trials, probes, label, passing, confidence = [], [], None, [], None
    for step in run["steps"]:
        if _broken_inputs(step):
            trials.append({"step": step["step_id"], "node": step["node_name"], "tested": False,
                           "reason": "inputs already broken upstream"})
            continue
        candidate = _fix_output(engine, run, step, fixer, allow_gold)
        if candidate is None:
            trials.append({"step": step["step_id"], "node": step["node_name"], "tested": False,
                           "reason": "no different output from a careful re-run"})
            continue
        result = engine.replay(run_id, step["step_id"], {"output": deepcopy(candidate)}, k=k, purpose="counterfactual")
        probes.extend(result["replay_run_ids"])
        trials.append({"step": step["step_id"], "node": step["node_name"], "tested": True,
                       "passed": result["passed"], "k": k})
        if result["passed"] >= threshold:
            passing.append(step["step_id"])
            if label is None:
                label, confidence = step["step_id"], result["passed"] / k
            if not find_all:
                break
    if cleanup and probes:
        engine.store.delete_runs(probes)
    run = engine.store.get_run(run_id)
    run.update(label_step=label, label_method="counterfactual" if label else "counterfactual_unresolved",
               label_candidates=passing, label_confidence=confidence,
               counterfactual={"k": k, "threshold": threshold, "trials": trials,
                               "fixer": fixer.label + " (sees gold answer)" if isinstance(fixer, ChatLLM) and allow_gold
                               else ("careful re-run by " + fixer.label + " with a stricter instruction (no gold answer)"
                                     if isinstance(fixer, ChatLLM) else "careful-rerun (no gold answer)"),
                               "uses_gold_answer": bool(isinstance(fixer, ChatLLM) and allow_gold),
                               "contributing_steps": passing if find_all else None})
    engine.store.save_run(run)
    return run
