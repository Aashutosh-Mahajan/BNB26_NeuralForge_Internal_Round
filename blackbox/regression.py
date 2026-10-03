"""Turn a confirmed failure into a pytest regression test (offline, no API calls).

The generated test re-creates the task on the offline sandbox agent, re-injects the
recorded bad output at the guilty step, and asserts that (1) the run fails, (2) Black
Box blames that step, and (3) the fix that confirmed the diagnosis makes it pass.
"""
from __future__ import annotations

import json
import pprint
import re

TEMPLATE = '''"""Regression test exported by Black Box from run {run_id}.

Task: {prompt}
Root cause: step {step} ({node}). Recorded on {model}; replayed here on the offline sandbox agent.
"""
from blackbox.diagnosis import diagnose
from blackbox.engine import Engine
from blackbox.llm import SandboxLLM

PROMPT = {prompt_literal}
FAMILY = {family!r}
PARAMS = {params}
GUILTY_STEP = {step}
BAD_OUTPUT = {bad}
FIXED_OUTPUT = {fix}


def _baseline():
    engine = Engine(":memory:", llm=SandboxLLM(0.0))
    run = engine.run(PROMPT, FAMILY, dict(PARAMS))
    assert run["success"], "the clean task must pass before the regression is reproduced"
    return engine, run


def test_{name}_fails_and_is_localized():
    engine, run = _baseline()
    replay = engine.replay(run["run_id"], GUILTY_STEP, {{"output": BAD_OUTPUT}}, k=1)
    broken = engine.store.get_run(replay["run_id"])
    assert broken["success"] is False
    assert diagnose(broken)["root_cause"]["step"] == GUILTY_STEP


def test_{name}_fix_restores_the_answer():
    engine, run = _baseline()
    broken = engine.store.get_run(engine.replay(run["run_id"], GUILTY_STEP, {{"output": BAD_OUTPUT}}, k=1)["run_id"])
    fixed = engine.replay(broken["run_id"], GUILTY_STEP, {{"output": FIXED_OUTPUT}}, k=1)
    assert fixed["passed"] == 1
'''


def _fix_for(engine, run: dict, step: int) -> dict:
    replays = [r for r in engine.store.list_runs(limit=5000)
               if r.get("parent_run_id") == run["run_id"] and r.get("replay") and r.get("success")
               and r["replay"].get("from_step") == step and "output" in r["replay"].get("patch", {})]
    if replays:
        return replays[0]["replay"]["patch"]["output"]
    options = [o for o in engine.suggest_fix(run["run_id"], step)["options"] if o.get("kind") != "output_substitution"]
    if not options:
        raise ValueError("No fix is available for this step; confirm one with a replay first")
    return options[0]["patch"]["output"]


def export_test(engine, run: dict, step: int) -> dict:
    if run.get("success") is not False:
        raise ValueError("Only failed runs can be exported as regression tests")
    if not isinstance(step, int) or not 1 <= step <= len(run["steps"]):
        raise ValueError("step must be a recorded step")
    fix = _fix_for(engine, run, step)
    params = {**run["params"], "frozen_at": run["frozen_at"]}
    name = re.sub(r"[^a-z0-9]+", "_", f"{run['task_family']}_{run['steps'][step - 1]['node_name']}_{run['run_id'][-6:]}".lower())
    source = TEMPLATE.format(run_id=run["run_id"], prompt=run["prompt"].replace('"""', "'''"),
                             prompt_literal=json.dumps(run["prompt"], ensure_ascii=False),
                             family=run["task_family"], params=pprint.pformat(params, width=100),
                             step=step, node=run["steps"][step - 1]["node_name"], model=run.get("model"),
                             bad=pprint.pformat(run["steps"][step - 1]["output"], width=100),
                             fix=pprint.pformat(fix, width=100), name=name)
    return {"filename": f"test_regression_{name}.py", "source": source}
