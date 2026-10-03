"""Bounded live recovery: validate each step as it finishes and repair before dependants run.

Checks use only information available at that moment (the step's inputs, output and
earlier steps), never later steps. They are rule-based validators; a learned prefix
detector is not trained. Repairs reuse the strategy catalogue but never substitute a
recorded output, and stop after ``max_attempts``; otherwise the step is escalated.
"""
from __future__ import annotations

from datetime import datetime, timezone

from .verifier import FRESHNESS_DAYS, SOURCE_TOLERANCE


def _time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def prefix_checks(run: dict, step: dict) -> list[str]:
    """Violations for one finished step, using only data available now."""
    out = step.get("output")
    name = step.get("node_name")
    deps = (step.get("input") or {}).get("dependencies", {}) if isinstance(step.get("input"), dict) else {}
    problems = []
    if not isinstance(out, dict) or out == {}:
        return ["empty output"]
    if out.get("error"):
        problems.append("tool error: " + str(out["error"])[:80])
    if name == "currency_rate":
        quoted, frozen = _time(out.get("as_of")), _time(run.get("frozen_at"))
        if quoted and frozen and (frozen - quoted).total_seconds() / 86400 > FRESHNESS_DAYS:
            problems.append("stale FX quote")
        rate, backup = out.get("rate"), out.get("backup_rate")
        if isinstance(rate, (int, float)) and isinstance(backup, (int, float)) and backup and abs(rate - backup) / backup > SOURCE_TOLERANCE:
            problems.append("FX sources disagree")
    if name == "doc_search":
        listed = out.get("documents")
        top = listed[0] if isinstance(listed, list) and listed else None
        if isinstance(top, dict) and top.get("version") not in (None, "current"):
            problems.append("archived document ranked first")
    if name in ("formula_search", "policy_reference"):
        if out.get("version") not in (None, "current") or out.get("factor") not in (None, 1, 1.0):
            problems.append("archived or altered reference")
    if name == "router":
        from .agent.llm_nodes import PRIMARY_TOOL
        plan = (deps.get("planner") or {}).get("constraints") if isinstance(deps.get("planner"), dict) else None
        args = out.get("arguments") or {}
        if isinstance(plan, dict) and any(args.get(k) != v for k, v in plan.items()):
            problems.append("arguments differ from the plan")
        expected = PRIMARY_TOOL.get(run.get("task_family"))
        if expected and out.get("tool") != expected:
            problems.append(f"routes to {out.get('tool')!r}, which cannot serve this task")
        if isinstance(out.get("attempts"), int) and isinstance(out.get("step_budget"), int) and out["attempts"] > out["step_budget"]:
            problems.append("attempt budget exceeded (loop)")
    if name == "planner" and isinstance(run.get("prompt"), str):
        from .agent.tasks import DEFAULTS, PARSERS
        family = run.get("task_family")
        if family in PARSERS:
            expected = {**DEFAULTS[family], **PARSERS[family](run["prompt"]), **(run.get("context") or {})}
            constraints = out.get("constraints") or {}
            if any(constraints.get(k) != expected[k] for k in DEFAULTS[family]):
                problems.append("plan disagrees with the task text")
    if name in ("reasoner", "final_answer", "memory") and len(deps) == 1:
        source = next(iter(deps.values()))
        if isinstance(source, dict) and not source.get("error"):
            want = source.get("value", source.get("answer"))
            got = out.get("value", out.get("answer"))
            if want is not None and got != want:
                problems.append("output differs from its input value")
    return problems


def recover(engine, run: dict, step: dict, llm, max_attempts: int = 2):
    """Try catalogue repairs (no recorded-output substitution). Returns (output, record)."""
    from .agent.graph import NodeContext, execute
    from .strategies import candidates
    record = {"violations": prefix_checks(run, step), "attempts": [], "recovered": False}
    if not record["violations"]:
        return None, None
    probe = {**run, "steps": run["steps"] + [step]}
    options = [c for c in candidates(engine, probe, step["step_id"], [], allow_billed=llm.provider != "openai")
               if c["status"] == "suggested" and c["kind"] != "output_substitution" and c["patch"]]
    for card in options[:max_attempts]:
        patch = card["patch"]
        if "output" in patch:
            output = patch["output"]
        else:
            ctx = NodeContext(llm=llm, seed=run.get("seed", 7), variant=211 + len(record["attempts"]),
                              run_id=run["run_id"], purpose="live_recovery", overrides=patch)
            try:
                output, _ = execute(step["step_id"], run["task_family"], step["input"], ctx)
            except Exception as exc:
                output = {"error": str(exc)}
        remaining = prefix_checks(run, {**step, "output": output})
        record["attempts"].append({"strategy": card["id"], "kind": card["kind"], "remaining_violations": remaining})
        if not remaining:
            record["recovered"] = True
            record["strategy"] = card["id"]
            return output, record
    record["escalated"] = True
    return None, record
