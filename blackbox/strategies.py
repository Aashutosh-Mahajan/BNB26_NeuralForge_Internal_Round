"""Repair strategy catalogue: candidate alternatives for a suspected step.

Each candidate states what changes, why, its preconditions and its kind:

- ``tool_execution``     the tool is actually called again (now, at the run's frozen time)
- ``llm_rerun``          the LLM step is executed again with a stricter instruction
- ``argument_repair``    tool arguments are rebuilt deterministically from the plan
- ``output_substitution`` a recorded output from another run is reused (not a real repair)
- ``ask_user``           no safe automatic repair; escalate

Building a candidate never marks it successful; only a tested branch can pass.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime

from .agent.graph import LLM_NODES, NodeContext, execute
from .agent.tools import backup_currency_rate, currency_rate, doc_search, parse_time
from .storage import canonical
from .verifier import FRESHNESS_DAYS

STRICT_INSTRUCTIONS = {
    "planner": "Copy every number, unit and constraint exactly as written in the task; convert 'k' and 'lakh' to plain numbers.",
    "router": "Pass every plan constraint through unchanged in arguments.",
    "reasoner": "Report the calculator's value exactly; do not recompute or round.",
    "final_answer": "Return only the reasoner's answer value, unchanged.",
}


def _pre(name, ok, detail, hard=True):
    """hard: the candidate cannot run at all; soft: a policy concern that testing can confirm."""
    return {"name": name, "ok": bool(ok), "detail": detail, "hard": hard}


def _card(sid, label, kind, rationale, preconditions, patch=None, provenance=None, recompute=None):
    hard_ok = all(p["ok"] for p in preconditions if p.get("hard", True))
    soft_ok = all(p["ok"] for p in preconditions if not p.get("hard", True))
    executable = kind != "ask_user"
    status = ("manual" if not executable else "unavailable" if not hard_ok else
              "suggested" if soft_ok else "needs_review")
    return {"id": sid, "label": label, "kind": kind, "rationale": rationale, "preconditions": preconditions,
            "patch": patch if hard_ok and executable else None, "provenance": provenance or {},
            "status": status,
            "unavailable_reason": "; ".join(p["detail"] for p in preconditions if not p["ok"]) or None,
            "expected_recompute": recompute}


def _descendants(run, step_id):
    found, frontier = [], {step_id}
    for s in run["steps"]:
        if any(p in frontier for p in s.get("parent_step_ids", [])):
            frontier.add(s["step_id"])
            found.append(s["step_id"])
    return found


def _args(step):
    route = (step.get("input") or {}).get("dependencies", {}).get("router") if isinstance(step.get("input"), dict) else None
    args = route.get("arguments") if isinstance(route, dict) else None
    return args if isinstance(args, dict) else {}


def _rerun(engine, run, step, instruction=None):
    llm = engine._llm_for(run)
    ctx = NodeContext(llm=llm, seed=run.get("seed", 7), variant=131, run_id=run["run_id"], purpose="alternative",
                      overrides={"prompt": instruction} if instruction else {})
    output, _ = execute(step["step_id"], run["task_family"], step["input"], ctx)
    return output, llm


def candidates(engine, run: dict, step_id: int, successful_runs: list[dict] | None = None,
               allow_billed: bool = False) -> list[dict]:
    step = run["steps"][step_id - 1]
    name, kind = step["node_name"], step["node_type"]
    recompute = _descendants(run, step_id)
    out = step["output"] if isinstance(step.get("output"), dict) else {}
    cards = []
    llm = engine._llm_for(run)
    billed = llm.provider == "openai"
    budget_ok = _pre("No billed API calls", allow_billed or not billed,
                     f"Re-running needs {llm.label}, which bills API credits; allow billed experiments to test it.")

    if name in LLM_NODES:
        if billed and not allow_billed:
            changed, used = True, llm  # Not probed: probing would itself be a billed call.
        else:
            try:
                output, used = _rerun(engine, run, step, STRICT_INSTRUCTIONS.get(name))
                changed = canonical(output) != canonical(step["output"])
            except Exception:
                changed, used = False, llm
        cards.append(_card(
            "llm_rerun", f"Re-run {name} with a stricter instruction", "llm_rerun",
            f"The {name} output disagrees with its inputs; a careful re-run may correct it.",
            [budget_ok, _pre("Re-run gives a different output", changed, "The re-run produced the same output.")],
            {"prompt": STRICT_INSTRUCTIONS.get(name)} if name in STRICT_INSTRUCTIONS else None,
            {"model": used.label}, recompute))
        if name == "router":
            plan = (step["input"].get("dependencies", {}).get("planner") or {}).get("constraints")
            repaired = deepcopy(out)
            repaired["arguments"] = deepcopy(plan) if isinstance(plan, dict) else None
            repaired.setdefault("tool", "")
            from .agent.llm_nodes import PRIMARY_TOOL
            repaired["tool"] = PRIMARY_TOOL.get(run["task_family"], repaired["tool"])
            repaired["attempts"], repaired["step_budget"] = 1, out.get("step_budget", 3)
            cards.append(_card(
                "repair_arguments", "Rebuild the tool call from the plan", "argument_repair",
                "The router's tool or arguments do not match the plan; rebuild them deterministically.",
                [_pre("Plan is available", isinstance(plan, dict), "The planner produced no constraints."),
                 _pre("Repair changes the call", canonical(repaired) != canonical(out), "The call already matches the plan.")],
                {"output": repaired}, {"source": "planner constraints"}, recompute))
    else:
        try:
            retry, _ = _rerun(engine, run, step)
            same = canonical(retry) == canonical(step["output"])
        except Exception as exc:
            retry, same = {"error": str(exc)}, True
        cards.append(_card(
            "retry_tool", f"Retry {name}", "tool_execution",
            "A transient or bad response may not repeat when the tool is called again.",
            [_pre("Retry budget left", (step.get("retries") or 0) < 3, "Retry budget of 3 is used up."),
             _pre("Retry returns a different result", not same, "The tool returned the same result again.")],
            {"output": retry}, {"executed": "tool called again at the run's frozen time"}, recompute))

    if name == "currency_rate":
        args = _args(step)
        base, target = args.get("base_currency", "INR"), args.get("target_currency", "USD")
        try:
            backup = backup_currency_rate(base, target, run["frozen_at"])
            ok = True
        except (KeyError, ValueError):
            backup, ok = None, False
        cards.append(_card(
            "backup_source", "Use the backup FX provider", "tool_execution",
            "The primary quote may be stale or wrong; an independent provider gives a second quote.",
            [_pre("Backup provider supports this pair", ok, f"No backup quote for {base}/{target}."),
             _pre("Backup quote differs", ok and canonical(backup) != canonical(out), "Backup returned the same quote.")],
            {"output": backup}, {"executed": "sandbox-backup-feed (mocked second provider)"}, recompute))
        # A real past observation with its real timestamp, never relabelled as fresh.
        population = successful_runs or []
        frozen = parse_time(run["frozen_at"])
        observations = []
        for other in population:
            if other.get("task_id") == run.get("task_id") or other.get("task_family") != "finance":
                continue
            fx = next((s for s in other.get("steps", []) if s.get("node_name") == "currency_rate"), None)
            o = fx.get("output") if fx else None
            if not isinstance(o, dict) or not o.get("as_of"):
                continue
            if (other.get("params", {}).get("base_currency"), other.get("params", {}).get("target_currency")) != (base, target):
                continue
            seen = parse_time(o["as_of"])
            if seen <= frozen:
                observations.append((seen, o, other["run_id"]))
        if observations:
            seen, cached, source_run = max(observations, key=lambda x: x[0])
            age = (frozen - seen).total_seconds() / 86400
            cards.append(_card(
                "recent_cached_quote", "Use the most recent recorded quote", "output_substitution",
                "If no live provider works, a recent recorded quote may be acceptable as an approximation.",
                [_pre(f"Quote is at most {FRESHNESS_DAYS:g} day old", age <= FRESHNESS_DAYS,
                      f"The newest recorded quote is {age:.1f} days old; the task needs a fresh quote.", hard=False),
                 _pre("Quote differs from the suspect output", canonical(cached) != canonical(out), "Same quote.")],
                {"output": deepcopy(cached)}, {"source_run": source_run, "quote_time": cached["as_of"],
                                              "age_days": round(age, 2), "timestamp_kept": True}, recompute))

    if name == "doc_search":
        args = _args(step)
        try:
            fresh = doc_search(args.get("policy", ""), args.get("category", ""), run["frozen_at"])
            current = [d for d in fresh["documents"] if d.get("version") == "current"]
            fresh = {**fresh, "documents": current, "scores": fresh["scores"][:len(current)]}
            ok = bool(current)
        except (KeyError, ValueError):
            fresh, ok = None, False
        cards.append(_card(
            "current_documents_only", "Retrieve again, current policies only", "tool_execution",
            "An archived document outranked the current policy; filter retrieval to current versions.",
            [_pre("A current document exists", ok, "No current document matches."),
             _pre("Result differs", ok and canonical(fresh) != canonical(out), "Same documents.")],
            {"output": fresh}, {"executed": "doc_search with version=current filter"}, recompute))

    if name == "memory":
        source = next(iter((step["input"].get("dependencies") or {}).values()), None)
        restored = {"value": source.get("value"), "source_step": 5} if isinstance(source, dict) else None
        cards.append(_card(
            "restore_memory", "Restore memory from its source step", "argument_repair",
            "Memory disagrees with the value it was asked to store.",
            [_pre("Source value available", bool(restored and restored["value"] is not None), "No source value."),
             _pre("Restoring changes memory", restored is not None and canonical(restored) != canonical(out), "Already equal.")],
            {"output": restored}, {"source": "dependency value"}, recompute))

    match = next((r for r in (successful_runs or []) if r.get("task_id") == run.get("task_id")
                  and r.get("run_id") != run["run_id"] and len(r.get("steps", [])) >= step_id), None)
    if match:
        recorded = match["steps"][step_id - 1]["output"]
        cards.append(_card(
            "matching_success", "Reuse the output of a successful run of this task", "output_substitution",
            "Diagnostic only: shows whether this step alone explains the failure. Not a real repair.",
            [_pre("Output differs", canonical(recorded) != canonical(step["output"]), "Same output.")],
            {"output": deepcopy(recorded)}, {"source_run": match["run_id"]}, recompute))

    cards.append(_card("ask_user", "Ask the user or return an informative failure", "ask_user",
                       "When no compliant repair exists, never invent one: report what is missing.", [], None,
                       {}, recompute))
    return cards
