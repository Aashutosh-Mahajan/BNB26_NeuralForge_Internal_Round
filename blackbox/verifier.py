"""Independent acceptance checks for a run: pass / fail / unknown per check.

These are outcome checks, separate from diagnosis (they are never model inputs).
`answer` uses the expected value computed independently by the task's own verifier;
it is "unknown" for runs without one (e.g. external agents).
"""
from __future__ import annotations

from datetime import datetime, timezone

FRESHNESS_DAYS = 1.0      # a "today's rate" task accepts quotes at most one day old
SOURCE_TOLERANCE = 0.02   # primary vs backup FX source may differ by at most 2%


def _time(value):
    try:
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _step(run, name):
    return next((s for s in run.get("steps", []) if s.get("node_name") == name), None)


def _check(cid, label, status, detail):
    return {"id": cid, "label": label, "status": status, "detail": detail}


def acceptance_checks(run: dict) -> dict:
    checks = []
    family = run.get("task_family")
    gold = run.get("gold_answer")
    if gold is None:
        checks.append(_check("answer", "Answer matches the expected value", "unknown",
                             "No independent expected value exists for this run."))
    else:
        checks.append(_check("answer", "Answer matches the expected value", "pass" if run.get("success") else "fail",
                             f"Got {run.get('final_answer')!r}, expected {gold!r} (computed independently)."))
    errors = [s["step_id"] for s in run.get("steps", []) if s.get("tool_error")]
    checks.append(_check("no_errors", "No step returned an error", "fail" if errors else "pass",
                         f"Errors at steps {errors}." if errors else "Every step returned a result."))
    if family == "finance":
        fx = _step(run, "currency_rate")
        out = fx.get("output") if fx and isinstance(fx.get("output"), dict) else {}
        out = out if isinstance(out, dict) else {}
        quoted, frozen = _time(out.get("as_of")), _time(run.get("frozen_at"))
        if quoted and frozen:
            age = (frozen - quoted).total_seconds() / 86400
            checks.append(_check("fresh_quote", f"FX quote is at most {FRESHNESS_DAYS:g} day old",
                                 "pass" if abs(age) <= FRESHNESS_DAYS else "fail",
                                 f"Quote time {out.get('as_of')}, {age:.1f} days before the run."))
        else:
            checks.append(_check("fresh_quote", f"FX quote is at most {FRESHNESS_DAYS:g} day old", "unknown",
                                 "The quote has no timestamp."))
        rate, backup = out.get("rate"), out.get("backup_rate")
        if isinstance(rate, (int, float)) and isinstance(backup, (int, float)) and backup:
            gap = abs(rate - backup) / abs(backup)
            checks.append(_check("source_agreement", "Primary and backup FX sources agree (within 2%)",
                                 "pass" if gap <= SOURCE_TOLERANCE else "fail", f"Difference {gap:.1%}."))
    if family == "doc_qa":
        docs = (_step(run, "doc_search") or {}).get("output") or {}
        listed = docs.get("documents") if isinstance(docs, dict) else None
        top = listed[0] if isinstance(listed, list) and listed else None
        if isinstance(top, dict) and top.get("version"):
            checks.append(_check("current_policy", "The answer uses the current policy document",
                                 "pass" if top["version"] == "current" else "fail",
                                 f"Top document: {top.get('id')} ({top['version']})."))
    plan, route = _step(run, "planner"), _step(run, "router")
    if plan and route and isinstance(plan.get("output"), dict) and isinstance(route.get("output"), dict):
        wanted = plan["output"].get("constraints") or {}
        passed = route["output"].get("arguments") or {}
        missing = sorted(k for k in wanted if passed.get(k) != wanted[k])
        checks.append(_check("arguments_match_plan", "Tool arguments match the plan", "fail" if missing else "pass",
                             f"Differs on {missing}." if missing else "All planned values were passed on."))
    statuses = {c["status"] for c in checks}
    outcome = "failed" if "fail" in statuses else "unknown" if checks[0]["status"] == "unknown" else "passed"
    return {"outcome": outcome, "checks": checks, "verifier": "blackbox.verifier v1 (independent of diagnosis)"}
