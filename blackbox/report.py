"""Self-contained incident report (Markdown) for a failed run."""
from __future__ import annotations

import json
from datetime import datetime, timezone

STATUS_WORD = {"pass": "pass", "fail": "FAIL", "unknown": "unknown"}


def _value(v):
    return json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v


def incident_report(run: dict, diagnosis: dict | None, experiments: list[dict]) -> str:
    acceptance = run.get("acceptance") or {}
    d = diagnosis or {}
    root = d.get("root_cause") or {}
    evidence = d.get("evidence") or {}
    lines = [f"# Incident report: run {run['run_id']}", "",
             f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by Black Box.*", "",
             "## What happened", "",
             f"- **Task:** {run.get('prompt')}",
             f"- **Agent model:** {run.get('model') or 'unknown'} ({run.get('llm_provider')})",
             f"- **Outcome:** {acceptance.get('outcome', run.get('status', '')).upper()}",
             f"- **Final answer:** {_value(run.get('final_answer'))}"
             + (f" (expected {_value(run.get('gold_answer'))})" if run.get("gold_answer") is not None else ""),
             "", "### Acceptance checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for c in acceptance.get("checks", []):
        lines.append(f"| {c['label']} | {STATUS_WORD.get(c['status'], c['status'])} | {c['detail']} |")
    lines += ["", f"Verifier: {acceptance.get('verifier', 'n/a')}", "", "## Diagnosis", ""]
    if root:
        lines += [f"- **Leading suspect:** step {root.get('step')} `{root.get('node')}` "
                  f"(ranking score {root.get('confidence', 0):.0%}; not a calibrated probability)",
                  f"- **Other suspects:** " + ", ".join(f"step {s['step']} `{s['node']}`" for s in d.get("top_suspects", [])[1:3]),
                  f"- **Method:** {d.get('method')} (model version {d.get('model_version', 'n/a')})",
                  f"- **Evidence:** {evidence.get('summary', '')}"]
        for f in evidence.get("factors", [])[:4]:
            lines.append(f"  - {f.get('description')}")
        if len(evidence.get("data_flow", [])) > 1:
            lines.append(f"- **Affected downstream steps:** {', '.join(map(str, evidence['data_flow'][1:]))}")
    else:
        lines.append("No failure was diagnosed.")
    lines += ["", "## Alternatives tested", ""]
    if not experiments:
        lines.append("No alternatives were tested. The diagnosis remains an untested hypothesis.")
    for x in experiments:
        lines += [f"Experiment {x['experiment_id']} at step {x['step_id']} ({x['created_at'][:19]}):", "",
                  "| Alternative | Type | Result | Answer | Failed checks | Re-run steps | Tokens | Cost |",
                  "|---|---|---|---|---|---|---|---|"]
        for b in x["branches"]:
            lines.append(f"| {b['label']} | {b['kind']} | {b['status']} | {_value(b.get('final_answer', '—'))} | "
                         f"{', '.join(b.get('failed_checks') or []) or ('—' if b.get('tested') else b.get('unavailable_reason') or '—')} | "
                         f"{', '.join(map(str, b.get('rerun_steps', []))) or '—'} | {b.get('billed_tokens', 0)} | "
                         f"${b.get('cost_usd', 0):.5f} |")
        t = x["totals"]
        lines += ["", f"**Verdict:** {x['verdict']}. {x['verdict_text']}",
                  f"Total experiment cost: {t['billed_tokens']} tokens, ${t['cost_usd']:.5f}, {t['wall_ms']:.0f} ms "
                  f"across {t['tested']} tested branches ({t['passed']} passed, {t['rejected']} rejected).", ""]
    lines += ["## Limitations", "",
              "- The diagnosis ranks observable steps; it cannot see the model's private reasoning.",
              "- A passing alternative supports the suspect as the origin; it does not prove a unique cause.",
              "- Output substitutions reuse recorded values and are diagnostic, not real repairs.",
              "- Offline-sandbox branches are deterministic; repeated variants are not independent evidence.",
              f"- Original run preserved: {run['run_id']} was never modified; every branch is a separate run."]
    return "\n".join(lines) + "\n"
