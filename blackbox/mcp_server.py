"""MCP server: lets Claude, Cursor or any MCP client ask Black Box why a run failed.

  python -m blackbox.mcp_server            # stdio transport, talks to the running API

It wraps the existing HTTP API (BLACKBOX_URL, default http://127.0.0.1:8010), so the
dashboard server must be running. Read-only except `test_fix`, which records replay
forks (the original run is never changed). Nothing here calls OpenAI.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server.mcpserver import MCPServer

BASE = os.environ.get("BLACKBOX_URL", "http://127.0.0.1:8010").rstrip("/") + "/api"
server = MCPServer(
    name="blackbox",
    title="Black Box agent flight recorder",
    instructions="Diagnose failed AI-agent runs: list failures, get the guilty step with evidence, "
                 "explain it in plain English, and test a fix by partial replay.",
)


def _call(path: str, body: dict | None = None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = json.loads(error.read() or b"{}").get("detail", str(error))
        raise ValueError(f"Black Box API error {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Black Box API is not reachable at {BASE}; start the server first") from error


@server.tool(description="List recent runs that failed, with the step Black Box blames for each.")
def list_failed_runs(limit: int = 10) -> list[dict]:
    runs = _call(f"/runs?status=FAILED&limit={int(limit)}")["runs"]
    return [{"run_id": r["run_id"], "task": r.get("prompt"), "model": r.get("model"),
             "root_cause_step": (r.get("root_cause") or {}).get("step"),
             "root_cause_node": (r.get("root_cause") or {}).get("node"),
             "blame": (r.get("root_cause") or {}).get("confidence"), "p_fail": r.get("p_fail")} for r in runs]


@server.tool(description="Why did this run fail? Returns the guilty step, confidence, evidence and affected steps.")
def diagnose_run(run_id: str) -> dict:
    run = _call(f"/runs/{urllib.parse.quote(run_id)}")
    evidence = run.get("evidence") or {}
    return {"run_id": run_id, "status": run.get("status"), "task": run.get("prompt"),
            "final_answer": run.get("final_answer"), "p_fail": run.get("p_fail"),
            "root_cause": run.get("root_cause"), "summary": evidence.get("summary"),
            "reasons": [f["description"] for f in evidence.get("factors", [])[:5]],
            "affected_steps": evidence.get("data_flow", [])[1:],
            "model_votes": evidence.get("model_votes"), "replay_check": evidence.get("counterfactual"),
            "guilty_step_output": next((s["output"] for s in run.get("steps", [])
                                        if s["step_id"] == (run.get("root_cause") or {}).get("step")), None)}


@server.tool(description="Plain-English explanation of a failed run (template explainer, no API cost).")
def explain_run(run_id: str) -> str:
    return _call(f"/runs/{urllib.parse.quote(run_id)}/explain?mode=template")["text"]


@server.tool(description="Test the diagnosis: apply the first suggested fix at a step and replay K times.")
def test_fix(run_id: str, step: int | None = None, k: int = 3) -> dict:
    if step is None:
        step = (diagnose_run(run_id).get("root_cause") or {}).get("step")
    options = _call(f"/runs/{urllib.parse.quote(run_id)}/suggest-fix?step={int(step)}")["options"]
    if not options:
        raise ValueError(f"No automatic fix is available for step {step}")
    result = _call(f"/runs/{urllib.parse.quote(run_id)}/replay",
                   {"from_step": int(step), "patch": {"output": options[0]["patch"]["output"]}, "k": int(k)})
    return {"fix": options[0]["label"], "passed": result["passed"], "k": result["k"], "verdict": result["verdict"],
            "steps_reused": result["reused_count"], "steps_rerun": result["rerun_steps"],
            "replay_run_id": result["run_id"]}


@server.tool(description="Overall stats: runs, failures and which components cause most failures.")
def failure_stats() -> dict:
    stats = _call("/stats")
    return {"totals": stats["totals"], "failures_by_component": stats["by_component"],
            "diagnosis_latency_ms": stats.get("diagnosis_latency_ms"), "method": stats.get("method")}


if __name__ == "__main__":
    server.run("stdio")
