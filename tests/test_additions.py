"""Tests for acceptance checks, alternatives, live recovery, the LangGraph adapter, and new API routes."""
from __future__ import annotations

import operator
from typing import Annotated, TypedDict

import pytest
from fastapi.testclient import TestClient

from blackbox.api.app import create_app
from blackbox.engine import Engine
from blackbox.llm import SandboxLLM
from blackbox.storage import Store, redact

PROMPT = "Convert INR 50,000 to USD and compute EMI for 12 months at 9%."
NOW = "2026-10-03T12:00:00+00:00"


@pytest.fixture
def engine():
    e = Engine(":memory:", llm=SandboxLLM(0.0))
    yield e
    e.store.close()


def test_acceptance_checks_flag_the_stale_quote(engine):
    clean = engine.run(PROMPT, params={"frozen_at": NOW})
    assert clean["acceptance"]["outcome"] == "passed"
    broken = engine.inject(clean["run_id"], 3, "stale_data")
    status = {c["id"]: c["status"] for c in broken["acceptance"]["checks"]}
    assert broken["acceptance"]["outcome"] == "failed"
    assert status["fresh_quote"] == "fail" and status["source_agreement"] == "fail" and status["answer"] == "fail"


def test_alternatives_pass_reject_and_keep_original(engine):
    engine.run("Convert INR 80,000 to USD and compute EMI for 24 months at 10%.", params={"frozen_at": "2026-09-19T12:00:00+00:00"})
    clean = engine.run(PROMPT, params={"frozen_at": NOW})
    broken = engine.inject(clean["run_id"], 3, "stale_data")
    snapshot = engine.store.get_run(broken["run_id"])
    cards = {c["id"]: c for c in engine.alternatives(broken["run_id"], 3)["candidates"]}
    assert cards["recent_cached_quote"]["status"] == "needs_review"          # 14-day-old quote: soft precondition
    assert cards["recent_cached_quote"]["provenance"]["timestamp_kept"] is True
    assert cards["ask_user"]["status"] == "manual"
    x = engine.test_alternatives(broken["run_id"], 3, ["retry_tool", "backup_source", "recent_cached_quote"], k=1)
    result = {b["id"]: b["status"] for b in x["branches"]}
    assert result == {"retry_tool": "passed", "backup_source": "passed", "recent_cached_quote": "rejected"}
    assert x["verdict"] == "supported" and x["totals"]["rejected"] == 1
    assert engine.store.get_run(broken["run_id"]) == snapshot
    assert engine.store.list_experiments(broken["run_id"])[0]["experiment_id"] == x["experiment_id"]


def test_live_recovery_repairs_before_dependants_and_does_not_interrupt_clean_runs(engine):
    clean = engine.run(PROMPT, params={"frozen_at": NOW}, live_recovery=True)
    assert clean["live_recovery"]["interruptions"] == 0
    without = engine.run(PROMPT, params={"frozen_at": NOW}, faults={3: "stale_data"})
    assert without["success"] is False
    repaired = engine.run(PROMPT, params={"frozen_at": NOW}, faults={3: "stale_data"}, live_recovery=True)
    assert repaired["success"] is True
    assert repaired["steps"][2]["recovery"]["recovered"] is True
    assert repaired["steps"][2]["planted_fault"] == "stale_data"


def test_replay_modes_and_retries(engine):
    clean = engine.run(PROMPT, params={"frozen_at": NOW})
    broken = engine.inject(clean["run_id"], 3, "stale_data")
    fix = engine.suggest_fix(broken["run_id"], 3)["options"][0]["patch"]
    recorded = engine.replay(broken["run_id"], 3, fix, k=1, mode="recorded")
    fresh = engine.replay(broken["run_id"], 3, fix, k=1, mode="fresh")
    assert recorded["reused_steps"] == [4, 7] and fresh["reused_steps"] == [] and fresh["cache_served_steps"] == []
    flaky = Engine(":memory:", llm=SandboxLLM(0.0, tool_noise=0.6))
    runs = [flaky.run("Buy 8 items at $12 each with a 10% discount.", "math", {"frozen_at": NOW}, seed=i) for i in range(20)]
    assert any(s["retries"] for r in runs for s in r["steps"])
    assert all(s["effect"] for s in runs[0]["steps"])


def test_redaction_masks_credentials():
    masked = redact({"api_key": "abc", "Authorization": "Bearer sk-abcdefghijklmnopqrstu",
                     "text": "token sk-proj-ABCDEFGHIJKLMNOPQRSTUVWX used", "tokens_in": 12})
    assert masked == {"api_key": "[REDACTED]", "Authorization": "[REDACTED]", "text": "token [REDACTED] used", "tokens_in": 12}


def test_external_langgraph_fork_and_resume_keep_independent_work():
    from langgraph.graph import END, START, StateGraph

    world = {"down": False, "stale": False}

    class S(TypedDict, total=False):
        x: int
        a: int
        b: int
        total: int
        log: Annotated[list, operator.add]

    def left(s):
        return {"a": 1 if world["stale"] else 10, "log": ["left"]}

    def right(s):
        if world["down"]:
            raise ConnectionError("503")
        return {"b": 5, "log": ["right"]}

    g = StateGraph(S)
    g.add_node("plan", lambda s: {"log": ["plan"]})
    g.add_node("left", left)
    g.add_node("right", right)
    g.add_node("join", lambda s: {"total": s["a"] + s["b"], "log": ["join"]})
    g.add_edge(START, "plan"); g.add_edge("plan", "left"); g.add_edge("plan", "right")
    g.add_edge(["left", "right"], "join"); g.add_edge("join", END)
    from blackbox import wrap
    agent = wrap(g.compile(), Store(":memory:"), name="toy", check=lambda s: s.get("total") == 15)
    world["stale"] = True
    bad = agent.invoke({"x": 1, "log": []})
    world["stale"] = False
    assert bad["status"] == "FAILED"
    assert [s["parent_step_ids"] for s in bad["steps"]] == [[], [1], [1], [2, 3]]
    forked = agent.fork(bad["run_id"], 2, {"a": 10})
    assert forked["status"] == "PASSED"
    assert [s["action"] for s in forked["steps"]] == ["checkpoint", "patched", "reused", "rerun"]
    world["down"] = True
    crashed = agent.invoke({"x": 1, "log": []})
    assert crashed["status"] == "ERROR" and crashed["last_checkpoint"]["checkpoint_id"]
    world["down"] = False
    resumed = agent.resume(crashed["run_id"])
    assert resumed["status"] == "PASSED"
    assert [s["node_name"] for s in resumed["steps"] if s["action"] == "rerun"] == ["right", "join"]


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(tmp_path / "traces.db", seed_demo=False, metrics_path=tmp_path / "m.json")) as c:
        yield c


def _finance(client):
    import time
    run_id = client.post("/api/runs", json={"prompt": PROMPT, "task_family": "finance", "provider": "sandbox"}).json()["run_id"]
    for _ in range(100):
        if client.get(f"/api/runs/{run_id}").json()["status"] in ("PASSED", "FAILED"):
            return run_id
        time.sleep(0.05)
    raise AssertionError("run did not finish")


def test_incident_api_experiments_report_review_memory(client):
    run_id = _finance(client)
    bad = client.post(f"/api/runs/{run_id}/inject", json={"step_id": 3, "fault_type": "stale_data"}).json()["new_run_id"]
    alternatives = client.get(f"/api/runs/{bad}/alternatives").json()
    assert alternatives["step_id"] == 3 and alternatives["candidates"]
    x = client.post(f"/api/runs/{bad}/experiments", json={"step": 3, "strategies": ["retry_tool", "backup_source"], "k": 1}).json()
    assert x["verdict"] == "supported"
    report = client.get(f"/api/runs/{bad}/report")
    assert report.status_code == 200 and "Leading suspect" in report.text and "Alternatives tested" in report.text
    review = client.post(f"/api/runs/{bad}/review", json={"verdict": "confirmed"}).json()
    assert review["label_step"] == 3 and review["queued_for_training"] is True
    assert client.post(f"/api/runs/{bad}/review", json={"verdict": "wrong_step"}).status_code == 422
    memory = client.get(f"/api/runs/{bad}/memory").json()
    assert memory["node"] == "currency_rate"
    run = client.get(f"/api/runs/{bad}").json()
    assert "abstain" in run["diagnosis"]


def test_claude_code_hooks_are_recorded_and_failures_diagnosable(client):
    base = {"session_id": "abc123", "transcript_path": "/tmp/t", "cwd": "/repo"}
    client.post("/api/ingest/claude-code", json={**base, "hook_event_name": "UserPromptSubmit", "prompt": "fix the tests"})
    client.post("/api/ingest/claude-code", json={**base, "hook_event_name": "PostToolUse", "tool_name": "Read",
                                                 "tool_input": {"file_path": "a.py"}, "tool_response": "ok"})
    result = client.post("/api/ingest/claude-code", json={**base, "hook_event_name": "PostToolUseFailure", "tool_name": "Bash",
                                                          "tool_input": {"command": "npm test"}, "error": "exit code 1",
                                                          "error_type": "execution_error"}).json()
    assert result == {"run_id": "CC-abc123", "steps": 2, "status": "FAILED"}
    run = client.get("/api/runs/CC-abc123").json()
    assert run["prompt"] == "fix the tests" and run["capabilities"]["fork"] is False
    assert run["steps"][1]["output"]["error_type"] == "execution_error"
