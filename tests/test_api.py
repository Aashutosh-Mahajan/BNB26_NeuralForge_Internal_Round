from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient
from blackbox.api.app import create_app


@pytest.fixture
def client(tmp_path):
    app = create_app(tmp_path / "traces.db", seed_demo=False, metrics_path=tmp_path / "metrics.json")
    with TestClient(app) as connection:
        yield connection


def start_finance(client):
    response = client.post("/api/runs", json={"prompt": "Convert INR 50,000 to USD and compute EMI for 12 months at 9%.", "task_family": "finance"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    messages = []
    with client.websocket_connect(f"/api/runs/{run_id}/stream") as websocket:
        while True:
            event = websocket.receive_json()
            messages.append(event)
            if event["type"] in {"complete", "error"}:
                break
    assert messages[-1]["type"] == "complete"
    return run_id, messages


def test_run_stream_records_every_step_and_late_subscriber_gets_backlog(client):
    run_id, messages = start_finance(client)
    run = client.get(f"/api/runs/{run_id}").json()
    assert run["status"] == "PASSED"
    expected = [step["step_id"] for step in run["steps"]]
    assert [event["step_id"] for event in messages if event["type"] == "step"] == expected
    assert len({event["event_id"] for event in messages}) == len(messages)
    late = []
    with client.websocket_connect(f"/runs/{run_id}/stream") as websocket:
        while True:
            event = websocket.receive_json()
            late.append(event)
            if event["type"] == "complete":
                break
    assert late == messages
    # Resume cursors deliver exactly the remaining durable events.
    with client.websocket_connect(f"/runs/{run_id}/stream?after={messages[1]['event_id']}") as websocket:
        assert websocket.receive_json() == messages[2]
    assert client.get("/api/runs").json()["total"] == 1


def test_inject_repair_compare_preserves_original(client):
    run_id, _ = start_finance(client)
    original = client.get(f"/runs/{run_id}").json()
    failure = client.post(f"/runs/{run_id}/inject", json={"step_id": 3, "fault_type": "stale_data"})
    assert failure.status_code == 200, failure.text
    broken = failure.json()
    assert broken["success"] is False
    assert broken["root_cause"]["step"] == 3
    failed_id = broken["new_run_id"]
    fixes = client.get(f"/runs/{failed_id}/suggest-fix?step=3")
    assert fixes.status_code == 200
    assert fixes.json()["options"]
    # The independently recorded original output is an explicit user intervention.
    patch = {"output": original["steps"][2]["output"]}
    replay = client.post(f"/api/runs/{failed_id}/replay", json={"from_step": 3, "patch": patch, "k": 3})
    assert replay.status_code == 200, replay.text
    result = replay.json()
    assert result["passed"] == 3
    assert result["k"] == 3
    assert result["tokens_saved_pct"] is not None
    assert result["patch_type"] == "output"
    assert result["reused_steps"] or result["checkpoint_steps"]
    comparison = client.get("/api/compare", params={"a": failed_id, "b": result["new_run_id"]}).json()
    assert comparison["first_divergence"] == 3
    assert comparison["outcome"]["before"] is False
    assert comparison["outcome"]["after"] is True
    # Replays cannot overwrite the run being investigated.
    assert client.get(f"/runs/{failed_id}").json()["success"] is False
    assert client.get(f"/runs/{run_id}").json()["steps"] == original["steps"]


def test_missing_ids_and_invalid_inputs_are_client_errors(client):
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get("/compare", params={"a": "missing", "b": "missing"}).status_code == 404
    assert client.post("/runs/missing/replay", json={"from_step": 3, "patch": {"output": {}}, "k": 1}).status_code == 404
    assert client.post("/runs", json={"prompt": "   ", "task_family": "finance"}).status_code == 422
    assert client.post("/runs", json={"prompt": "test", "task_family": "unknown"}).status_code == 422
    run_id, _ = start_finance(client)
    for payload in (
        {"from_step": 3, "patch": {}, "k": 1},
        {"from_step": 3, "patch": {"output": {}, "model": "anything"}, "k": 1},
        {"from_step": 3, "patch": {"output": {}}, "k": 0},
        {"from_step": 999, "patch": {"output": {}}, "k": 1},
        {"from_step": 3, "patch": [], "k": 1},
    ):
        response = client.post(f"/runs/{run_id}/replay", json=payload)
        assert response.status_code == 422, response.text
    assert client.post(f"/runs/{run_id}/inject", json={"step_id": 3, "fault_type": "unknown"}).status_code == 422
    assert client.get(f"/runs/{run_id}/suggest-fix?step=999").status_code == 422
    with client.websocket_connect("/runs/missing/stream") as websocket:
        assert websocket.receive_json()["type"] == "error"


def test_metrics_are_absent_until_an_artifact_exists(client):
    metrics = client.get("/eval").json()
    assert metrics["status"] == "not_available"
    assert metrics["top1"] is None
    assert metrics["auroc"] is None
    assert client.get("/stats").json()["totals"] == {"runs": 0, "passed": 0, "failed": 0, "running": 0}
    assert len(client.get("/faults").json()["faults"]) == 12


def test_metrics_only_serve_saved_measurements(tmp_path):
    artifact = tmp_path / "metrics.json"
    artifact.write_text(json.dumps({"top1": 0.25, "top3": 0.5, "method": "test_fixture"}), encoding="utf-8")
    with TestClient(create_app(tmp_path / "traces.db", seed_demo=False, metrics_path=artifact)) as client:
        assert client.get("/api/eval").json()["top1"] == 0.25
        artifact.write_text("broken json", encoding="utf-8")
        assert client.get("/api/eval").json()["status"] == "not_available"


def test_seed_only_runs_on_empty_database(tmp_path):
    database = tmp_path / "traces.db"
    with TestClient(create_app(database, seed_demo=True)) as client:
        runs = client.get("/runs").json()["runs"]
        assert len(runs) == 5
        assert {run["task_family"] for run in runs} == {"finance", "sql", "doc_qa", "math"}
        assert sum(run["status"] == "FAILED" for run in runs) == 1
    with TestClient(create_app(database, seed_demo=True)) as client:
        assert client.get("/runs").json()["total"] == 5



def test_invalid_task_parameters_do_not_queue_a_run(client):
    response = client.post("/runs", json={"prompt": "Compute my EMI", "task_family": "finance", "params": {"months": 0}})
    assert response.status_code == 422
    response = client.post("/runs", json={"prompt": "Compute my EMI", "task_family": "finance", "params": {"frozen_at": "invalid"}})
    assert response.status_code == 422
    assert client.get("/runs").json()["total"] == 0


def test_execution_error_is_persisted_and_terminates_stream(client, monkeypatch):
    def failure(*args, **kwargs):
        raise RuntimeError("synthetic execution failure")

    monkeypatch.setattr(client.app.state.engine, "run", failure)
    response = client.post("/runs", json={"prompt": "Compute an EMI", "task_family": "finance"})
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    with client.websocket_connect(f"/runs/{run_id}/stream") as websocket:
        event = websocket.receive_json()
        assert event["type"] == "error"
        assert event["status"] == "ERROR"
    assert client.get(f"/runs/{run_id}").json()["status"] == "ERROR"


def test_restart_marks_interrupted_runs_and_preserves_an_error_event(tmp_path):
    from blackbox.storage import Store

    database = tmp_path / "traces.db"
    store = Store(database)
    store.save_run({"run_id": "interrupted", "created_at": "2026-10-03T00:00:00+00:00", "status": "RUNNING", "steps": []})
    store.close()
    with TestClient(create_app(database, seed_demo=False)) as client:
        assert client.get("/runs/interrupted").json()["status"] == "ERROR"
        with client.websocket_connect("/runs/interrupted/stream") as websocket:
            event = websocket.receive_json()
            assert event["type"] == "error"
            assert "restart" in event["message"]


def test_api_documentation_works_through_the_vite_prefix(client):
    response = client.get("/api/docs")
    assert response.status_code == 200
    assert "/api/openapi.json" in response.text
    schema = client.get("/api/openapi.json").json()
    assert schema["servers"] == [{"url": "/api"}]
    assert "/runs" in schema["paths"]


def test_llm_status_models_spans_and_explanation(client):
    run_id, _ = start_finance(client)
    status = client.get("/api/llm/status").json()
    from blackbox.config import settings
    assert status["openai_model"] == settings().openai_model
    assert status["pricing_per_million"] == {"input": 0.10, "cached_input": 0.01, "output": 0.50}
    assert "available" in client.get("/api/models/status").json()
    spans = client.get(f"/api/runs/{run_id}/spans").json()["spans"]
    assert len(spans) == 11
    failure = client.post(f"/runs/{run_id}/inject", json={"step_id": 3, "fault_type": "stale_data"}).json()
    explanation = client.get(f"/api/runs/{failure['new_run_id']}/explain?mode=template").json()
    assert "Step 3" in explanation["text"]
    stats = client.get("/api/stats").json()
    assert stats["suspiciousness"] and "cost_usd" in stats


def test_prompt_patch_rejected_for_tool_steps(client):
    run_id, _ = start_finance(client)
    response = client.post(f"/runs/{run_id}/replay", json={"from_step": 3, "patch": {"prompt": "x"}, "k": 1})
    assert response.status_code == 422
    response = client.post(f"/runs/{run_id}/replay", json={"from_step": 9, "patch": {"temperature": 0.4}, "k": 1})
    assert response.status_code == 200, response.text
