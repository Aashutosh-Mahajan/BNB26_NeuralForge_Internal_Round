"""FastAPI backend: runs, live stream, diagnosis, replay, comparison, evaluation and LLM status."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from blackbox.agent.tasks import parameters
from blackbox.agent.tools import parse_time
from blackbox.config import ROOT
from blackbox.diagnosis import diagnose_many
from blackbox.engine import Engine, FAULT_CATALOG
from blackbox.explain import compare_runs
from blackbox.llm import provider_status, UsageLedger
from blackbox.models.ensemble import ensemble_status, load_ensemble
from blackbox.models.spectrum import ochiai
from blackbox.recorder.otel import spans_for_run
from blackbox.storage import Store
from .events import EventLog

logger = logging.getLogger(__name__)


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRequest(RequestModel):
    prompt: str = Field(min_length=3, max_length=4000)
    task_family: Literal["finance", "sql", "doc_qa", "math"] = "finance"
    params: dict[str, Any] | None = None
    provider: Literal["default", "sandbox", "openai", "ollama"] = "default"
    live_recovery: bool = False
    plant_fault: dict[str, Any] | None = None  # {"step": 3, "fault": "stale_data"}: demo of a mid-run failure

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("Provide a task of at least three characters.")
        return value


class InjectionRequest(RequestModel):
    step_id: int = Field(ge=1, strict=True)
    fault_type: str = Field(min_length=1)


class OutputPatch(RequestModel):
    output: Any = None
    prompt: str | None = Field(default=None, max_length=2000)
    model: str | None = Field(default=None, max_length=100)
    temperature: float | None = Field(default=None, ge=0, le=2)

    @field_validator("output")
    @classmethod
    def finite_json(cls, value: Any) -> Any:
        try:
            json.dumps(value, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ValueError("Patch output must be finite JSON.") from error
        return value

    def as_patch(self) -> dict[str, Any]:
        return dict(self.model_dump(exclude_unset=True))


class ExperimentRequest(RequestModel):
    step: int = Field(ge=1, strict=True)
    strategies: list[str] | None = None
    k: int = Field(default=3, ge=1, le=10, strict=True)
    allow_billed: bool = False
    mode: Literal["recorded", "fresh"] = "recorded"


class ReviewRequest(RequestModel):
    verdict: Literal["confirmed", "wrong_step", "uncertain"]
    label_step: int | None = Field(default=None, ge=1)
    note: str | None = Field(default=None, max_length=1000)


class ReplayRequest(RequestModel):
    from_step: int = Field(ge=1, strict=True)
    patch: OutputPatch
    k: int = Field(default=5, ge=1, le=10, strict=True)
    mode: Literal["recorded", "fresh"] = "recorded"


def create_app(db_path: str | Path | None = None, *, seed_demo: bool | None = None,
               metrics_path: str | Path | None = None) -> FastAPI:
    """Create isolated applications for the server, scripts, or integration tests."""
    database = Path(db_path or os.environ.get("BLACKBOX_DB", "data/traces.db"))
    metrics_file = Path(metrics_path or os.environ.get("BLACKBOX_METRICS", "data/metrics.json"))
    should_seed = seed_demo if seed_demo is not None else os.environ.get("BLACKBOX_SEED_DEMO", "true").lower() == "true"
    background_tasks: set[asyncio.Task] = set()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        database.parent.mkdir(parents=True, exist_ok=True)
        application.state.store = Store(database)
        application.state.engine = Engine(application.state.store)
        application.state.engines = {}
        application.state.diagnoses = {}
        application.state.events = EventLog(database)
        # Interrupted processes leave explicit errors instead of streams that wait forever.
        for run in application.state.store.list_runs():
            if run.get("status") in {"QUEUED", "RUNNING"}:
                run.update(status="ERROR", success=False, error="Execution interrupted by a server restart.")
                application.state.store.save_run(run)
                application.state.events.append(run["run_id"], {"type": "error", "status": "ERROR", "message": run["error"]})
        await asyncio.to_thread(warm_models)
        if should_seed and not application.state.store.list_runs():
            await asyncio.to_thread(seed_examples)
        yield
        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)
        application.state.store.close()

    application = FastAPI(title="Black Box", version="0.2.0", lifespan=lifespan, servers=[{"url": "/api"}],
                          description="Flight recorder and crash investigator for AI agents.")
    origins = os.environ.get("BLACKBOX_CORS_ORIGINS", "http://localhost:5174,http://127.0.0.1:5174").split(",")
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])
    router = APIRouter()

    def get_run(run_id: str) -> dict[str, Any]:
        try:
            return application.state.store.get_run(run_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Run not found.") from error

    def successes() -> list[dict[str, Any]]:
        return [run for run in application.state.store.list_runs() if run.get("success") is True]

    def engine_for(provider: str | None):
        if not provider or provider == "default":
            return application.state.engine
        if provider not in application.state.engines:
            engine = Engine(application.state.store, llm=provider)
            if engine.llm.provider != provider:
                raise HTTPException(status_code=422, detail=f"Provider {provider} is unavailable: {engine.llm_warning}")
            application.state.engines[provider] = engine
        return application.state.engines[provider]

    def diagnoses_for(runs: list[dict[str, Any]], reference: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Completed runs are immutable, so a diagnosis is cached per model version."""
        model = load_ensemble()
        version = model.version if model else "heuristic"
        cache = application.state.diagnoses
        todo = [r for r in runs if (r["run_id"], version) not in cache]
        if todo:
            for run, result in zip(todo, diagnose_many(todo, successful_runs=reference)):
                cache[(run["run_id"], version)] = result
        return [cache[(r["run_id"], version)] for r in runs]

    def counterfactual(run: dict[str, Any]) -> dict[str, Any] | None:
        replays = [r for r in application.state.store.list_runs(limit=5000)
                   if r.get("parent_run_id") == run["run_id"] and r.get("replay")]
        if not replays:
            return None
        latest = replays[0]
        group = [r for r in replays if r["replay"].get("from_step") == latest["replay"].get("from_step")
                 and r["replay"].get("patch") == latest["replay"].get("patch")]
        passed = sum(bool(r.get("success")) for r in group)
        verdict = ("confirmed" if passed / len(group) >= 0.5 and not run.get("success")
                   else "rejected" if not passed else "inconclusive")
        return {"from_step": latest["replay"]["from_step"], "passed": passed, "k": len(group),
                "verdict": verdict, "replay_run_id": latest["run_id"]}

    def detailed(run: dict[str, Any]) -> dict[str, Any]:
        if run.get("status") not in {"PASSED", "FAILED"}:
            return {**run, "diagnosis": None}
        diagnosis = dict(diagnoses_for([run], successes())[0])
        diagnosis["evidence"] = {**diagnosis.get("evidence", {}), "counterfactual": counterfactual(run)}
        return {**run, **diagnosis, "diagnosis": diagnosis}

    def step_event(step: dict[str, Any]) -> dict[str, Any]:
        return {"type": "step", "step": step, "step_id": step["step_id"],
                "node": step.get("node_name"), "output": step.get("output"),
                "latency_ms": step.get("latency_ms", 0)}

    def history(run: dict[str, Any]) -> None:
        events = application.state.events
        if not events.after(run["run_id"]):
            for step in run.get("steps", []):
                events.append(run["run_id"], step_event(step))
            events.append(run["run_id"], {"type": "complete", "status": run["status"], "run": run})

    def warm_models() -> None:
        """Load the ensemble and encoders once so diagnosis latency excludes model loading."""
        if load_ensemble() is None:
            return
        try:
            probe = Engine(":memory:").run("Buy 2 items at $3 each with a 10% discount.", "math")
            diagnose_many([probe])
        except Exception:
            logger.warning("Model warm-up failed", exc_info=True)

    def seed_examples() -> None:
        examples = {
            "finance": "Convert INR 50,000 to USD and compute EMI for 12 months at 9%.",
            "sql": "Find the total sales in the north region.",
            "doc_qa": "How many days do I have to request a refund?",
            "math": "Buy 8 items at $12 each with a 10% discount. What is the total?",
        }
        finance_id = None
        # An older finance run: its 14-day-old quote is what "recent cached quote" finds,
        # so the demo shows that alternative being tested and rejected by the freshness check.
        from datetime import timedelta
        old = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        history(application.state.engine.run("Convert INR 80,000 to USD and compute EMI for 24 months at 10%.",
                                             task_family="finance", params={"frozen_at": old}))
        for family, prompt in examples.items():
            run = application.state.engine.run(prompt, task_family=family)
            history(run)
            if family == "finance":
                finance_id = run["run_id"]
        history(application.state.engine.inject(finance_id, 3, "stale_data"))

    def metrics() -> dict[str, Any]:
        unavailable = {"status": "not_available", "top1": None, "top3": None, "mrr": None,
                       "within_one": None, "unseen_top1": None, "auroc": None,
                       "baselines": {}, "splits": {}, "message": "No measured evaluation artifact. Run python -m blackbox.evaluation."}
        if not metrics_file.is_file():
            return unavailable
        try:
            payload = json.loads(metrics_file.read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("metrics.json must contain an object")
            # Preserve the evaluator's provenance and measured metrics; never substitute PRD targets.
            # Later local (zero-cost) experiments live in separate files and are attached read-only.
            extras = {}
            for key, name in (("natural_local", "metrics_natural.json"), ("hybrid_judge", "hybrid_judge.json"),
                              ("whowhen_windows", "benchmark_whowhen_windows.json"), ("extra", "metrics_extra.json"),
                              ("hard_negatives", "metrics_hardneg.json"), ("heldout_workflow", "metrics_heldout_workflow.json"),
                              ("live_recovery", "metrics_live.json")):
                extra = metrics_file.parent / name
                if extra.is_file():
                    try:
                        extras[key] = json.loads(extra.read_text(encoding="utf-8"))
                    except ValueError:
                        logger.warning("Unreadable %s", extra)
            return {"status": "available", **payload, "extras": extras}
        except (OSError, ValueError) as error:
            logger.warning("Unable to load evaluation artifact: %s", error)
            return {**unavailable, "message": "The evaluation artifact cannot be read; regenerate metrics.json."}

    async def execute(run_id: str, request: RunRequest) -> None:
        try:
            callback = lambda step: application.state.events.append(run_id, step_event(step))
            engine = engine_for(request.provider)
            faults = ({int(request.plant_fault["step"]): str(request.plant_fault["fault"])}
                      if request.plant_fault else None)
            run = await asyncio.to_thread(engine.run, request.prompt,
                                          task_family=request.task_family, params=request.params,
                                          on_step=callback, run_id=run_id, live_recovery=request.live_recovery,
                                          faults=faults)
            application.state.events.append(run_id, {"type": "complete", "status": run["status"], "run": run})
            if run["status"] == "FAILED":
                from blackbox.alerts import maybe_alert
                diagnosis = await asyncio.to_thread(lambda: diagnoses_for([run], successes())[0])
                await asyncio.to_thread(maybe_alert, run, diagnosis)
        except Exception:
            logger.exception("Sandbox run %s failed during execution", run_id)
            run = get_run(run_id)
            run.update(status="ERROR", success=False, error="Execution failed; inspect the server log for details.")
            application.state.store.save_run(run)
            application.state.events.append(run_id, {"type": "error", "status": "ERROR", "message": run["error"]})

    @router.get("/health")
    def health():
        engine = application.state.engine
        return {"status": "ok", "mode": engine.llm.provider, "model": engine.llm.model, "version": "0.2.0",
                "llm_warning": engine.llm_warning}

    @router.get("/llm/status")
    def llm_status():
        engine = application.state.engine
        return {**provider_status(), "active_provider": engine.llm.provider, "active_model": engine.llm.model,
                "warning": engine.llm_warning, "usage": UsageLedger().summary()}

    @router.get("/models/status")
    def models_status():
        return ensemble_status()

    @router.get("/dataset")
    def dataset_info():
        manifests = {}
        for path in sorted((ROOT / "data").glob("dataset_*.manifest.json")):
            try:
                manifests[path.name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
        return {"datasets": manifests}

    @router.get("/faults")
    def faults():
        return {"faults": FAULT_CATALOG}

    @router.post("/runs", status_code=202)
    async def start_run(request: RunRequest):
        engine_for(request.provider)
        try:
            parameter_values = dict(request.params or {})
            if "frozen_at" in parameter_values:
                parse_time(parameter_values.pop("frozen_at"))
            parameters(request.prompt, request.task_family, parameter_values)
            json.dumps(parameter_values, allow_nan=False)
        except (ValueError, TypeError, KeyError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        run_id = uuid.uuid4().hex
        application.state.store.save_run({"run_id": run_id, "prompt": request.prompt,
            "task_family": request.task_family, "status": "QUEUED", "steps": [],
            "created_at": datetime.now(timezone.utc).isoformat(), "success": None,
            "total_tokens": 0, "total_time": 0})
        task = asyncio.create_task(execute(run_id, request))
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)
        return {"run_id": run_id, "status": "QUEUED"}

    @router.get("/runs")
    def list_runs(limit: int = Query(default=100, ge=1, le=1000), status: str | None = None):
        all_runs = application.state.store.list_runs(limit=10000)
        selected = [run for run in all_runs if status is None or run.get("status") == status.upper()]
        normal = [run for run in all_runs if run.get("success") is True]
        page = selected[:limit]
        complete = [run for run in page if run.get("status") in {"PASSED", "FAILED"}]
        found = dict(zip([r["run_id"] for r in complete], diagnoses_for(complete, normal)))
        summaries = []
        for run in page:
            diagnosis = found.get(run["run_id"], {})
            root = diagnosis.get("root_cause") or {}
            summaries.append({key: value for key, value in {**run, **diagnosis,
                "step_count": len(run.get("steps", [])), "suspect_step": root.get("step"),
                "blame": root.get("confidence")}.items() if key not in {"steps", "checkpoints", "step_scores", "evidence"}})
        return {"runs": summaries, "total": len(selected)}

    @router.get("/runs/{run_id}")
    def read_run(run_id: str):
        return detailed(get_run(run_id))

    @router.websocket("/runs/{run_id}/stream")
    async def stream(websocket: WebSocket, run_id: str, after: int = 0):
        await websocket.accept()
        try:
            try:
                run = get_run(run_id)
            except HTTPException:
                await websocket.send_json({"type": "error", "message": "Run not found."})
                await websocket.close(code=1008)
                return
            if run.get("status") in {"PASSED", "FAILED"}:
                history(run)
            cursor = max(0, after)
            while True:
                events = application.state.events.after(run_id, cursor)
                for event in events:
                    await websocket.send_json(event)
                    cursor = event["event_id"]
                    if event["type"] in {"complete", "error"}:
                        await websocket.close(code=1000)
                        return
                if not events and get_run(run_id).get("status") in {"PASSED", "FAILED", "ERROR"}:
                    # Reconnected clients already beyond the terminal event receive an explicit terminator.
                    latest = application.state.events.after(run_id)
                    if latest and cursor >= latest[-1]["event_id"] and latest[-1]["type"] in {"complete", "error"}:
                        await websocket.close(code=1000)
                        return
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            return

    def ready(run_id: str) -> dict[str, Any]:
        run = get_run(run_id)
        if run.get("status") not in {"PASSED", "FAILED"}:
            raise HTTPException(status_code=409, detail="Wait for the run to finish first.")
        return run

    @router.post("/runs/{run_id}/inject")
    async def inject(run_id: str, request: InjectionRequest):
        ready(run_id)
        try:
            run = await asyncio.to_thread(application.state.engine.inject, run_id, request.step_id, request.fault_type)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        history(run)
        return {**detailed(run), "new_run_id": run["run_id"]}

    @router.post("/runs/{run_id}/replay")
    async def replay(run_id: str, request: ReplayRequest):
        ready(run_id)
        try:
            result = await asyncio.to_thread(application.state.engine.replay, run_id,
                request.from_step, request.patch.as_patch(), request.k, "replay", True, request.mode)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        for replay_id in result.get("replay_run_ids", [result.get("new_run_id")]):
            if replay_id:
                history(get_run(replay_id))
        return result

    @router.get("/runs/{run_id}/explain")
    async def explain(run_id: str, mode: Literal["auto", "template", "llm", "qlora"] = "auto"):
        run = detailed(ready(run_id))
        from blackbox.explain.narrator import narrate
        return await asyncio.to_thread(narrate, run, run["diagnosis"], mode)

    @router.get("/runs/{run_id}/alternatives")
    def alternatives(run_id: str, step: int | None = Query(default=None, ge=1), allow_billed: bool = False):
        run = ready(run_id)
        chosen = step or ((detailed(run).get("root_cause") or {}).get("step")) or 1
        try:
            return application.state.engine.alternatives(run_id, chosen, successes(), allow_billed)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.post("/runs/{run_id}/experiments")
    async def run_experiments(run_id: str, request: ExperimentRequest):
        ready(run_id)
        try:
            result = await asyncio.to_thread(application.state.engine.test_alternatives, run_id, request.step,
                                             request.strategies, request.k, successes(), request.allow_billed,
                                             request.mode)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        for branch in result["branches"]:
            for replay_id in branch.get("branch_run_ids", []):
                history(get_run(replay_id))
        return result

    @router.get("/runs/{run_id}/experiments")
    def list_experiments(run_id: str):
        get_run(run_id)
        return {"run_id": run_id, "experiments": application.state.store.list_experiments(run_id)}

    @router.get("/runs/{run_id}/report", response_class=PlainTextResponse)
    def report(run_id: str):
        run = ready(run_id)
        from blackbox.report import incident_report
        full = detailed(run)
        return PlainTextResponse(incident_report(run, full.get("diagnosis"), application.state.store.list_experiments(run_id)),
                                 media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="incident-{run_id}.md"'})

    @router.get("/runs/{run_id}/memory")
    def incident_memory(run_id: str, step: int | None = Query(default=None, ge=1)):
        run = ready(run_id)
        from blackbox.memory import similar_incidents
        chosen = step or ((detailed(run).get("root_cause") or {}).get("step")) or 1
        return similar_incidents(application.state.store, run, chosen)

    @router.post("/runs/{run_id}/review")
    def review(run_id: str, request: ReviewRequest):
        run = ready(run_id)
        if run.get("status") != "FAILED":
            raise HTTPException(status_code=422, detail="Only failed runs can be reviewed")
        diagnosis = detailed(run).get("root_cause") or {}
        label = diagnosis.get("step") if request.verdict == "confirmed" else request.label_step
        if request.verdict == "wrong_step" and not label:
            raise HTTPException(status_code=422, detail="Give the correct step when the suspect is wrong")
        run["review"] = {"verdict": request.verdict, "label_step": label if request.verdict != "uncertain" else None,
                         "suspect_step": diagnosis.get("step"), "note": request.note,
                         "reviewed_at": datetime.now(timezone.utc).isoformat(), "queued_for_training": request.verdict != "uncertain"}
        application.state.store.save_run(run)
        return run["review"]

    @router.post("/ingest/claude-code")
    def ingest_claude_code(event: dict[str, Any]):
        """Record Claude Code hook events (PostToolUse, PostToolUseFailure, UserPromptSubmit, Stop)."""
        from blackbox.storage import redact
        session = str(event.get("session_id") or "unknown")[:64]
        run_id = "CC-" + session
        store = application.state.store
        try:
            run = store.get_run(run_id)
        except KeyError:
            now = datetime.now(timezone.utc).isoformat()
            run = {"run_id": run_id, "task_id": "claude-code", "task_family": "claude-code", "template_id": "claude-code",
                   "prompt": "Claude Code session " + session, "params": {}, "status": "RECORDED", "success": None,
                   "created_at": now, "frozen_at": now, "final_answer": None, "gold_answer": None, "steps": [],
                   "llm_provider": "external", "model": "claude-code", "adapter": "claude-code-hooks",
                   "capabilities": {"record": True, "diagnose": True, "checkpoint": False, "fork": False,
                                    "resume": False, "selective_reuse": False},
                   "total_tokens": 0, "billed_tokens": 0, "cost_usd": 0.0}
        name = event.get("hook_event_name", "")
        if name == "UserPromptSubmit" and event.get("prompt"):
            run["prompt"] = str(event["prompt"])[:4000]
        elif name in ("PostToolUse", "PostToolUseFailure"):
            response = event.get("tool_response")
            error = event.get("error") or (response.get("error") if isinstance(response, dict) else None)
            failed = name == "PostToolUseFailure" or bool(error)
            index = len(run["steps"]) + 1
            output = ({"error": str(error or "tool failed")[:2000], "error_type": event.get("error_type")}
                      if failed else redact(response))
            run["steps"].append({"run_id": run_id, "step_id": index, "node_name": str(event.get("tool_name", "tool")),
                                 "node_type": "tool", "parent_step_ids": [index - 1] if index > 1 else [],
                                 "input": redact(event.get("tool_input")), "output": output, "state_before": {}, "state_after": {},
                                 "state_diff": {}, "checkpoint_id": None, "tool_error": failed, "latency_ms": 0,
                                 "tokens_in": 0, "tokens_out": 0, "retries": 0, "action": "executed", "llm_call": False})
            if failed:
                run["status"], run["success"] = "FAILED", False
        elif name in ("Stop", "SessionEnd") and run["status"] == "RECORDED":
            run["success"] = None
        failures = [s["step_id"] for s in run["steps"] if s.get("tool_error")]
        run["acceptance"] = {"outcome": "failed" if failures else "unknown",
                             "checks": [{"id": "no_errors", "label": "No tool call failed", "status": "fail" if failures else "unknown",
                                         "detail": f"Failed tool calls at steps {failures}." if failures else "No independent check for this session."}],
                             "verifier": "tool-failure signal only"}
        store.save_run(run)
        return {"run_id": run_id, "steps": len(run["steps"]), "status": run["status"]}

    @router.get("/runs/{run_id}/regression-test")
    def regression_test(run_id: str, step: int | None = Query(default=None, ge=1)):
        run = ready(run_id)
        from blackbox.regression import export_test
        chosen = step or ((detailed(run).get("root_cause") or {}).get("step"))
        try:
            return export_test(application.state.engine, run, chosen)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/runs/{run_id}/spans")
    def spans(run_id: str):
        return {"run_id": run_id, "conventions": "OpenInference", "spans": spans_for_run(get_run(run_id))}

    @router.get("/runs/{run_id}/suggest-fix")
    def suggest_fix(run_id: str, step: int = Query(ge=1)):
        ready(run_id)
        try:
            return application.state.engine.suggest_fix(run_id, step)
        except (KeyError, ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @router.get("/compare")
    def compare(a: str, b: str):
        return compare_runs(ready(a), ready(b))

    @router.get("/stats")
    def stats():
        runs = application.state.store.list_runs(limit=10000)
        passed = [run for run in runs if run.get("status") == "PASSED"]
        failed = [run for run in runs if run.get("status") == "FAILED"]
        components: dict[str, int] = {}
        latencies = []
        for run, diagnosis in zip(failed, diagnoses_for(failed, passed)):
            root = diagnosis.get("root_cause") or {}
            component = root.get("node", "unknown")
            components[component] = components.get(component, 0) + 1
            latency = diagnosis.get("latency_ms", diagnosis.get("diagnosis_latency_ms"))
            if isinstance(latency, (float, int)):
                latencies.append(latency)
        return {"totals": {"runs": len(runs), "passed": len(passed), "failed": len(failed),
                    "running": sum(run.get("status") in {"QUEUED", "RUNNING"} for run in runs)},
                "by_component": [{"component": key, "count": value} for key, value in sorted(components.items(), key=lambda pair: -pair[1])],
                "diagnosis_latency_ms": sum(latencies) / len(latencies) if latencies else None,
                "top1": metrics().get("top1"), "mode": application.state.engine.llm.provider,
                "total_tokens": sum(run.get("total_tokens", 0) for run in runs),
                "billed_tokens": sum(run.get("billed_tokens", 0) for run in runs),
                "cost_usd": round(sum(run.get("cost_usd", 0) for run in runs), 6),
                "suspiciousness": sorted(ochiai(passed + failed), key=lambda c: -c["suspiciousness"])[:8],
                "method": "ensemble" if load_ensemble() else "observable_evidence_heuristic"}

    @router.get("/eval")
    def evaluation():
        return metrics()

    @application.get("/api/docs", include_in_schema=False)
    def api_documentation():
        return get_swagger_ui_html(openapi_url="/api/openapi.json", title="Black Box API")

    @application.get("/api/openapi.json", include_in_schema=False)
    def api_schema():
        return application.openapi()

    application.include_router(router)
    application.include_router(router, prefix="/api", include_in_schema=False)
    dashboard = Path(__file__).resolve().parents[2] / "dashboard" / "dist"
    # The source checkout and Docker image both keep dashboard beside blackbox.
    if not dashboard.exists():
        dashboard = Path(__file__).resolve().parents[2].parent / "dashboard" / "dist"
    if dashboard.is_dir():
        if (dashboard / "assets").is_dir():
            application.mount("/assets", StaticFiles(directory=dashboard / "assets"), name="assets")

        @application.get("/", include_in_schema=False)
        def dashboard_home():
            return FileResponse(dashboard / "index.html", headers={"Cache-Control": "no-cache"})

        @application.get("/{path:path}", include_in_schema=False)
        def dashboard_fallback(path: str):
            if path == "api" or path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found.")
            return FileResponse(dashboard / "index.html", headers={"Cache-Control": "no-cache"})
    return application


app = create_app()
