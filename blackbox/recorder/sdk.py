"""``blackbox.wrap``: record the built-in agent or any compiled LangGraph graph.

For external LangGraph graphs the adapter uses LangGraph's own runtime:

- state comes from native checkpoints (reducers applied by LangGraph, not dict merges);
- dependencies come from supersteps: a task depends on the tasks of the previous
  superstep, narrowed to named sources when a join trigger names them;
- node exceptions end the run as an ERROR incident with its last good checkpoint;
- ``fork`` patches one node's output at its native checkpoint and lets LangGraph
  continue (LangGraph re-executes the following nodes); ``resume`` continues a
  crashed run from its last good checkpoint. Both create new linked runs.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import time
import uuid

from ..storage import Store, canonical, content_key, redact

CAPABILITIES = {
    "builtin": {"record": True, "diagnose": True, "checkpoint": True, "fork": True, "resume": True,
                "selective_reuse": True},
    "langgraph": {"record": True, "diagnose": True, "checkpoint": True, "fork": True, "resume": True,
                  "selective_reuse": False},
    "langgraph_no_checkpointer": {"record": True, "diagnose": True, "checkpoint": False, "fork": False,
                                  "resume": False, "selective_reuse": False},
}


@dataclass(frozen=True)
class SandboxAgent:
    """The built-in LangGraph agent (provider chosen by LLM_PROVIDER)."""
    task_family: str = "finance"
    params: dict = field(default_factory=dict)


class WrappedAgent:
    capabilities = CAPABILITIES["builtin"]

    def __init__(self, agent, db_path, llm=None):
        from ..engine import Engine
        self.agent = agent
        self.engine = Engine(db_path, llm=llm)

    def invoke(self, inputs, on_step=None):
        if isinstance(inputs, str):
            prompt, params = inputs, {}
        elif isinstance(inputs, dict):
            prompt = inputs["prompt"]
            params = inputs.get("params", {})
        else:
            raise TypeError("invoke expects a prompt string or {'prompt': ..., 'params': ...}")
        configured = deepcopy(self.agent.params)
        configured.update(params)
        return self.engine.run(prompt, self.agent.task_family, configured, on_step=on_step)

    run = invoke


def _message(m):
    data = {"type": getattr(m, "type", m.__class__.__name__), "content": getattr(m, "content", None)}
    for key in ("name", "tool_calls", "tool_call_id"):
        value = getattr(m, key, None)
        if value:
            data[key] = value
    return data


def _jsonable(value):
    try:
        canonical(value)
        return redact(deepcopy(value))
    except (TypeError, ValueError):
        return redact(json.loads(json.dumps(value, default=lambda o: _message(o) if hasattr(o, "content") else repr(o))))


def _prompt_text(inputs, parent=None) -> str:
    if inputs is None:
        return (parent or {}).get("prompt", "")
    if isinstance(inputs, dict):
        for key in ("request", "question", "input", "query", "task", "prompt"):
            if isinstance(inputs.get(key), str):
                return inputs[key][:4000]
    return json.dumps(_jsonable(inputs))[:4000]


def _node_type(name: str) -> str:
    lowered = name.lower()
    for key, kind in (("plan", "planner"), ("parse", "planner"), ("route", "router"), ("retriev", "retriever"),
                      ("search", "retriever"), ("memory", "memory"), ("reason", "reasoner"), ("final", "final"),
                      ("answer", "final"), ("agent", "reasoner"), ("llm", "reasoner"), ("tool", "tool"),
                      ("fetch", "tool"), ("compute", "tool"), ("calc", "tool")):
        if key in lowered:
            return kind
    return "tool"


def _usage(result):
    usage = {"tokens_in": 0, "tokens_out": 0}
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for message in messages if isinstance(messages, list) else []:
        meta = getattr(message, "usage_metadata", None) or {}
        usage["tokens_in"] += meta.get("input_tokens", 0)
        usage["tokens_out"] += meta.get("output_tokens", 0)
    return usage


class WrappedGraph:
    """Records an external compiled LangGraph run with native checkpoints."""

    def __init__(self, graph, db_path, name="external-langgraph", check=None, ensure_checkpointer=True):
        self.store = db_path if isinstance(db_path, Store) else Store(db_path)
        self.name = name
        self.check = check
        if getattr(graph, "checkpointer", None) is None and ensure_checkpointer and hasattr(graph, "builder"):
            from langgraph.checkpoint.memory import InMemorySaver
            graph = graph.builder.compile(checkpointer=InMemorySaver())
        self.graph = graph
        self.capabilities = CAPABILITIES["langgraph" if getattr(graph, "checkpointer", None) is not None
                                          else "langgraph_no_checkpointer"]

    # ------------------------------------------------------------ recording
    def _new_run(self, inputs, parent=None, thread_id=None):
        now = datetime.now(timezone.utc).isoformat()
        return {"run_id": "R-" + uuid.uuid4().hex[:16], "task_id": "external:" + self.name, "task_family": self.name,
                "template_id": self.name, "prompt": _prompt_text(inputs, parent),
                "params": {}, "status": "RUNNING", "success": None, "created_at": now, "frozen_at": now,
                "final_answer": None, "gold_answer": None, "steps": [], "llm_provider": "external", "model": self.name,
                "execution_mode": "external-langgraph", "orchestrator": "langgraph", "adapter": "langgraph",
                "capabilities": self.capabilities, "thread_id": thread_id or uuid.uuid4().hex,
                "parent_run_id": (parent or {}).get("run_id"), "total_tokens": 0, "billed_tokens": 0, "cost_usd": 0.0}

    def _stream(self, run, payload, config, on_step=None, prefix=(), seed_previous=(), carry=()):
        """Consume task/checkpoint events; returns the run with steps, state and outcome.

        ``carry``: prefix steps that completed inside a superstep that later failed; LangGraph
        keeps their writes, so they belong to the superstep that finishes on resume."""
        run["steps"].extend(deepcopy(list(prefix)))
        pending = {}
        superstep_tasks = [s for s in run["steps"] if s["step_id"] in set(carry)]
        previous_tasks = [s for s in run["steps"] if s["step_id"] in set(seed_previous)]
        last_checkpoint, state = None, {}
        started = time.perf_counter()
        error = None
        try:
            for mode, event in self.graph.stream(payload, config, stream_mode=["tasks", "checkpoints"]):
                if mode == "checkpoints":
                    last_checkpoint = event["config"]["configurable"].get("checkpoint_id")
                    state = _jsonable(event["values"])
                    if not any(t["action"] == "executed" for t in superstep_tasks):
                        continue  # no task finished since the last checkpoint (e.g. the resume point)
                    for step in superstep_tasks:
                        step["native_checkpoint"] = {"thread_id": run["thread_id"], "checkpoint_id": last_checkpoint}
                        step["state_after"] = deepcopy(state)
                        self.store.save_checkpoint(step["checkpoint_id"], run["run_id"], step["step_id"], state)
                    if superstep_tasks:
                        previous_tasks = superstep_tasks
                    superstep_tasks = []
                    self.store.save_run(run)
                    continue
                if "input" in event and "result" not in event:  # task started
                    pending[event["id"]] = (event, time.perf_counter())
                    continue
                begun, t0 = pending.pop(event["id"], (event, time.perf_counter()))
                index = len(run["steps"]) + 1
                named = {t.split(":")[1] for t in begun.get("triggers", ()) if t.startswith("join:")}
                sources = [p for p in previous_tasks if not named or p["node_name"] in "+".join(named)] or previous_tasks
                result = _jsonable(event.get("result"))
                err = event.get("error")
                output = {"error": str(err), "error_type": type(err).__name__} if err else result
                step = {"run_id": run["run_id"], "step_id": index, "node_name": event["name"],
                        "node_type": _node_type(event["name"]), "task_id": event["id"],
                        "attempt": sum(s["node_name"] == event["name"] for s in run["steps"]) + 1,
                        "parent_step_ids": [p["step_id"] for p in sources], "triggers": list(begun.get("triggers", ())),
                        "input": _jsonable(begun.get("input")), "output": output, "state_before": deepcopy(state),
                        "state_after": deepcopy(state), "state_diff": {event["name"]: {"after": output}},
                        "checkpoint_id": f"{run['run_id']}:{index}", "cache_key": content_key(event["name"], _jsonable(begun.get("input"))),
                        **_usage(event.get("result")), "cost_usd": 0.0,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 3), "retries": 0,
                        "tool_error": bool(err), "action": "executed", "actual_execution": True}
                step["llm_call"] = step["tokens_in"] > 0
                run["steps"].append(step)
                superstep_tasks.append(step)
                if on_step:
                    on_step(deepcopy(step))
        except Exception as exc:  # A crashed node ends the run as an incident, never "running".
            error = exc
        run["final_state"] = state
        run["last_checkpoint"] = {"thread_id": run["thread_id"], "checkpoint_id": last_checkpoint}
        for key in ("answer", "output", "final_answer", "result"):
            if key in state:
                run["final_answer"] = state[key]
                break
        if error is not None:
            run.update(status="ERROR", success=False, error=f"{type(error).__name__}: {error}",
                       error_type=type(error).__name__)
            failed = next((s for s in reversed(run["steps"]) if s.get("tool_error")), None)
            run["crashed_step"] = failed["step_id"] if failed else None
        verdict = None
        if self.check is not None and error is None:
            try:
                verdict = self.check(state)
            except Exception as exc:
                verdict, run["check_error"] = None, str(exc)
        if error is None:
            run["success"] = verdict
            run["status"] = "PASSED" if verdict else "FAILED" if verdict is False else "RECORDED"
        run["acceptance"] = {"outcome": "passed" if verdict else "failed" if (verdict is False or error) else "unknown",
                             "checks": [{"id": "user_check", "label": "Your acceptance check",
                                         "status": "pass" if verdict else "fail" if verdict is False else "unknown",
                                         "detail": run.get("error") or run.get("check_error") or
                                         ("Check returned " + json.dumps(verdict) if self.check else "No check supplied.")}],
                             "verifier": "user-supplied check" if self.check else "none"}
        run["total_tokens"] = sum(s.get("tokens_in", 0) + s.get("tokens_out", 0) for s in run["steps"])
        run["total_time"] = round((time.perf_counter() - started) * 1000, 3)
        self.store.save_run(run)
        from .otel import export_run
        export_run(run)
        return run

    def invoke(self, inputs, config=None, *, on_step=None):
        run = self._new_run(inputs)
        config = {**(config or {}), "configurable": {**((config or {}).get("configurable") or {}), "thread_id": run["thread_id"]}}
        self.store.save_run(run)
        return self._stream(run, inputs, config, on_step)

    run = invoke

    # --------------------------------------------------------- fork / resume
    def _require(self, run_id):
        if not self.capabilities["fork"]:
            raise RuntimeError("This graph has no checkpointer, so it can be recorded but not forked or resumed")
        return self.store.get_run(run_id)

    def fork(self, run_id: str, step_id: int, output: dict, on_step=None):
        """Replace one node's output at its native checkpoint and let LangGraph continue."""
        original = self._require(run_id)
        step = next(s for s in original["steps"] if s["step_id"] == step_id)
        native = step.get("native_checkpoint")
        if not native:
            raise ValueError("That step has no native checkpoint")
        base = {"configurable": {"thread_id": native["thread_id"], "checkpoint_ns": "", "checkpoint_id": native["checkpoint_id"]}}
        forked = self.graph.update_state(base, output, as_node=step["node_name"])
        run = self._new_run(None, parent=original, thread_id=native["thread_id"])
        run["fork"] = {"from_step": step_id, "node": step["node_name"], "patch": _jsonable(output),
                       "from_checkpoint": native["checkpoint_id"],
                       "semantics": "LangGraph applies reducers to the patch and re-executes the nodes that follow"}
        # Keep every step that does not depend on the patched one (e.g. parallel branches).
        affected, frontier = set(), {step_id}
        for s in original["steps"]:
            if any(p in frontier for p in s.get("parent_step_ids", [])):
                frontier.add(s["step_id"])
                affected.add(s["step_id"])
        kept = [s for s in original["steps"] if s["step_id"] not in affected]
        renumber = {s["step_id"]: i for i, s in enumerate(kept, start=1)}
        prefix = []
        for s in kept:
            copy = {**deepcopy(s), "run_id": run["run_id"], "step_id": renumber[s["step_id"]],
                    "parent_step_ids": [renumber[p] for p in s.get("parent_step_ids", []) if p in renumber],
                    "checkpoint_id": f"{run['run_id']}:{renumber[s['step_id']]}", "actual_execution": False}
            if s["step_id"] == step_id:
                copy.update(output=_jsonable(output), action="patched", tool_error=False)
            else:
                copy["action"] = "checkpoint" if s["step_id"] < step_id else "reused"
            prefix.append(copy)
        same_superstep = [renumber[s["step_id"]] for s in kept
                          if (s.get("native_checkpoint") or {}).get("checkpoint_id") == native["checkpoint_id"]]
        self.store.save_run(run)
        result = self._stream(run, None, forked, on_step, prefix, same_superstep)
        for s in result["steps"][len(prefix):]:
            s["action"] = "rerun"
        self.store.save_run(result)
        return result

    def resume(self, run_id: str, on_step=None):
        """Continue a crashed run from its last good checkpoint (e.g. after fixing a tool)."""
        original = self._require(run_id)
        if original.get("status") != "ERROR":
            raise ValueError("Only a crashed (ERROR) run can be resumed")
        last = original["last_checkpoint"]
        # Continue the thread (not a replay of a past checkpoint): LangGraph keeps the writes of
        # tasks that finished before the crash and runs only what is left.
        config = {"configurable": {"thread_id": last["thread_id"], "checkpoint_ns": ""}}
        run = self._new_run(None, parent=original, thread_id=last["thread_id"])
        run["resumed_from"] = {"checkpoint_id": last["checkpoint_id"], "crashed_step": original.get("crashed_step")}
        kept = [s for s in original["steps"] if not s.get("tool_error")]
        renumber = {s["step_id"]: i for i, s in enumerate(kept, start=1)}
        prefix = [{**deepcopy(s), "run_id": run["run_id"], "step_id": renumber[s["step_id"]],
                   "parent_step_ids": [renumber[p] for p in s.get("parent_step_ids", []) if p in renumber],
                   "checkpoint_id": f"{run['run_id']}:{renumber[s['step_id']]}", "action": "checkpoint",
                   "actual_execution": False} for s in kept]
        seed = [renumber[s["step_id"]] for s in kept
                if (s.get("native_checkpoint") or {}).get("checkpoint_id") == last["checkpoint_id"]]
        carry = [renumber[s["step_id"]] for s in kept if not s.get("native_checkpoint")]
        self.store.save_run(run)
        result = self._stream(run, None, config, on_step, prefix, seed, carry)
        for s in result["steps"][len(prefix):]:
            s["action"] = "rerun"
        self.store.save_run(result)
        return result


def wrap(agent, db_path="data/traces.db", llm=None, *, name="external-langgraph", check=None):
    """Wrap the built-in ``SandboxAgent`` or any compiled LangGraph (``.stream``) for recording.

    ``check(final_state) -> True | False | None`` is an optional independent acceptance check.
    """
    if isinstance(agent, SandboxAgent):
        return WrappedAgent(agent, db_path, llm)
    if hasattr(agent, "stream") and hasattr(agent, "invoke"):
        return WrappedGraph(agent, db_path, name=name, check=check)
    raise TypeError("wrap() accepts blackbox.SandboxAgent or a compiled LangGraph graph")
