"""``blackbox.wrap``: record the built-in agent or any compiled LangGraph graph."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import time
import uuid

from ..storage import Store, canonical, content_key


@dataclass(frozen=True)
class SandboxAgent:
    """The built-in LangGraph agent (provider chosen by LLM_PROVIDER)."""
    task_family: str = "finance"
    params: dict = field(default_factory=dict)


class WrappedAgent:
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


def _jsonable(value):
    try:
        canonical(value)
        return deepcopy(value)
    except (TypeError, ValueError):
        return json.loads(json.dumps(value, default=lambda o: getattr(o, "content", None) or repr(o)))


def _node_type(name: str) -> str:
    lowered = name.lower()
    for key, kind in (("plan", "planner"), ("route", "router"), ("retriev", "retriever"), ("search", "retriever"),
                      ("memory", "memory"), ("reason", "reasoner"), ("final", "final"), ("answer", "final"),
                      ("agent", "reasoner"), ("llm", "reasoner"), ("tool", "tool")):
        if key in lowered:
            return kind
    return "tool"


class WrappedGraph:
    """Records every node update of an external compiled LangGraph as a step.

    Each step gets a checkpoint (the accumulated state) and a content-addressed
    cache entry. Diagnosis and trace diff work on these runs; checkpoint replay
    with patches is available for the built-in agent.
    """

    def __init__(self, graph, db_path, name="external-langgraph"):
        self.graph = graph
        self.store = db_path if isinstance(db_path, Store) else Store(db_path)
        self.name = name

    def invoke(self, inputs, config=None, *, success=None, gold_answer=None, on_step=None):
        from .otel import export_run
        run_id = "R-" + uuid.uuid4().hex[:16]
        now = datetime.now(timezone.utc).isoformat()
        run = {"run_id": run_id, "task_id": "external", "task_family": self.name, "template_id": self.name,
               "prompt": json.dumps(_jsonable(inputs))[:4000], "params": {}, "status": "RUNNING", "success": None,
               "created_at": now, "frozen_at": now, "final_answer": None, "gold_answer": gold_answer, "steps": [],
               "llm_provider": "external", "model": None, "execution_mode": "external-langgraph",
               "orchestrator": "langgraph", "total_tokens": 0, "billed_tokens": 0, "cost_usd": 0.0}
        state = _jsonable(inputs) if isinstance(inputs, dict) else {"input": _jsonable(inputs)}
        last_by_node: dict[str, int] = {}
        start = time.perf_counter()
        stamp = start
        for update in self.graph.stream(inputs, config=config, stream_mode="updates"):
            for node, delta in update.items():
                now_t = time.perf_counter()
                before = deepcopy(state)
                output = _jsonable(delta)
                if isinstance(output, dict):
                    state.update(output)
                index = len(run["steps"]) + 1
                parents = [last_by_node[n] for n in last_by_node][-1:] if run["steps"] else []
                usage = {"tokens_in": 0, "tokens_out": 0}
                for message in (delta or {}).get("messages", []) if isinstance(delta, dict) else []:
                    meta = getattr(message, "usage_metadata", None) or {}
                    usage["tokens_in"] += meta.get("input_tokens", 0)
                    usage["tokens_out"] += meta.get("output_tokens", 0)
                step = {"run_id": run_id, "step_id": index, "node_name": node, "node_type": _node_type(node),
                        "parent_step_ids": parents, "input": before, "output": output,
                        "state_before": before, "state_after": deepcopy(state),
                        "state_diff": {node: {"before": None, "after": output}},
                        "checkpoint_id": f"{run_id}:{index}", "cache_key": content_key(node, before),
                        **usage, "cost_usd": 0.0, "latency_ms": round((now_t - stamp) * 1000, 3), "retries": 0,
                        "tool_error": isinstance(output, dict) and bool(output.get("error")), "action": "executed",
                        "actual_execution": True, "llm_call": usage["tokens_in"] > 0}
                stamp = now_t
                run["steps"].append(step)
                last_by_node[node] = index
                self.store.save_checkpoint(step["checkpoint_id"], run_id, index, state)
                self.store.cache_output(step["cache_key"], output)
                self.store.save_run(run)
                if on_step:
                    on_step(deepcopy(step))
        run["final_answer"] = state.get("answer", state.get("output", state.get("messages", [None])[-1] if state.get("messages") else None))
        run["success"] = success if success is not None else (None if gold_answer is None else run["final_answer"] == gold_answer)
        run["status"] = "PASSED" if run["success"] else "FAILED" if run["success"] is False else "RECORDED"
        run["total_tokens"] = sum(s["tokens_in"] + s["tokens_out"] for s in run["steps"])
        run["total_time"] = round((time.perf_counter() - start) * 1000, 3)
        self.store.save_run(run)
        export_run(run)
        return run

    run = invoke


def wrap(agent, db_path="data/traces.db", llm=None):
    """Wrap the built-in ``SandboxAgent`` or any compiled LangGraph (``.stream``) for recording."""
    if isinstance(agent, SandboxAgent):
        return WrappedAgent(agent, db_path, llm)
    if hasattr(agent, "stream") and hasattr(agent, "invoke"):
        return WrappedGraph(agent, db_path)
    raise TypeError("wrap() accepts blackbox.SandboxAgent or a compiled LangGraph graph")
