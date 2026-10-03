"""Explicit dependency DAG for the four task families, with LLM and tool nodes."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import time

from . import tools
from .llm_nodes import NODES, PRIMARY_TOOL, node_rng
from ..llm.client import BaseLLM, SandboxLLM, estimate_tokens

LLM_NODES = frozenset(NODES)


def specification(family):
    primary = {"finance": ("currency_rate", "tool"), "sql": ("schema_lookup", "tool"),
               "doc_qa": ("doc_search", "retriever"), "math": ("line_items", "tool")}[family]
    transform = {"finance": "convert_currency", "sql": "sql_query", "doc_qa": "extract_policy", "math": "subtotal"}[family]
    return [
        ("planner", "planner", []), ("router", "router", [1]),
        (*primary, [2]), ("formula_search" if family in ("finance", "math") else "policy_reference", "retriever", [1]),
        (transform, "tool", [2, 3]), ("memory", "memory", [5]),
        ("date_util", "tool", [2]), ("calculator", "tool", [2, 4, 6, 7]),
        ("reasoner", "reasoner", [8]), ("final_answer", "final", [9]),
    ]


@dataclass
class NodeContext:
    llm: BaseLLM = field(default_factory=SandboxLLM)
    seed: int = 7
    variant: int = 0
    run_id: str | None = None
    purpose: str = "agent"
    overrides: dict = field(default_factory=dict)


def _required(output, key):
    if not isinstance(output, dict) or output.get(key) is None:
        raise ValueError(f"Required tool result unavailable: {key}")
    return output[key]


def _tool(step_id, family, inputs, ctx):
    deps = inputs.get("dependencies", {})
    at = inputs["frozen_at"]
    route = deps.get("router", {})
    route = route if isinstance(route, dict) else {}
    args = route.get("arguments", {})
    args = args if isinstance(args, dict) else {}
    noise = getattr(ctx.llm, "noise", 0) or 0
    if noise and step_id in (3, 5) and node_rng(ctx.seed, f"tool{step_id}", ctx.variant).random() < noise * 0.25:
        # Natural flakiness: a transient upstream timeout.
        raise TimeoutError("Upstream tool request timed out after 5000 ms")
    if step_id == 3:
        expected = PRIMARY_TOOL[family]
        if route.get("attempts", 1) > route.get("step_budget", 3):
            raise RuntimeError("Execution budget exhausted")
        if route.get("tool") != expected:
            raise ValueError(f"Selected tool {route.get('tool')} cannot provide {expected}")
        if family == "finance":
            return tools.currency_rate(args["base_currency"], args["target_currency"], at)
        if family == "sql":
            return tools.schema_lookup()
        if family == "doc_qa":
            return tools.doc_search(args["policy"], args["category"], at)
        return {"quantity": args["quantity"], "unit_price": args["unit_price"], "as_of": at}
    if step_id == 4:
        return {"method": "amortized" if family == "finance" else "standard", "source": "sandbox-reference",
                "version": "current", "factor": 1.0}
    if step_id == 5:
        if family == "finance":
            return {"value": round(args["amount"] * _required(deps["currency_rate"], "rate"), 8), "currency": args["target_currency"]}
        if family == "sql":
            schema = deps["schema_lookup"]
            column = (schema.get("metrics") or {}).get(args.get("metric"), _required(schema, "aggregate_column")) \
                if isinstance(schema, dict) else _required(schema, "aggregate_column")
            result = tools.sql_query(args["region"], column, args.get("quarter", "all"))
            return {**result, "value": result["rows"][0]["total"]}
        if family == "doc_qa":
            docs = _required(deps["doc_search"], "documents")
            return {"value": docs[0]["days"], "document_id": docs[0]["id"], "version": docs[0].get("version")}
        items = deps["line_items"]
        return {"value": _required(items, "quantity") * _required(items, "unit_price")}
    if step_id == 6:
        result = next(iter(deps.values()))
        return {"value": _required(result, "value"), "source_step": 5}
    if step_id == 7:
        return tools.date_util(at, args.get("months", 0) if family == "finance" else 0)
    if step_id == 8:
        value = _required(deps["memory"], "value")
        reference = deps.get("formula_search", deps.get("policy_reference"))
        factor = _required(reference, "factor")
        term = _required(deps["date_util"], "term_months")
        if family == "finance":
            value = tools.calculator(value, term, args["annual_rate"], _required(reference, "method"))
        elif family == "math":
            value = round(value * (1 - args["discount_pct"] / 100), 2)
        value = round(value * factor, 2)
        return {"value": f"{int(value)} days" if family == "doc_qa" else value, "method": reference["method"]}
    raise ValueError(f"Unknown step: {step_id}")


def execute(step_id, family, inputs, ctx: NodeContext | None = None):
    """Run one node. Returns (output, meta); meta records the LLM call if any."""
    ctx = ctx or NodeContext()
    name = specification(family)[step_id - 1][0]
    meta = {"llm_call": False}
    if name not in LLM_NODES:
        return _tool(step_id, family, inputs, ctx), meta
    messages, parse, sandbox = NODES[name]
    system, user = messages(inputs, family)
    if ctx.overrides.get("prompt"):
        user += "\nAdditional instruction: " + str(ctx.overrides["prompt"])
    seed = ctx.seed + ctx.variant
    llm = ctx.llm
    meta = {"llm_call": True, "provider": llm.provider, "model": ctx.overrides.get("model") or llm.model,
            "prompt": system + "\n\n" + user, "seed": seed, "temperature": ctx.overrides.get("temperature")}
    if isinstance(llm, SandboxLLM):
        start = time.perf_counter()
        noise = llm.noise if ctx.overrides.get("temperature") is None else min(1.0, llm.noise + ctx.overrides["temperature"] * 0.2)
        output = sandbox(inputs, family, node_rng(ctx.seed, name, ctx.variant), noise)
        if noise:
            # Simulation ground truth (never a model input): did the mistake fire here?
            meta["sim_noise"] = output != sandbox(inputs, family, node_rng(ctx.seed, name, ctx.variant), 0.0)
        text = str(output)
        meta.update(tokens_in=estimate_tokens(system + user), tokens_out=estimate_tokens(text) + 40, cached_in=0,
                    reasoning=0, cost_usd=0.0, estimated=True, llm_latency_ms=(time.perf_counter() - start) * 1000,
                    temperature=meta["temperature"] if meta["temperature"] is not None else 0)
        return output, meta
    result = llm.complete_json(system, user, seed=seed, temperature=ctx.overrides.get("temperature"),
                               model=ctx.overrides.get("model"), purpose=ctx.purpose, run_id=ctx.run_id, node=name)
    meta.update(tokens_in=result.tokens_in, tokens_out=result.tokens_out, cached_in=result.cached_in,
                reasoning=result.reasoning, cost_usd=result.cost_usd, estimated=result.estimated,
                llm_latency_ms=result.latency_ms, temperature=result.temperature, seed=result.seed,
                model=result.model, raw_response=result.text[:2000])
    if result.error:
        return {"error": result.error, "error_type": "InvalidModelOutput"}, meta
    return parse(result.data, inputs, family), meta


def build_langgraph(family, step_runner):
    """Express the agent as a LangGraph StateGraph; the recorder runs each node.

    ``step_runner(index, name, kind, parents)`` performs checkpoint restore, cache
    lookup, patching and execution for one node and returns its output.
    """
    from typing import TypedDict
    from langgraph.graph import END, START, StateGraph

    class AgentState(TypedDict, total=False):
        outputs: dict

    graph = StateGraph(AgentState)
    nodes = specification(family)
    for index, (name, kind, parents) in enumerate(nodes, start=1):
        def node(state, index=index, name=name, kind=kind, parents=parents):
            output = step_runner(index, name, kind, parents)
            return {"outputs": {**state.get("outputs", {}), name: deepcopy(output)}}
        graph.add_node(f"{index:02d}_{name}", node)
    names = [f"{i:02d}_{n}" for i, (n, _, _) in enumerate(nodes, start=1)]
    graph.add_edge(START, names[0])
    for left, right in zip(names, names[1:]):
        graph.add_edge(left, right)
    graph.add_edge(names[-1], END)
    return graph.compile()
