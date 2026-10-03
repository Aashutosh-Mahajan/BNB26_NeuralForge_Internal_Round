"""Explicit dependency DAG for the four sandbox task families."""
from copy import deepcopy
from . import tools

def specification(family):
    primary = {"finance": ("currency_rate", "tool"), "sql": ("schema_lookup", "tool"), "doc_qa": ("doc_search", "retriever"), "math": ("line_items", "tool")}[family]
    transform = {"finance": "convert_currency", "sql": "sql_query", "doc_qa": "extract_policy", "math": "subtotal"}[family]
    return [
        ("planner", "planner", []), ("router", "router", [1]),
        (*primary, [2]), ("formula_search" if family in ("finance", "math") else "policy_reference", "retriever", [1]),
        (transform, "tool", [2, 3]), ("memory", "memory", [5]),
        ("date_util", "tool", [2]), ("calculator", "tool", [2, 4, 6, 7]),
        ("reasoner", "reasoner", [8]), ("final_answer", "final", [9]),
    ]

def _required(output, key):
    if not isinstance(output, dict) or output.get(key) is None:
        raise ValueError(f"Required tool result unavailable: {key}")
    return output[key]

def execute(step_id, family, inputs):
    deps = inputs.get("dependencies", {})
    at = inputs["frozen_at"]
    if step_id == 1:
        return {"task_family": family, "constraints": deepcopy(inputs["params"]), "steps": ["retrieve", "compute", "verify"]}
    if step_id == 2:
        plan = deps["planner"]
        primary = {"finance": "currency_rate", "sql": "schema_lookup", "doc_qa": "doc_search", "math": "line_items"}[family]
        return {"tool": primary, "arguments": deepcopy(plan["constraints"]), "attempts": 1, "step_budget": 3}
    route = deps.get("router", {})
    args = route.get("arguments", {})
    if step_id == 3:
        expected = {"finance": "currency_rate", "sql": "schema_lookup", "doc_qa": "doc_search", "math": "line_items"}[family]
        if route.get("attempts", 1) > route.get("step_budget", 3):
            raise RuntimeError("Execution budget exhausted")
        if route.get("tool") != expected:
            raise ValueError(f"Selected tool {route.get('tool')} cannot provide {expected}")
        if family == "finance":
            return tools.currency_rate(args["base_currency"], args["target_currency"], at)
        if family == "sql":
            return tools.schema_lookup()
        if family == "doc_qa":
            return tools.doc_search(args["policy"], args["days"], at)
        return {"quantity": args["quantity"], "unit_price": args["unit_price"], "as_of": at}
    if step_id == 4:
        return {"method": "amortized" if family == "finance" else "standard", "source": "sandbox-reference", "version": "current", "factor": 1.0}
    if step_id == 5:
        if family == "finance":
            return {"value": round(args["amount"] * _required(deps["currency_rate"], "rate"), 8), "currency": args["target_currency"]}
        if family == "sql":
            result = tools.sql_query(args["region"], args["scale"], _required(deps["schema_lookup"], "aggregate_column"))
            return {**result, "value": result["rows"][0]["total"]}
        if family == "doc_qa":
            docs = _required(deps["doc_search"], "documents")
            return {"value": docs[0]["days"], "document_id": docs[0]["id"]}
        items = deps["line_items"]
        return {"value": _required(items, "quantity") * _required(items, "unit_price")}
    if step_id == 6:
        result = next(iter(deps.values()))
        return {"value": _required(result, "value"), "source_step": 5}
    if step_id == 7:
        return tools.date_util(at, args.get("months", 0))
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
    if step_id == 9:
        return {"answer": _required(deps["calculator"], "value"), "rationale": "Computed from the retrieved evidence and recorded task constraints."}
    if step_id == 10:
        return {"answer": _required(deps["reasoner"], "answer")}
    raise ValueError(f"Unknown step: {step_id}")
