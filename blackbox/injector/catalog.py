"""Realistic interventions; injection metadata never enters tool outputs."""
from copy import deepcopy
from datetime import timedelta
from ..agent.tools import currency_rate, parse_time

FAULT_CATALOG = [
    {"id": "wrong_tool_value", "label": "Wrong tool value", "node_types": ["tool"], "held_out": False},
    {"id": "empty_result", "label": "Empty result", "node_types": ["tool"], "held_out": False},
    {"id": "tool_timeout", "label": "Tool timeout", "node_types": ["tool"], "held_out": False},
    {"id": "stale_data", "label": "Stale market data", "node_types": ["tool"], "held_out": False},
    {"id": "retrieval_poisoning", "label": "Retrieval poisoning", "node_types": ["retriever"], "held_out": False},
    {"id": "wrong_tool_choice", "label": "Wrong tool choice", "node_types": ["router"], "held_out": False},
    {"id": "wrong_arguments", "label": "Wrong arguments", "node_types": ["router"], "held_out": False},
    {"id": "dropped_constraint", "label": "Dropped constraint", "node_types": ["planner"], "held_out": False},
    {"id": "hallucinated_fact", "label": "Hallucinated fact", "node_types": ["reasoner"], "held_out": False},
    {"id": "premature_final", "label": "Premature final answer", "node_types": ["router"], "held_out": True},
    {"id": "memory_overwrite", "label": "Memory overwrite", "node_types": ["memory"], "held_out": True},
    {"id": "loop_repetition", "label": "Loop / repetition", "node_types": ["router"], "held_out": True},
]

def _change_number(output):
    for key in ("rate", "value", "quantity", "unit_price", "term_months", "answer", "days"):
        if key in output:
            value = output[key]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                output[key] = value * 1.35 + (1 if value == 0 else 0)
                return output
            if isinstance(value, str) and value.endswith(" days"):
                output[key] = f"{int(value.split()[0]) + 15} days"
                return output
    if output.get("aggregate_column"):
        output["aggregate_column"] = "units"
        return output
    raise ValueError("This step has no compatible value to corrupt")

def _arguments(args, family):
    key = {"finance": "amount", "sql": "scale", "doc_qa": "days", "math": "quantity"}[family]
    args[key] = args[key] * 2
    return args

def inject_output(run, step, fault_type):
    fault = next((f for f in FAULT_CATALOG if f["id"] == fault_type), None)
    if fault is None:
        raise ValueError(f"Unknown fault type: {fault_type}")
    if step["node_type"] not in fault["node_types"]:
        raise ValueError(f"{fault['label']} is incompatible with {step['node_type']}")
    out = deepcopy(step["output"])
    if not isinstance(out, dict):
        raise ValueError("This intervention requires a structured step output")
    if fault_type == "empty_result":
        return {}
    if fault_type == "tool_timeout":
        return {"error": "Tool request exceeded the 5000 ms deadline", "retryable": True}
    if fault_type in ("wrong_tool_value", "hallucinated_fact", "memory_overwrite"):
        return _change_number(out)
    if fault_type == "stale_data":
        if step["node_name"] != "currency_rate":
            raise ValueError("Stale market data requires a currency_rate step")
        args = run["params"]
        old_time = (parse_time(run["frozen_at"]) - timedelta(days=366)).isoformat()
        stale = currency_rate(args["base_currency"], args["target_currency"], old_time)
        stale["backup_rate"] = out["backup_rate"]
        return stale
    if fault_type == "retrieval_poisoning":
        if "documents" in out:
            doc = out["documents"][0]
            doc.update({"id": "returns-archive", "days": doc["days"] + 15, "version": "archived", "text": "Returns are accepted within the historical policy window."})
        else:
            out["factor"] = 1.15
            out["version"] = "archived"
            out["source"] = "archived-reference"
        return out
    if fault_type == "wrong_tool_choice":
        out["tool"] = "doc_search" if out["tool"] != "doc_search" else "calculator"
    elif fault_type == "wrong_arguments":
        _arguments(out["arguments"], run["task_family"])
    elif fault_type == "dropped_constraint":
        constraints = out["constraints"]
        key = {"finance": "months", "sql": "region", "doc_qa": "days", "math": "discount_pct"}[run["task_family"]]
        fallback = {"months": 24, "region": "east", "days": 14, "discount_pct": 0}[key]
        if constraints[key] == fallback:
            fallback = {"months": 36, "region": "west", "days": 60, "discount_pct": 25}[key]
        constraints[key] = fallback
    elif fault_type == "premature_final":
        out["tool"] = "final_answer"
    elif fault_type == "loop_repetition":
        out["attempts"] = out["step_budget"] + 1
    return out
