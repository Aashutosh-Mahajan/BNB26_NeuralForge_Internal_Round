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
    documents = output.get("documents")
    if isinstance(documents, list) and documents and isinstance(documents[0], dict) and "days" in documents[0]:
        documents[0]["days"] = documents[0]["days"] + 15
        documents[0]["text"] = documents[0].get("text", "").replace(f" {documents[0]['days'] - 15} days", f" {documents[0]['days']} days")
        return output
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
        metrics = output.get("metrics") or {}
        output["metrics"] = {name: ("units" if column == "revenue" else "revenue") for name, column in metrics.items()}
        output["aggregate_column"] = "units" if output["aggregate_column"] == "revenue" else "revenue"
        return output
    raise ValueError("This step has no compatible value to corrupt")

def _arguments(args, family):
    if family == "finance":
        args["amount"] = args["amount"] * 10  # thousands vs lakh unit confusion
    elif family == "sql":
        args["region"] = {"north": "south", "south": "north", "east": "west", "west": "east"}[args["region"]]
    elif family == "doc_qa":
        args["category"] = "apparel" if args["category"] != "apparel" else "electronics"
    else:
        args["quantity"] = args["quantity"] * 2
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
            # The archived edition of the same policy outranks the current one.
            docs = out["documents"]
            archived = next((i for i, d in enumerate(docs) if d.get("version") == "archived"), None)
            if archived is None:
                raise ValueError("No archived distractor document is available")
            docs.insert(0, docs.pop(archived))
            out["scores"] = sorted(out.get("scores", []), reverse=True)
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
        family = run["task_family"]
        if family == "sql":
            key, fallback = ("quarter", "all") if constraints.get("quarter") != "all" else ("region", "north")
        else:
            key = {"finance": "months", "doc_qa": "category", "math": "discount_pct"}[family]
            fallback = {"months": 12, "category": "electronics", "discount_pct": 0}[key]
        if constraints.get(key) == fallback:
            fallback = {"months": 36, "category": "apparel", "discount_pct": 25, "region": "west", "quarter": "Q1"}[key]
        constraints[key] = fallback
    elif fault_type == "premature_final":
        out["tool"] = "final_answer"
    elif fault_type == "loop_repetition":
        out["attempts"] = out["step_budget"] + 1
    return out
