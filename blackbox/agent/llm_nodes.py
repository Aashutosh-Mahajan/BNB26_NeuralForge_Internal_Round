"""The four LLM nodes (planner, router, reasoner, final answer).

Each node has: messages() for a real provider, parse() to validate a model
response, and sandbox() — a rule-based policy with an optional, seeded
mistake rate that imitates natural LLM failures for data generation.
"""
from __future__ import annotations

import json
import math
import random
import re
from copy import deepcopy

from .tasks import DEFAULTS, PARSERS
from .tools import CATEGORIES, POLICIES, QUARTERS, REGIONS

PRIMARY_TOOL = {"finance": "currency_rate", "sql": "schema_lookup", "doc_qa": "doc_search", "math": "line_items"}
SCHEMAS = {
    "finance": {"amount": "number: loan principal in INR as a plain number ('60k' = 60000, '1.2 lakh' = 120000)",
                "months": "integer: loan term in months (convert years to months)",
                "annual_rate": "number: annual interest rate in percent (9 for 9%)",
                "base_currency": "string: 'INR'", "target_currency": "string: 'USD'"},
    "sql": {"region": f"string: one of {'|'.join(REGIONS)}", "metric": "string: 'revenue' or 'units'",
            "quarter": f"string: one of {'|'.join(QUARTERS)} or 'all'"},
    "doc_qa": {"policy": f"string: one of {'|'.join(POLICIES)}", "category": f"string: one of {'|'.join(CATEGORIES)}"},
    "math": {"quantity": "number: how many items", "unit_price": "number: USD price per item",
             "discount_pct": "number: percent discount (0 if none)"},
}
TOOL_CATALOG = {
    "currency_rate": "currency_rate(base_currency, target_currency): live FX rate with as_of timestamp (finance tasks)",
    "schema_lookup": "schema_lookup(region, metric, quarter): sales schema, then a SQL aggregate (sales reporting)",
    "doc_search": "doc_search(policy, category): retrieves store policy documents (policy questions)",
    "line_items": "line_items(quantity, unit_price, discount_pct): prices order lines (shopping totals)",
    "calculator": "calculator(expression): arithmetic only",
    "final_answer": "final_answer(answer): stop and answer immediately",
}
NUMERIC = {"amount", "months", "annual_rate", "quantity", "unit_price", "discount_pct"}


def _js(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value)
        try:
            number = float(cleaned)
            return int(number) if number.is_integer() else number
        except ValueError:
            return value
    return value


def _coerce(family: str, raw) -> dict:
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key in SCHEMAS[family]:
        if key in raw:
            value = raw[key]
            if key in NUMERIC:
                value = _number(value)
                if key == "months" and isinstance(value, float) and value.is_integer():
                    value = int(value)
            elif isinstance(value, str):
                value = value.strip()
                value = value.upper() if key in ("quarter", "base_currency", "target_currency") and value.lower() != "all" else value.lower()
            out[key] = value
    return out


# --------------------------------------------------------------------- planner
def planner_messages(inputs, family):
    system = (f"You are the planning node of a tool-using {family} agent. Extract every constraint of the user's task.\n"
              "Respond with JSON only, exactly: {\"constraints\": {...}, \"steps\": [\"retrieve\", \"compute\", \"verify\"]}\n"
              "Constraint fields:\n" + "\n".join(f"- {k}: {v}" for k, v in SCHEMAS[family].items()))
    user = f"Task: {inputs['prompt']}"
    if inputs.get("context"):
        user += f"\nAttached task fields: {_js(inputs['context'])}"
    return system, user


def planner_parse(data, inputs, family):
    constraints = _coerce(family, (data or {}).get("constraints", data))
    steps = (data or {}).get("steps") if isinstance(data, dict) else None
    return {"task_family": family, "constraints": constraints,
            "steps": steps if isinstance(steps, list) else ["retrieve", "compute", "verify"]}


def _misread(value, rng):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
        digits = [i for i, c in enumerate(text) if c.isdigit()]
        choice = rng.random()
        if choice < 0.4 and len(digits) >= 2:  # transposed digits
            i = rng.randrange(len(digits) - 1)
            a, b = digits[i], digits[i + 1]
            chars = list(text)
            chars[a], chars[b] = chars[b], chars[a]
            mutated = _number("".join(chars))
            if mutated != value:
                return mutated
        if choice < 0.7:
            return _number(value * rng.choice([10, 0.1]))
        return _number(round(value * rng.choice([0.5, 2, 1.25]), 2))
    return value


def planner_sandbox(inputs, family, rng, noise):
    constraints = {**DEFAULTS[family], **PARSERS[family](inputs["prompt"]), **(inputs.get("context") or {})}
    constraints = {k: v for k, v in constraints.items() if k in SCHEMAS[family]}
    if rng.random() < noise:
        mode = rng.choice(["misread", "misread", "default", "category"])
        numeric = [k for k in constraints if k in NUMERIC]
        if mode == "misread" and numeric:
            key = rng.choice(numeric)
            constraints[key] = _misread(constraints[key], rng)
        elif family == "sql":
            key = rng.choice(["region", "quarter", "metric"])
            options = {"region": REGIONS, "quarter": (*QUARTERS, "all"), "metric": ("revenue", "units")}[key]
            constraints[key] = rng.choice([o for o in options if o != constraints[key]])
        elif family == "doc_qa":
            key = rng.choice(["policy", "category"])
            options = POLICIES if key == "policy" else CATEGORIES
            constraints[key] = rng.choice([o for o in options if o != constraints[key]])
        elif numeric:
            key = rng.choice(numeric)
            constraints[key] = {"months": 12, "annual_rate": 10, "amount": 100000, "quantity": 1,
                                "unit_price": 10, "discount_pct": 0}.get(key, constraints[key])
    if family == "finance" and isinstance(constraints.get("months"), float):
        constraints["months"] = int(round(constraints["months"]))
    return {"task_family": family, "constraints": constraints, "steps": ["retrieve", "compute", "verify"]}


# ---------------------------------------------------------------------- router
def router_messages(inputs, family):
    plan = inputs["dependencies"].get("planner", {})
    system = ("You are the router node of a tool-using agent. Choose the first tool to call.\nTools:\n" +
              "\n".join(f"- {d}" for d in TOOL_CATALOG.values()) +
              "\n\"arguments\" is the shared argument bundle that EVERY later step reads (conversion, "
              "calculator, query), so it must contain ALL plan constraints copied unchanged with the same keys — "
              "not only the ones the first tool needs.\n"
              "Respond with JSON only: {\"tool\": \"<name>\", \"arguments\": {<all plan constraints>}}")
    user = f"Task family: {family}\nPlan: {_js(plan)}"
    return system, user


def router_parse(data, inputs, family):
    data = data if isinstance(data, dict) else {}
    args = data.get("arguments", {})
    if isinstance(args, dict) and isinstance(args.get("constraints"), dict):
        args = {**args["constraints"], **{k: v for k, v in args.items() if k != "constraints"}}
    return {"tool": str(data.get("tool", "")), "arguments": _coerce(family, args) if isinstance(args, dict) else {},
            "attempts": 1, "step_budget": 3}


def router_sandbox(inputs, family, rng, noise):
    plan = inputs["dependencies"].get("planner", {})
    constraints = plan.get("constraints", {}) if isinstance(plan, dict) else {}
    out = {"tool": PRIMARY_TOOL[family], "arguments": deepcopy(constraints) if isinstance(constraints, dict) else {},
           "attempts": 1, "step_budget": 3}
    if rng.random() < noise:
        mode = rng.choice(["tool", "argument", "argument", "loop"])
        numeric = [k for k in out["arguments"] if k in NUMERIC]
        if mode == "tool":
            out["tool"] = rng.choice([t for t in ("doc_search", "calculator", "currency_rate", "schema_lookup") if t != out["tool"]])
        elif mode == "argument" and numeric:
            key = rng.choice(numeric)
            out["arguments"][key] = _misread(out["arguments"][key], rng)
        elif mode == "argument" and family == "sql":
            out["arguments"]["region"] = rng.choice([r for r in REGIONS if r != out["arguments"].get("region")])
        elif mode == "argument" and family == "doc_qa":
            out["arguments"]["category"] = rng.choice([c for c in CATEGORIES if c != out["arguments"].get("category")])
        else:
            out["attempts"] = 4
    return out


# -------------------------------------------------------------------- reasoner
def reasoner_messages(inputs, family):
    system = ("You are the reasoning node. The calculator has computed the result of the task. State the answer, "
              "copying the computed value exactly (same number, no extra rounding or units for numbers; "
              "keep text answers such as '30 days' as given).\n"
              "Respond with JSON only: {\"answer\": <value>, \"rationale\": \"<one short sentence>\"}")
    user = f"Task: {inputs.get('task', '')}\nCalculator output: {_js(inputs['dependencies'].get('calculator'))}"
    return system, user


def reasoner_parse(data, inputs, family):
    data = data if isinstance(data, dict) else {}
    answer = data.get("answer")
    if isinstance(answer, str) and family != "doc_qa":
        answer = _number(answer)
    return {"answer": answer, "rationale": str(data.get("rationale", ""))[:300]}


def reasoner_sandbox(inputs, family, rng, noise):
    calc = inputs["dependencies"].get("calculator", {})
    if not isinstance(calc, dict) or calc.get("value") is None:
        raise ValueError("Required tool result unavailable: value")
    answer = calc["value"]
    if rng.random() < noise:
        if isinstance(answer, (int, float)):
            answer = rng.choice([round(answer), round(answer * 1.1, 2), _misread(answer, rng), round(answer / 12, 2)])
            if isinstance(answer, float) and not math.isfinite(answer):
                answer = 0
        elif isinstance(answer, str) and answer.endswith(" days"):
            answer = f"{int(answer.split()[0]) + rng.choice([-7, 5, 15])} days"
    return {"answer": answer, "rationale": "Computed from the retrieved evidence and recorded task constraints."}


# ----------------------------------------------------------------------- final
def final_messages(inputs, family):
    system = ("You are the final-answer node. Return the value of the reasoner's \"answer\" field unchanged as the "
              "final answer. The answer must be the bare value — a number, or a short text such as '30 days' — "
              "never an object and never including the rationale.\n"
              "Respond with JSON only: {\"answer\": <value>}")
    user = f"Task: {inputs.get('task', '')}\nReasoner output: {_js(inputs['dependencies'].get('reasoner'))}"
    return system, user


def final_parse(data, inputs, family):
    answer = (data if isinstance(data, dict) else {}).get("answer")
    if isinstance(answer, str) and family != "doc_qa":
        answer = _number(answer)
    return {"answer": answer}


def final_sandbox(inputs, family, rng, noise):
    reasoner = inputs["dependencies"].get("reasoner", {})
    if not isinstance(reasoner, dict) or reasoner.get("answer") is None:
        raise ValueError("Required tool result unavailable: answer")
    answer = reasoner["answer"]
    if rng.random() < noise and isinstance(answer, (int, float)):
        answer = rng.choice([round(answer), _misread(answer, rng)])
    return {"answer": answer}


NODES = {
    "planner": (planner_messages, planner_parse, planner_sandbox),
    "router": (router_messages, router_parse, router_sandbox),
    "reasoner": (reasoner_messages, reasoner_parse, reasoner_sandbox),
    "final_answer": (final_messages, final_parse, final_sandbox),
}


def node_rng(seed: int, node: str, variant: int = 0) -> random.Random:
    return random.Random(f"{seed}:{node}:{variant}")
