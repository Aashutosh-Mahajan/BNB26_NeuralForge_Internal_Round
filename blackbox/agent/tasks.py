"""Task parameters and independent sandbox outcome verification."""
import hashlib
import math
import re
from .tools import calculator, currency_rate, sql_query
from ..storage import canonical
FAMILIES = ("finance", "sql", "doc_qa", "math")

def _number(pattern, prompt, default):
    match = re.search(pattern, prompt, re.I)
    value = float(match.group(1).replace(",", "")) if match else default
    if not _finite_number(value):
        raise ValueError("Task prompt numbers must be finite")
    return value

def _finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False

def parameters(prompt, family, overrides=None):
    if family not in FAMILIES:
        raise ValueError(f"Unknown task family: {family}")
    defaults = {
        "finance": {"amount": _number(r"(?:₹|INR\s*|Rs\.?\s*)([\d,]+(?:\.\d+)?)", prompt, 50000), "months": int(_number(r"(\d+)\s*months?", prompt, 12)), "annual_rate": _number(r"(\d+(?:\.\d+)?)\s*%", prompt, 9), "base_currency": "INR", "target_currency": "USD"},
        "sql": {"region": next((r for r in ("north", "south", "east", "west") if r in prompt.lower()), "north"), "scale": 1},
        "doc_qa": {"policy": "returns", "days": 30, "version": "current"},
        "math": {"quantity": _number(r"(?:buy\s+|purchase\s+|quantity\s*)(\d+(?:\.\d+)?)", prompt, 8), "unit_price": _number(r"(?:\$|price\s*)(\d+(?:\.\d+)?)", prompt, 12), "discount_pct": _number(r"(\d+(?:\.\d+)?)\s*%", prompt, 10)}
    }[family]
    defaults.update(overrides or {})
    defaults.setdefault("template_id", f"{family}-default")
    text_names = {"finance": ("base_currency", "target_currency"), "sql": ("region",), "doc_qa": ("policy", "version"), "math": ()}[family]
    for key in ("template_id", *text_names):
        if not isinstance(defaults[key], str) or not defaults[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    names = {"finance": ("amount", "months", "annual_rate"), "sql": ("scale",), "doc_qa": ("days",), "math": ("quantity", "unit_price", "discount_pct")}[family]
    for key in names:
        value = defaults[key]
        if not _finite_number(value):
            raise ValueError(f"{key} must be a finite number")
    if family == "finance" and (defaults["months"] < 1 or defaults["months"] != int(defaults["months"]) or defaults["annual_rate"] < 0 or defaults["amount"] <= 0):
        raise ValueError("Finance requires amount > 0, whole months > 0, annual_rate >= 0")
    if family == "math" and not 0 <= defaults["discount_pct"] <= 100:
        raise ValueError("discount_pct must be between 0 and 100")
    if family == "doc_qa" and (defaults["days"] < 1 or defaults["days"] != int(defaults["days"])):
        raise ValueError("days must be a positive integer")
    return defaults

def task_id(prompt, family, params):
    return "T-" + hashlib.sha256(canonical({"prompt": prompt, "family": family, "params": params}).encode()).hexdigest()[:16]

def gold_answer(family, params, frozen_at):
    if family == "finance":
        rate = currency_rate(params["base_currency"], params["target_currency"], frozen_at)["rate"]
        return calculator(params["amount"] * rate, params["months"], params["annual_rate"])
    if family == "sql":
        return sql_query(params["region"], params["scale"])["rows"][0]["total"]
    if family == "doc_qa":
        return f"{params['days']} days"
    return round(params["quantity"] * params["unit_price"] * (1 - params["discount_pct"] / 100), 2)

def verify(answer, gold):
    if isinstance(gold, (int, float)):
        return _finite_number(answer) and abs(answer - gold) <= 0.005
    return str(answer).strip().lower() == str(gold).strip().lower()
