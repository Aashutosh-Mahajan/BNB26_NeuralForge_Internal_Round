"""Task parsing, gold answers and independent outcome verification."""
import hashlib
import math
import re

from .tools import CATEGORIES, POLICIES, QUARTERS, REGIONS, calculator, currency_rate, parse_time, policy_days, sql_query
from ..storage import canonical

FAMILIES = ("finance", "sql", "doc_qa", "math")
PARAM_KEYS = {
    "finance": ("amount", "months", "annual_rate", "base_currency", "target_currency"),
    "sql": ("region", "metric", "quarter"),
    "doc_qa": ("policy", "category"),
    "math": ("quantity", "unit_price", "discount_pct"),
}
_PRODUCT_CATEGORY = {
    "laptop": "electronics", "phone": "electronics", "headphone": "electronics", "tablet": "electronics",
    "electronic": "electronics", "camera": "electronics", "shirt": "apparel", "jacket": "apparel",
    "shoe": "apparel", "dress": "apparel", "apparel": "apparel", "clothing": "apparel", "sofa": "furniture",
    "table": "furniture", "chair": "furniture", "furniture": "furniture", "bed": "furniture",
    "grocer": "groceries", "food": "groceries", "produce": "groceries", "book": "books", "novel": "books",
    "textbook": "books", "toy": "toys", "puzzle": "toys", "lego": "toys", "doll": "toys",
}
_POLICY_WORDS = (("price_match", ("price match", "price-match", "price adjustment")),
                 ("damage_claim", ("damage", "damaged", "broken on arrival")),
                 ("cancellation", ("cancel",)), ("exchange", ("exchange", "swap")),
                 ("warranty", ("warranty", "repair", "guarantee")), ("returns", ("return", "refund", "send back")))
_ORDINAL_QUARTER = {"first": "Q1", "second": "Q2", "third": "Q3", "fourth": "Q4", "last": "Q4"}


def _finite_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _num(text):
    value = float(text.replace(",", ""))
    if not math.isfinite(value):
        raise ValueError("Task prompt numbers must be finite")
    return value


def _clean_int(value):
    return int(value) if _finite_number(value) and float(value).is_integer() else value


def parse_finance(prompt):
    text = prompt.replace("₹", " INR ")
    out = {}
    money = re.search(r"(?:INR|Rs\.?|rupees?)\s*([\d,]+(?:\.\d+)?)\s*(k|lakhs?|lakh)?\b|([\d,]+(?:\.\d+)?)\s*(k|lakhs?)?\s*(?:INR|rupees)", text, re.I)
    if money:
        number, unit = (money.group(1), money.group(2)) if money.group(1) else (money.group(3), money.group(4))
        amount = _num(number) * {"k": 1000, "lakh": 100000, "lakhs": 100000}.get((unit or "").lower(), 1)
        out["amount"] = _clean_int(round(amount, 2))
    years = re.search(r"(\d+(?:\.\d+)?)[\s-]*years?", text, re.I)
    months = re.search(r"(\d+)[\s-]*months?", text, re.I)
    if months:
        out["months"] = int(_num(months.group(1)))
    elif years:
        out["months"] = int(round(_num(years.group(1)) * 12))
    rate = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if rate:
        out["annual_rate"] = _clean_int(_num(rate.group(1)))
    out.update(base_currency="INR", target_currency="USD")
    return out


def parse_sql(prompt):
    text = prompt.lower()
    out = {"metric": "units" if re.search(r"\bunits?\b|quantity|how many items|items sold", text) else "revenue"}
    region = next((r for r in REGIONS if re.search(rf"\b{r}(?:ern)?\b", text)), None)
    if region:
        out["region"] = region
    quarter = re.search(r"\bq([1-4])\b", text)
    ordinal = re.search(r"\b(first|second|third|fourth|last)[\s-]+quarter\b", text)
    out["quarter"] = f"Q{quarter.group(1)}" if quarter else _ORDINAL_QUARTER[ordinal.group(1)] if ordinal else "all"
    return out


def parse_doc_qa(prompt):
    text = prompt.lower()
    out = {}
    for policy, words in _POLICY_WORDS:
        if any(word in text for word in words):
            out["policy"] = policy
            break
    for word, category in _PRODUCT_CATEGORY.items():
        if re.search(rf"\b{word}", text):
            out["category"] = category
            break
    return out


def parse_math(prompt):
    text = prompt
    out = {}
    price = re.search(r"\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:dollars|usd)\b", text, re.I)
    if price:
        out["unit_price"] = _clean_int(_num(price.group(1) or price.group(2)))
        text = text[:price.start()] + " " + text[price.end():]
    discount = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if discount:
        out["discount_pct"] = _clean_int(_num(discount.group(1)))
        text = text[:discount.start()] + " " + text[discount.end():]
    elif re.search(r"no discount|without (?:a |any )?discount|full price", text, re.I):
        out["discount_pct"] = 0
    quantity = re.search(r"(\d+(?:\.\d+)?)", text)
    if quantity:
        out["quantity"] = _clean_int(_num(quantity.group(1)))
    return out


PARSERS = {"finance": parse_finance, "sql": parse_sql, "doc_qa": parse_doc_qa, "math": parse_math}
DEFAULTS = {
    "finance": {"amount": 50000, "months": 12, "annual_rate": 9, "base_currency": "INR", "target_currency": "USD"},
    "sql": {"region": "north", "metric": "revenue", "quarter": "all"},
    "doc_qa": {"policy": "returns", "category": "electronics"},
    "math": {"quantity": 8, "unit_price": 12, "discount_pct": 10},
}


def validate(family, values):
    names = {"finance": ("amount", "months", "annual_rate"), "math": ("quantity", "unit_price", "discount_pct")}.get(family, ())
    for key in names:
        if not _finite_number(values.get(key)):
            raise ValueError(f"{key} must be a finite number")
    for key in {"finance": ("base_currency", "target_currency"), "sql": ("region", "metric", "quarter"),
                "doc_qa": ("policy", "category"), "math": ()}[family]:
        if not isinstance(values.get(key), str) or not values[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if family == "finance" and (values["months"] < 1 or values["months"] != int(values["months"]) or
                                values["annual_rate"] < 0 or values["amount"] <= 0):
        raise ValueError("Finance requires amount > 0, whole months > 0, annual_rate >= 0")
    if family == "math" and not 0 <= values["discount_pct"] <= 100:
        raise ValueError("discount_pct must be between 0 and 100")
    if family == "sql" and (values["region"] not in REGIONS or values["metric"] not in ("revenue", "units")
                            or values["quarter"] not in (*QUARTERS, "all")):
        raise ValueError("SQL tasks need a known region, metric (revenue|units) and quarter (Q1-Q4|all)")
    if family == "doc_qa" and (values["policy"] not in POLICIES or values["category"] not in CATEGORIES):
        raise ValueError(f"Document QA needs policy in {POLICIES} and category in {CATEGORIES}")


def parameters(prompt, family, overrides=None):
    """The task's ground-truth parameters: parsed prompt values plus explicit overrides."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown task family: {family}")
    values = dict(DEFAULTS[family])
    values.update(PARSERS[family](prompt))
    values.update(overrides or {})
    values.setdefault("template_id", f"{family}-default")
    if not isinstance(values["template_id"], str) or not values["template_id"].strip():
        raise ValueError("template_id must be a non-empty string")
    validate(family, values)
    return values


def task_id(prompt, family, params):
    return "T-" + hashlib.sha256(canonical({"prompt": prompt, "family": family, "params": params}).encode()).hexdigest()[:16]


def gold_answer(family, params, frozen_at):
    if family == "finance":
        rate = currency_rate(params["base_currency"], params["target_currency"], frozen_at)["rate"]
        return calculator(params["amount"] * rate, params["months"], params["annual_rate"])
    if family == "sql":
        return sql_query(params["region"], params["metric"], params["quarter"])["rows"][0]["total"]
    if family == "doc_qa":
        version = "current" if parse_time(frozen_at) >= parse_time("2026-04-01T00:00:00+00:00") else "archived"
        return f"{policy_days(params['policy'], params['category'], version)} days"
    return round(params["quantity"] * params["unit_price"] * (1 - params["discount_pct"] / 100), 2)


def _as_number(answer):
    if _finite_number(answer):
        return float(answer)
    if isinstance(answer, str):
        match = re.fullmatch(r"\s*(?:USD|\$|INR)?\s*(-?[\d,]*\.?\d+)\s*(?:USD|dollars|units)?\s*", answer, re.I)
        if match:
            try:
                value = float(match.group(1).replace(",", ""))
                return value if math.isfinite(value) else None
            except ValueError:
                return None
    return None


def verify(answer, gold):
    if isinstance(gold, (int, float)):
        value = _as_number(answer)
        return value is not None and abs(value - gold) <= 0.005
    if isinstance(gold, str) and gold.endswith(" days"):
        match = re.fullmatch(r"\s*(\d+)\s*(?:days?)?\s*\.?\s*", str(answer), re.I)
        return bool(match) and int(match.group(1)) == int(gold.split()[0])
    return str(answer).strip().lower() == str(gold).strip().lower()
