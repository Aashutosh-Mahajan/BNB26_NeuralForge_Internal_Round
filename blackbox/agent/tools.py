"""Deterministic, side-effect-free mocked tools with explicit time."""
import math
import sqlite3
from datetime import datetime, timezone, timedelta
from functools import lru_cache

REGIONS = ("north", "south", "east", "west")
QUARTERS = ("Q1", "Q2", "Q3", "Q4")
PRODUCTS = ("laptops", "phones", "monitors")
POLICIES = ("returns", "warranty", "cancellation", "exchange", "price_match", "damage_claim")
CATEGORIES = ("electronics", "apparel", "furniture", "groceries", "books", "toys")


def parse_time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _inr_usd(at):
    """A time-varying rate series; values a year apart differ by ~9%."""
    day = (at.date() - datetime(2026, 10, 3).date()).days
    return round(0.0119 + (day % 31) * 0.000001 + day * 0.000003, 6)


def currency_rate(frm, to, at_time):
    at = parse_time(at_time)
    daily = _inr_usd(at)
    rates = {("INR", "USD"): daily, ("USD", "INR"): 1 / daily, ("USD", "EUR"): 0.92, ("EUR", "USD"): 1 / 0.92}
    if frm == to:
        rate = 1.0
    elif (frm, to) in rates:
        rate = rates[(frm, to)]
    else:
        raise ValueError(f"Unsupported sandbox currency pair: {frm}/{to}")
    return {"rate": rate, "backup_rate": rate, "as_of": at.isoformat(), "source": "sandbox-market",
            "backup_source": "sandbox-backup-feed"}


def backup_currency_rate(frm, to, at_time):
    """The second, independent source used by the fix suggester."""
    result = currency_rate(frm, to, at_time)
    return {**result, "source": "sandbox-backup-feed"}


def calculator(principal, months, annual_rate, method="amortized"):
    principal, months, annual_rate = float(principal), int(months), float(annual_rate)
    if months < 1 or not all(math.isfinite(v) for v in (principal, annual_rate)):
        raise ValueError("A finite principal/rate and positive term are required")
    rate = annual_rate / 1200
    if method == "simple_interest":
        answer = principal * (1 + rate * months) / months
    elif rate == 0:
        answer = principal / months
    else:
        answer = principal * rate * (1 + rate) ** months / ((1 + rate) ** months - 1)
    return round(answer, 2)


@lru_cache(maxsize=1)
def _sales_rows():
    rows = []
    for r, region in enumerate(REGIONS):
        for q, quarter in enumerate(QUARTERS):
            for p, product in enumerate(PRODUCTS):
                units = 3 + (r * 7 + q * 5 + p * 3) % 17
                price = (120, 80, 45)[p] + r * 5
                rows.append((region, quarter, product, units * price, units))
    return rows


def sql_query(region, metric="revenue", quarter="all", column=None):
    column = column or metric
    if column not in ("revenue", "units"):
        raise ValueError("Unknown report column")
    region = str(region).lower()
    quarter = str(quarter or "all").upper()
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE sales (region TEXT, quarter TEXT, product TEXT, revenue REAL, units INTEGER)")
        conn.executemany("INSERT INTO sales VALUES (?, ?, ?, ?, ?)", _sales_rows())
        query = f"SELECT COALESCE(SUM({column}),0) FROM sales WHERE region=?"
        parameters = [region]
        if quarter != "ALL":
            query += " AND quarter=?"
            parameters.append(quarter)
        value = conn.execute(query, parameters).fetchone()[0]
    sql = f"SELECT SUM({column}) AS total FROM sales WHERE region = ?" + (" AND quarter = ?" if quarter != "ALL" else "")
    return {"rows": [{"total": value}], "row_count": 1, "query": sql, "parameters": parameters}


def schema_lookup():
    return {"table": "sales", "columns": ["region", "quarter", "product", "revenue", "units"],
            "metrics": {"revenue": "revenue", "units": "units"}, "aggregate_column": "revenue"}


def policy_days(policy, category, version="current"):
    p, c = POLICIES.index(policy), CATEGORIES.index(category)
    current = (30, 365, 14, 21, 10, 7)[p] + ((p + 2 * c) % 4) * 5
    return current if version == "current" else current + (15 if (p + c) % 2 else -5)


def _documents():
    docs = []
    for policy in POLICIES:
        for category in CATEGORIES:
            for version, effective in (("current", "2026-04-01T00:00:00+00:00"), ("archived", "2024-02-01T00:00:00+00:00")):
                days = policy_days(policy, category, version)
                label = policy.replace("_", " ")
                docs.append({"id": f"{policy}-{category}-{version}", "policy": policy, "category": category,
                             "version": version, "effective_date": effective, "days": days,
                             "title": f"{category.title()} {label} policy ({version})",
                             "text": f"{category.title()} {label} policy. Customers may submit a {label} request for "
                                     f"{category} items within {days} days of delivery."})
    return docs


def doc_search(policy, category, at_time, query=None):
    """Keyword retrieval over the policy corpus; current documents rank first."""
    terms = set(str(query or f"{policy} {category}").lower().replace("_", " ").split())
    scored = []
    for doc in _documents():
        if parse_time(doc["effective_date"]) > parse_time(at_time):
            continue
        overlap = len(terms & set(doc["text"].lower().replace(".", "").split()))
        exact = (doc["policy"] == policy) * 2.0 + (doc["category"] == category) * 1.5
        score = round(min(0.99, 0.2 + 0.08 * overlap + 0.12 * exact + (0.05 if doc["version"] == "current" else 0)), 4)
        scored.append((score, doc))
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    top = scored[:3]
    return {"documents": [dict(doc) for _, doc in top], "scores": [score for score, _ in top], "as_of": at_time}


def date_util(at_time, months=0):
    at = parse_time(at_time)
    return {"date": (at + timedelta(days=int(months) * 30)).date().isoformat(), "term_months": int(months)}
