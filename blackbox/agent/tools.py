"""Deterministic local tools with explicit time."""
import math
import sqlite3
from datetime import datetime, timezone, timedelta

def parse_time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)

def currency_rate(frm, to, at_time):
    at = parse_time(at_time)
    day = (at.date() - datetime(2026, 10, 3).date()).days
    daily = round(0.0119 + (day % 31) * 0.000001 - (2026 - at.year) * 0.0011, 6)
    rates = {("INR", "USD"): daily, ("USD", "INR"): 1 / daily, ("USD", "EUR"): 0.92, ("EUR", "USD"): 1 / 0.92}
    if frm == to:
        rate = 1.0
    elif (frm, to) in rates:
        rate = rates[(frm, to)]
    else:
        raise ValueError(f"Unsupported sandbox currency pair: {frm}/{to}")
    return {"rate": rate, "backup_rate": rate, "as_of": at.isoformat(), "source": "sandbox-market"}

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

def sql_query(region, scale=1, column="revenue"):
    if column not in ("revenue", "units"):
        raise ValueError("Unknown report column")
    with sqlite3.connect(":memory:") as conn:
        conn.execute("CREATE TABLE sales (region TEXT, revenue REAL, units INTEGER)")
        conn.executemany("INSERT INTO sales VALUES (?, ?, ?)", [("north", 120 * scale, 3), ("north", 180 * scale, 4), ("south", 210 * scale, 5), ("south", 90 * scale, 2), ("west", 450 * scale, 7), ("east", 75 * scale, 1)])
        value = conn.execute(f"SELECT COALESCE(SUM({column}),0) FROM sales WHERE region=?", (region.lower(),)).fetchone()[0]
    return {"rows": [{"total": value}], "row_count": 1, "query": f"SELECT SUM({column}) AS total FROM sales WHERE region = ?", "parameters": [region.lower()]}

def schema_lookup():
    return {"table": "sales", "columns": ["region", "revenue", "units"], "aggregate_column": "revenue"}

def doc_search(policy, days, at_time):
    return {"documents": [{"id": f"{policy}-current", "title": f"Current {policy} policy", "text": f"The {policy} window is {days} days from purchase.", "days": days, "version": "current", "as_of": at_time}], "scores": [0.96], "as_of": at_time}

def date_util(at_time, months=0):
    at = parse_time(at_time)
    return {"date": (at + timedelta(days=int(months) * 30)).date().isoformat(), "term_months": int(months)}
