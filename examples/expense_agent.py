"""A stranger's agent, wrapped with two lines of Black Box.

A travel-expense agent written with stock LangGraph (not one of Black Box's
built-in tasks): it parses a request, fetches an FX quote and a hotel price in
parallel, adds them up with a reducer-based log, and answers. It runs offline;
pass --llm ollama to let a local qwen2.5:3b parse the request instead.

  python examples/expense_agent.py            # records into data/traces.db
  python examples/expense_agent.py --llm ollama

Scenes:
  1. clean run                      -> passes the independent check
  2. FX tool serves a stale quote   -> silent wrong total; Black Box points at fetch_fx_rate
  3. fork at fetch_fx_rate with a fresh quote -> LangGraph re-runs what follows; passes
  4. hotel API crashes              -> ERROR incident with its last good checkpoint
  5. resume after the API recovers  -> completes from the checkpoint
"""
from __future__ import annotations

import argparse
import json
import operator
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

import blackbox  # noqa: E402

TODAY = datetime(2026, 10, 3, tzinfo=timezone.utc)
FRESH_EUR_USD = 1.087
HOTEL_PER_NIGHT = {"lisbon": 132.0, "berlin": 148.0, "paris": 189.0}
WORLD = {"stale_fx": False, "hotel_down": False, "fault": None}  # fault: name of a node to corrupt (benchmarks)


class Trip(TypedDict, total=False):
    request: str
    city: str
    nights: int
    meals_eur: float
    fx: dict
    hotel_usd: float
    total_usd: float
    answer: float
    log: Annotated[list, operator.add]


def parse_request(state: Trip, llm=None):
    text = state["request"].lower()
    if WORLD["fault"] == "parse_request":  # misreads the number of nights
        city = next(c for c in HOTEL_PER_NIGHT if c in text)
        nights = int(re.search(r"(\d+)\s*nights?", text).group(1)) + 2
        meals = float(re.search(r"(?:eur|€)\s*([\d.]+)", text).group(1))
        return {"city": city, "nights": nights, "meals_eur": meals, "log": ["parsed"]}
    if llm is not None:
        reply = llm.invoke("Extract JSON {city, nights, meals_eur} from: " + state["request"]).content
        data = json.loads(re.search(r"\{.*\}", reply, re.S).group(0))
        city, nights, meals = str(data["city"]).lower(), int(data["nights"]), float(data["meals_eur"])
    else:
        city = next(c for c in HOTEL_PER_NIGHT if c in text)
        nights = int(re.search(r"(\d+)\s*nights?", text).group(1))
        meals = float(re.search(r"(?:eur|€)\s*([\d.]+)", text).group(1))
    return {"city": city, "nights": nights, "meals_eur": meals, "log": ["parsed"]}


def fetch_fx_rate(state: Trip):
    if WORLD["stale_fx"] or WORLD["fault"] == "fetch_fx_rate":  # a provider cache serving last year's quote
        return {"fx": {"pair": "EUR/USD", "rate": 0.978, "as_of": (TODAY - timedelta(days=400)).isoformat()},
                "log": ["fx"]}
    return {"fx": {"pair": "EUR/USD", "rate": FRESH_EUR_USD, "as_of": TODAY.isoformat()}, "log": ["fx"]}


def fetch_hotel_price(state: Trip):
    if WORLD["hotel_down"]:
        raise ConnectionError("hotel pricing API returned 503")
    price = HOTEL_PER_NIGHT[state["city"]] * (1.45 if WORLD["fault"] == "fetch_hotel_price" else 1.0)
    return {"hotel_usd": round(price * state["nights"], 2), "log": ["hotel"]}


def compute_total(state: Trip):
    if WORLD["fault"] == "compute_total":  # forgets to convert the meals from EUR
        return {"total_usd": round(state["meals_eur"] + state["hotel_usd"], 2), "log": ["total"]}
    return {"total_usd": round(state["meals_eur"] * state["fx"]["rate"] + state["hotel_usd"], 2), "log": ["total"]}


def final_answer(state: Trip):
    if WORLD["fault"] == "final_answer":  # rounds the total away
        return {"answer": float(round(state["total_usd"], -1)), "log": ["answer"]}
    return {"answer": state["total_usd"], "log": ["answer"]}


def build(llm=None):
    graph = StateGraph(Trip)
    graph.add_node("parse_request", lambda s: parse_request(s, llm))
    graph.add_node("fetch_fx_rate", fetch_fx_rate)
    graph.add_node("fetch_hotel_price", fetch_hotel_price)
    graph.add_node("compute_total", compute_total)
    graph.add_node("final_answer", final_answer)
    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "fetch_fx_rate")
    graph.add_edge("parse_request", "fetch_hotel_price")
    graph.add_edge(["fetch_fx_rate", "fetch_hotel_price"], "compute_total")
    graph.add_edge("compute_total", "final_answer")
    graph.add_edge("final_answer", END)
    return graph.compile(checkpointer=InMemorySaver())


def expected_total(request: str) -> float:
    """Independent check, computed without the agent."""
    text = request.lower()
    city = next(c for c in HOTEL_PER_NIGHT if c in text)
    nights = int(re.search(r"(\d+)\s*nights?", text).group(1))
    meals = float(re.search(r"(?:eur|€)\s*([\d.]+)", text).group(1))
    return round(meals * FRESH_EUR_USD + HOTEL_PER_NIGHT[city] * nights, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--llm", choices=["offline", "ollama"], default="offline")
    parser.add_argument("--db", default="data/traces.db")
    args = parser.parse_args()
    llm = None
    if args.llm == "ollama":
        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="qwen2.5:3b", format="json", temperature=0)
    request = "Expense report: 3 nights in Lisbon plus EUR 240 of meals. Total in USD?"
    gold = expected_total(request)

    # --- the two Black Box lines -------------------------------------------
    agent = blackbox.wrap(build(llm), args.db, name="expense-agent", check=lambda s: abs(s.get("answer", -1) - gold) < 0.01)
    # -----------------------------------------------------------------------
    print("capabilities:", agent.capabilities)

    clean = agent.invoke({"request": request, "log": []})
    print(f"1. clean run      {clean['run_id']}  {clean['status']}  answer={clean['final_answer']} (expected {gold})")

    WORLD["stale_fx"] = True
    broken = agent.invoke({"request": request, "log": []})
    WORLD["stale_fx"] = False
    print(f"2. stale FX quote {broken['run_id']}  {broken['status']}  answer={broken['final_answer']}")
    diagnosis = blackbox.diagnose(broken)
    root = diagnosis["root_cause"]
    print(f"   leading suspect: step {root['step']} {root['node']} (ranking score {root['confidence']:.0%})")

    fx_step = next(s for s in broken["steps"] if s["node_name"] == "fetch_fx_rate")
    fresh = {"fx": {"pair": "EUR/USD", "rate": FRESH_EUR_USD, "as_of": TODAY.isoformat()}}
    forked = agent.fork(broken["run_id"], fx_step["step_id"], fresh)
    print(f"3. fork at step {fx_step['step_id']}  {forked['run_id']}  {forked['status']}  answer={forked['final_answer']}"
          f"  (re-ran: {[s['node_name'] for s in forked['steps'] if s['action'] == 'rerun']})")

    WORLD["hotel_down"] = True
    crashed = agent.invoke({"request": request, "log": []})
    print(f"4. hotel API down {crashed['run_id']}  {crashed['status']}  {crashed.get('error')}"
          f"  last checkpoint kept: {bool(crashed['last_checkpoint']['checkpoint_id'])}")
    WORLD["hotel_down"] = False
    resumed = agent.resume(crashed["run_id"])
    print(f"5. resumed        {resumed['run_id']}  {resumed['status']}  answer={resumed['final_answer']}")
    print("Open http://localhost:8010 to inspect these runs (restart the server or refresh the Runs page).")


if __name__ == "__main__":
    main()
