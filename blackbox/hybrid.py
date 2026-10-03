"""Hybrid judge (zero API cost): Black Box shortlists its top-3 steps, a local LLM picks one.

  python -m blackbox.hybrid --dataset data/dataset_openai.db --limit 100 --whowhen --output data/hybrid_judge.json

Compared on the same runs: Black Box alone, the local LLM judging the whole trace alone,
and the hybrid. Uses Ollama (LLM_PROVIDER-independent); never calls OpenAI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time

from .config import ROOT, settings
from .judge import ALL_AT_ONCE, serialize
from .models.train import has_label, ranking_metrics

logger = logging.getLogger("blackbox.hybrid")
SHORTLIST = ("You are an expert debugger of AI agent execution traces. The run below FAILED: its final answer is "
             "wrong. An anomaly model has shortlisted candidate steps, most likely first: {candidates}. "
             "Choose the ONE candidate that is the decisive root-cause mistake (not a step that merely used a bad "
             "value). Respond with JSON only: {{\"step\": <one of the candidate step ids>, \"reason\": \"<short>\"}}")


def _step(result, allowed=None):
    data = result.data if isinstance(result.data, dict) else {}
    try:
        step = int(data.get("step"))
    except (TypeError, ValueError):
        return None
    return step if allowed is None or step in allowed else None


def _ranking(run, first, fallback):
    ids = [s["step_id"] for s in run["steps"]]
    ordered = [first] if first in ids else []
    return ordered + [i for i in fallback if i not in ordered] + [i for i in ids if i not in ordered and i not in fallback]


class _Failed:
    data, error = None, "local LLM call failed"


def _call(llm, system, user, **kwargs):
    """Local model calls can drop (Ollama restarts); retry with backoff, then count as no answer."""
    for attempt in range(4):
        try:
            return llm.complete_json(system, user, **kwargs)
        except Exception as exc:
            logger.warning("local judge call failed (%s); retrying", exc)
            time.sleep(5 * (attempt + 1))
    return _Failed()


def judge_runs(runs, blackbox_rankings, llm, k=3):
    """Returns rankings for llm_alone and hybrid, plus call statistics."""
    alone, hybrid, latency, invalid = [], [], [], 0
    for i, (run, ranked) in enumerate(zip(runs, blackbox_rankings), 1):
        trace = serialize(run)
        start = time.perf_counter()
        solo = _call(llm, ALL_AT_ONCE, trace, seed=3, purpose="local_judge_all_at_once", run_id=run["run_id"])
        latency.append((time.perf_counter() - start) * 1000)
        alone.append(_ranking(run, _step(solo), []))
        candidates = ranked[:k]
        pick = _call(llm, SHORTLIST.format(candidates=", ".join(map(str, candidates))), trace, seed=3,
                     purpose="local_hybrid_judge", run_id=run["run_id"])
        chosen = _step(pick, set(candidates))
        invalid += chosen is None
        hybrid.append(_ranking(run, chosen if chosen is not None else candidates[0], candidates))
        if i % 20 == 0:
            logger.info("judged %d/%d", i, len(runs))
    return alone, hybrid, {"mean_call_ms": sum(latency) / max(1, len(latency)), "hybrid_invalid_picks": invalid}


def _report(runs, bb, alone, hybrid, stats, llm_label):
    return {"runs": len(runs), "judge_model": llm_label,
            "blackbox_alone": ranking_metrics(runs, bb),
            "local_llm_alone": ranking_metrics(runs, alone),
            "hybrid_blackbox_shortlist_plus_llm": ranking_metrics(runs, hybrid), **stats}


def run_hybrid(dataset, limit=100, whowhen=False, seed=42, k=3, save=None):
    from .diagnosis import diagnose_many
    from .llm import get_llm
    from .models.train import by_split, load_runs, rank_steps
    llm = get_llm("ollama")
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "cost_usd": 0.0,
           "protocol": f"Black Box top-{k} shortlist; the local LLM must pick one of them. Same runs for all three methods."}
    runs = [r for r in by_split(load_runs([dataset]))["test"] if has_label(r)]
    random.Random(seed).shuffle(runs)  # Same sample order as the GPT judge baseline in evaluation.py.
    runs = runs[:limit]
    diagnoses = diagnose_many(runs)
    bb = [[s["step"] for s in d["step_scores"]] for d in diagnoses]
    alone, hybrid, stats = judge_runs(runs, bb, llm, k)
    out["gpt6luna_test_split"] = _report(runs, bb, alone, hybrid, stats, llm.label)
    if save:
        save(out)  # keep the first result even if the longer benchmark is interrupted
    if whowhen:
        from .benchmarks.whowhen import load_cases
        from .features.semantic import build_matrices, shared_encoder
        from .models.ensemble import blend, load_ensemble
        model = load_ensemble()
        cases = load_cases()
        items = model.outputs.compute(build_matrices(cases, shared_encoder(), model.stats),
                                      [[s["node_name"] for s in c["steps"]] for c in cases])
        bb_ww = [rank_steps(c, blend(i, model.weights)) for c, i in zip(cases, items)]
        alone, hybrid, stats = judge_runs(cases, bb_ww, llm, k)
        out["who_and_when"] = _report(cases, bb_ww, alone, hybrid, stats, llm.label)
        out["who_and_when"]["context_tokens"] = settings().ollama_num_ctx
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "dataset_openai.db"))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--whowhen", action="store_true")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", default=str(ROOT / "data" / "hybrid_judge.json"))
    args = parser.parse_args()
    if Path(args.output).resolve() == (ROOT / "data" / "metrics.json").resolve():
        parser.error("Refusing to overwrite data/metrics.json")
    save = lambda data: Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
    report = run_hybrid(args.dataset, args.limit, args.whowhen, k=args.k, save=save)
    save(report)
    print(json.dumps({k: {m: v[m]["top1"] for m in ("blackbox_alone", "local_llm_alone", "hybrid_blackbox_shortlist_plus_llm")}
                      for k, v in report.items() if isinstance(v, dict) and "blackbox_alone" in v}, indent=2))


if __name__ == "__main__":
    main()
