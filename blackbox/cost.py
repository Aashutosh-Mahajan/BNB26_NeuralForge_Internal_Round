"""Project GPT-6 Luna spend for the full PRD pipeline from measured prompt sizes.

  python -m blackbox.cost --dataset data/dataset_sandbox.db

Prompt sizes come from the recorded sandbox dataset (the same prompts GPT-6 Luna
would receive). Output tokens are assumed: visible JSON plus hidden reasoning
tokens (billed as output) at low reasoning effort.
"""
from __future__ import annotations

import argparse
import json
import statistics

from .config import ROOT, settings
from .judge import serialize
from .models.train import has_label, load_runs

# Reasoning-token scenarios per call at reasoning_effort=low (billed as output).
SCENARIOS = {"low": 64, "typical": 192, "high": 512}
TOKENIZER_FACTOR = 1.25  # chars/4 under-counts JSON-heavy prompts


def project(dataset: str, tasks: int = 800, natural_to_label: int = 50, judge_runs: int = 100,
            whowhen_cases: int = 181) -> dict:
    runs = load_runs([dataset])
    llm_steps = [s for r in runs for s in r["steps"] if s.get("llm_call")]
    avg_in = statistics.mean(s["tokens_in"] for s in llm_steps) * TOKENIZER_FACTOR
    avg_visible_out = statistics.mean(s["tokens_out"] for s in llm_steps) * TOKENIZER_FACTOR
    clean_calls = 4 * tasks
    # Injections re-run only downstream LLM steps whose inputs changed (measured).
    injected = [r for r in runs if r.get("dataset_role") == "injected"]
    calls_per_injection = statistics.mean(sum(1 for s in r["steps"] if s.get("llm_call") and s.get("billed_tokens", 0) > 0)
                                          for r in injected) if injected else 2
    injection_calls = calls_per_injection * len(injected) * tasks / max(1, len({r["task_id"] for r in runs}))
    labeler_calls = natural_to_label * 3 * (1 + 3 * 2)        # ~3 fixes x (fixer + K=3 x 2 downstream)
    replay_eval_calls = 40 * 3 * 2
    failed = [r for r in runs if r.get("split") == "test" and has_label(r)][:judge_runs]
    judge_in = statistics.mean(len(serialize(r)) / 4 * TOKENIZER_FACTOR for r in failed) if failed else 1500
    judge_steps = statistics.mean(r["label_step"] for r in failed) if failed else 5
    rows = {
        "agent_clean_runs": (clean_calls, avg_in),
        "agent_injection_reruns": (injection_calls, avg_in),
        "counterfactual_labeling": (labeler_calls, avg_in * 1.6),
        "evaluation_replays": (replay_eval_calls, avg_in),
        "judge_all_at_once": (judge_runs, judge_in),
        "judge_step_by_step": (judge_runs * judge_steps, judge_in * 0.6),
        "whowhen_judge (optional)": (whowhen_cases, 13000),
    }
    cfg = settings()
    report = {"pricing_per_million": {"input": cfg.price_input, "cached_input": cfg.price_cached_input,
                                      "output": cfg.price_output},
              "measured": {"avg_input_tokens_per_agent_call": round(avg_in), "avg_visible_output_tokens": round(avg_visible_out),
                           "llm_calls_per_injection": round(calls_per_injection, 2), "judge_prompt_tokens": round(judge_in)},
              "rows": {}, "totals": {}}
    for scenario, reasoning in SCENARIOS.items():
        total = 0.0
        for name, (calls, tokens_in) in rows.items():
            out = (avg_visible_out if not name.startswith("judge") else 40) + reasoning
            cost = cfg.price_usd(int(calls * tokens_in), int(calls * out))
            report["rows"].setdefault(name, {"calls": round(calls), "input_tokens": round(calls * tokens_in)})
            report["rows"][name][f"usd_{scenario}"] = round(cost, 3)
            total += cost
        report["totals"][scenario] = round(total, 2)
    report["note"] = ("Prompt caching (cached input at $0.01/M) would lower input cost further; it is not assumed. "
                      "Set LLM_BUDGET_USD in .env as a hard cap.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "dataset_sandbox.db"))
    parser.add_argument("--tasks", type=int, default=800)
    args = parser.parse_args()
    print(json.dumps(project(args.dataset, args.tasks), indent=2))


if __name__ == "__main__":
    main()
