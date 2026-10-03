"""FR-15: run Black Box diagnosis on the Who&When failure-attribution benchmark.

  python -m blackbox.benchmarks.whowhen                     # ensemble + simple baselines (free)
  python -m blackbox.benchmarks.whowhen --judge openai      # + GPT-6 Luna judge baselines (billed)

Each Who&When case (a multi-agent conversation with an annotated mistake step and
agent) becomes a Black Box trace. Our models were trained on our own sandbox
agent, so this measures transfer to an unseen, much longer trace format.
Reference numbers from the paper are listed separately with their source.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random

import numpy as np

from ..config import ROOT
from ..models.train import rank_steps, ranking_metrics

logger = logging.getLogger("blackbox.whowhen")
REPO = "Kevin355/Who_and_When"
SUBSETS = ("Hand-Crafted", "Algorithm-Generated")
PAPER_REFERENCE = {
    "source": "Zhang et al., 'Which Agent Causes Task Failures and When?' ICML 2025 (arXiv:2505.00212)",
    "note": "Prompted-LLM step-level accuracy reported in the paper is roughly 8-25% depending on method/model; "
            "agent-level is up to ~54%. Different protocol details apply; not a like-for-like comparison.",
}


def _node_type(name: str) -> str:
    lowered = name.lower()
    if "orchestrator" in lowered or "planner" in lowered:
        return "planner" if "thought" in lowered or "plan" in lowered else "router"
    if any(k in lowered for k in ("surfer", "terminal", "computer", "executor", "tool")):
        return "tool"
    if lowered in ("human", "user"):
        return "other"
    return "reasoner"


def _agent(name: str) -> str:
    return name.split("(")[0].strip() if "->" not in name else "Orchestrator"


def load_cases() -> list[dict]:
    import duckdb
    from huggingface_hub import hf_hub_download
    cases = []
    for subset in SUBSETS:
        path = hf_hub_download(REPO, f"{subset}.parquet", repo_type="dataset", cache_dir=str(ROOT / "data" / "cache" / "hf"))
        connection = duckdb.connect()
        rows = connection.execute(f"SELECT * FROM read_parquet('{path}')").fetchall()
        columns = [d[0] for d in connection.description]
        for i, row in enumerate(rows):
            record = dict(zip(columns, row))
            history = record["history"]
            history = json.loads(history) if isinstance(history, str) else list(history)
            try:
                mistake = int(record["mistake_step"])
            except (TypeError, ValueError):
                continue
            if not 0 <= mistake < len(history):
                continue
            steps = []
            for j, entry in enumerate(history):
                name = str(entry.get("name") or entry.get("role") or "agent")
                steps.append({"step_id": j + 1, "node_name": name, "node_type": _node_type(name),
                              "parent_step_ids": [j] if j else [], "input": {},
                              "output": {"message": str(entry.get("content", ""))[:4000]},
                              "state_before": {}, "state_after": {}, "tool_error": False, "latency_ms": 0,
                              "tokens_in": 0, "tokens_out": 0, "retries": 0, "agent": _agent(name)})
            cases.append({"run_id": f"WW-{subset[:4]}-{i}", "task_family": "whowhen", "prompt": record.get("question", ""),
                          "created_at": "2025-01-01T00:00:00+00:00", "frozen_at": "2025-01-01T00:00:00+00:00",
                          "steps": steps, "success": False, "label_step": mistake + 1, "subset": subset,
                          "mistake_agent": record.get("mistake_agent"), "final_answer": None,
                          "gold_answer": record.get("groundtruth")})
    return cases


def _agent_accuracy(cases, rankings):
    hits = [c["steps"][r[0] - 1]["agent"].split()[0] == str(c["mistake_agent"]).split()[0]
            for c, r in zip(cases, rankings) if r]
    return float(np.mean(hits)) if hits else None


def _cross_validated(cases, encoded, seed=42, folds=5) -> dict:
    """In-domain step ranker: LightGBM on the 412-d step vectors + relative position."""
    import lightgbm as lgb
    order = list(range(len(cases)))
    random.Random(seed).shuffle(order)
    rankings = [None] * len(cases)

    def rows(index):
        matrix = encoded[index]["matrix"]
        position = np.arange(len(matrix))[:, None] / max(1, len(matrix) - 1)
        return np.concatenate([matrix, position, np.full_like(position, len(matrix) / 100)], axis=1)

    for fold in range(folds):
        test_idx = order[fold::folds]
        train_idx = [i for i in order if i not in set(test_idx)]
        x = np.concatenate([rows(i) for i in train_idx])
        y = np.concatenate([[float(s["step_id"] == cases[i]["label_step"]) for s in cases[i]["steps"]] for i in train_idx])
        booster = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "min_data_in_leaf": 20,
                             "feature_fraction": 0.3, "scale_pos_weight": float((len(y) - y.sum()) / max(1, y.sum())),
                             "verbose": -1, "seed": seed}, lgb.Dataset(x, y), num_boost_round=200)
        for i in test_idx:
            rankings[i] = rank_steps(cases[i], booster.predict(rows(i)))
    return {**ranking_metrics(cases, rankings), "agent_accuracy": _agent_accuracy(cases, rankings)}


def evaluate(cases, judge=None, judge_limit=None, seed=42) -> dict:
    from ..features.semantic import shared_encoder, build_matrices
    from ..models.ensemble import ModelOutputs, blend, load_ensemble
    model = load_ensemble()
    if model is None:
        raise RuntimeError("Train models first: python -m blackbox.models.train")
    encoded = build_matrices(cases, shared_encoder(), model.stats)
    names = [[s["node_name"] for s in c["steps"]] for c in cases]
    base = model.outputs
    # The Transformer's positional table covers 24 steps; longer traces use M2+M3.
    short = ModelOutputs(base.transformer, base.booster, base.autoencoder, (base.mean, base.std), base.ae_medians, base.device)
    long = ModelOutputs(None, base.booster, base.autoencoder, (base.mean, base.std), base.ae_medians, base.device)
    items = []
    for case, enc, name in zip(cases, encoded, names):
        runner = short if len(case["steps"]) <= 24 else long
        items.append(runner.compute([enc], [name])[0])
    rankings = {"ensemble": [rank_steps(c, blend(i, model.weights)) for c, i in zip(cases, items)],
                "m2_lightgbm": [rank_steps(c, i["m2"]) for c, i in zip(cases, items)],
                "m3_autoencoder": [rank_steps(c, i["m3"]) for c, i in zip(cases, items)]}
    rng = random.Random(seed)
    rankings["random"] = [rng.sample([s["step_id"] for s in c["steps"]], len(c["steps"])) for c in cases]
    rankings["first_step"] = [[s["step_id"] for s in c["steps"]] for c in cases]
    rankings["last_step"] = [[s["step_id"] for s in reversed(c["steps"])] for c in cases]
    report = {}
    for name, ranks in rankings.items():
        report[name] = {**ranking_metrics(cases, ranks), "agent_accuracy": _agent_accuracy(cases, ranks)}
    report["in_domain_lightgbm_5fold_cv"] = {**_cross_validated(cases, encoded, seed), "note":
        "Trained on Who&When itself with 5-fold cross-validation by case (no case in its own training fold)."}
    if judge:
        from ..judge import judge_all_at_once
        from ..llm import get_llm
        llm = get_llm(judge)
        subset = cases[:judge_limit] if judge_limit else cases
        ranks = [judge_all_at_once(c, llm) for c in subset]
        report[f"llm_all_at_once ({llm.model})"] = {**ranking_metrics(subset, ranks),
                                                     "agent_accuracy": _agent_accuracy(subset, ranks)}
    by_subset = {}
    for subset in SUBSETS:
        idx = [i for i, c in enumerate(cases) if c["subset"] == subset]
        by_subset[subset] = {"count": len(idx),
                             "ensemble_top1": ranking_metrics([cases[i] for i in idx], [rankings["ensemble"][i] for i in idx])["top1"],
                             "random_top1": ranking_metrics([cases[i] for i in idx], [rankings["random"][i] for i in idx])["top1"]}
    return {"benchmark": "Who&When", "generated_at": datetime.now(timezone.utc).isoformat(), "cases": len(cases),
            "mean_steps": float(np.mean([len(c["steps"]) for c in cases])), "results": report, "by_subset": by_subset,
            "paper_reference": PAPER_REFERENCE, "model_version": model.version,
            "notes": "Zero-shot transfer: no Who&When data was used for training. Traces longer than 24 steps "
                     "are scored by M2+M3 only (Transformer positional limit)."}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", choices=["openai", "ollama"], default=None)
    parser.add_argument("--judge-limit", type=int, default=None)
    parser.add_argument("--output", default=str(ROOT / "data" / "benchmark_whowhen.json"))
    parser.add_argument("--metrics", default=str(ROOT / "data" / "metrics.json"))
    args = parser.parse_args()
    cases = load_cases()
    logger.info("Loaded %d Who&When cases", len(cases))
    report = evaluate(cases, args.judge, args.judge_limit)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    metrics_path = Path(args.metrics)
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics.setdefault("public_benchmarks", {})["who_and_when"] = report
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: {m: v[m] for m in ("count", "top1", "top3", "agent_accuracy")} for k, v in report["results"].items()}, indent=2))


if __name__ == "__main__":
    main()
