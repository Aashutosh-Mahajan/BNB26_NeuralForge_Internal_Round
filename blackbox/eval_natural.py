"""Score natural failures from another agent model with the existing trained models.

  python -m blackbox.eval_natural --dataset data/dataset_ollama.db --output data/metrics_natural.json

No retraining and no API calls. The models never saw this agent model's runs, so this
is also a cross-model generalization result. Never writes data/metrics.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import ROOT
from .evaluation import Scorer, evaluate_split
from .models.ensemble import load_ensemble
from .models.train import has_label, load_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "dataset_ollama.db"))
    parser.add_argument("--output", default=str(ROOT / "data" / "metrics_natural.json"))
    args = parser.parse_args()
    if Path(args.output).resolve() == (ROOT / "data" / "metrics.json").resolve():
        parser.error("Refusing to overwrite data/metrics.json")
    model = load_ensemble()
    runs = load_runs([args.dataset])
    failed = [r for r in runs if r.get("success") is False]
    labelled = [r for r in failed if has_label(r)]
    report = evaluate_split(runs, Scorer(model), 42)
    agent_models = sorted({r.get("model") for r in runs})
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "model_version": model.version,
           "agent_models": agent_models, "runs": len(runs), "natural_failures": len(failed),
           "counterfactually_labelled": len(labelled),
           "failure_rate": len(failed) / len(runs) if runs else None,
           "label_steps": dict(Counter(r["steps"][r["label_step"] - 1]["node_name"] for r in labelled)),
           "top1": report["top1"], "top3": report["top3"], "mrr": report["mrr"], "auroc": report["auroc"],
           "report": report, "cost_usd": 0.0,
           "note": "Natural failures made by the agent model itself (nothing injected), labelled by counterfactual "
                   "replay with a careful-rerun fixer. Scored with the GPT-6 Luna-era models, no retraining."}
    Path(args.output).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("agent_models", "runs", "natural_failures", "counterfactually_labelled",
                                          "failure_rate", "label_steps", "top1", "top3", "auroc")}, indent=2))


if __name__ == "__main__":
    main()
