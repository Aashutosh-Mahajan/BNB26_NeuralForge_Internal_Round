"""Re-label the natural failures of an existing dataset with the current labeller.

  python -m blackbox.labeler.relabel --dataset data/dataset_ollama.db

Replays use the model that produced each run (Ollama runs replay on Ollama, free).
Refuses OpenAI-produced datasets, which would bill API credits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path

from ..engine import Engine
from ..llm.client import SandboxLLM
from .counterfactual import label_run


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()
    engine = Engine(args.dataset, llm=SandboxLLM(0.0))
    runs = [r for r in engine.store.list_runs(limit=10 ** 7) if r.get("dataset_role") == "natural"]
    if any(r.get("llm_provider") == "openai" for r in runs):
        parser.error("This dataset contains OpenAI runs; relabelling would bill API credits")
    for i, run in enumerate(runs, 1):
        label_run(engine, run["run_id"], k=args.k)
        if i % 10 == 0:
            logging.info("relabelled %d/%d", i, len(runs))
    runs = [engine.store.get_run(r["run_id"]) for r in runs]
    labelled = [r for r in runs if r.get("label_step")]
    summary = {"natural_failures": len(runs), "labelled": len(labelled),
               "label_nodes": dict(Counter(r["steps"][r["label_step"] - 1]["node_name"] for r in labelled)),
               "fixer": "careful-rerun (no gold answer)"}
    manifest = Path(args.dataset).with_suffix(".manifest.json")
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data.update(counterfactual_labelled=len(labelled), relabel=summary)
        manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
