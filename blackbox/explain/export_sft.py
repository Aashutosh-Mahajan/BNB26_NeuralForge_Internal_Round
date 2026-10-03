"""Export (trace + diagnosis -> narrative) pairs for the Kaggle QLoRA explainer (FR-16).

  python -m blackbox.explain.export_sft --dataset data/dataset_sandbox.db --out data/sft_explainer.jsonl
  python -m blackbox.explain.export_sft ... --teacher openai --limit 400   # GPT-written targets (costs tokens)
"""
from __future__ import annotations

import argparse
import json
import random

from ..diagnosis import diagnose_many
from ..models.train import has_label, load_runs
from .evidence import narrative
from .narrator import SYSTEM, prompt_for


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--out", default="data/sft_explainer.jsonl")
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--teacher", choices=["template", "openai"], default="template")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    runs = [r for r in load_runs(args.dataset) if r.get("split") == "train" and has_label(r)]
    random.Random(args.seed).shuffle(runs)
    runs = runs[:args.limit]
    teacher = None
    if args.teacher == "openai":
        from ..llm import get_llm
        teacher = get_llm("openai")
    written = 0
    with open(args.out, "w", encoding="utf-8") as handle:
        for i in range(0, len(runs), 64):
            chunk = runs[i:i + 64]
            for run, diagnosis in zip(chunk, diagnose_many(chunk)):
                # Targets are grounded in the true label, so the student learns faithful explanations.
                diagnosis["root_cause"]["step"] = run["label_step"]
                diagnosis["root_cause"]["node"] = run["steps"][run["label_step"] - 1]["node_name"]
                prompt = prompt_for(run, diagnosis)
                target = narrative(diagnosis, run)
                if teacher is not None:
                    result = teacher.complete_json(SYSTEM + " Respond with JSON only: {\"narrative\": \"...\"}", prompt,
                                                   seed=5, purpose="sft_teacher", run_id=run["run_id"])
                    target = (result.data or {}).get("narrative", target) if isinstance(result.data, dict) else target
                handle.write(json.dumps({"messages": [{"role": "system", "content": SYSTEM},
                                                      {"role": "user", "content": prompt},
                                                      {"role": "assistant", "content": target}]}) + "\n")
                written += 1
    print(json.dumps({"out": args.out, "examples": written, "teacher": args.teacher}))


if __name__ == "__main__":
    main()
