"""Reviewed training queue and gated model promotion.

  python -m blackbox.review_queue status
  python -m blackbox.review_queue export --out data/dataset_reviewed.db
  python -m blackbox.models.train --dataset data/dataset_sandbox.db --dataset data/dataset_openai.db \
      --dataset data/dataset_reviewed.db --out data/models_v2
  python -m blackbox.review_queue promote --candidate data/models_v2 --dataset data/dataset_openai.db
  python -m blackbox.review_queue promote ... --activate     # only then is the live model replaced

Runs never train the model automatically: a human confirms or corrects the suspect in
the dashboard, reviewed runs are exported, a new version is trained into its own
folder, and it is activated only if it is not worse on the held-out test split.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .config import ROOT, settings
from .storage import Store


def reviewed(store):
    return [r for r in store.list_runs(limit=10 ** 6)
            if (r.get("review") or {}).get("queued_for_training") and r["review"].get("label_step")]


def export(source: str, out: str):
    runs = reviewed(Store(source))
    target = Store(out)
    for run in runs:
        bucket = int(hashlib.sha256(run["run_id"].encode()).hexdigest()[:4], 16) % 10
        run = {**run, "label_step": run["review"]["label_step"], "label_method": "human_review",
               "split": "validation" if bucket < 2 else "train", "dataset_role": "reviewed"}
        target.save_run(run)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "source": Path(source).name, "runs": len(runs),
                "label_method": "human_review", "split": "hash(run_id): 20% validation, 80% train"}
    Path(out).with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _test_top1(model_dir: str, datasets: list[str]):
    from .evaluation import Scorer, evaluate_split
    from .models.ensemble import Ensemble
    from .models.train import by_split, load_runs
    test = by_split(load_runs(datasets))["test"]
    return evaluate_split(test, Scorer(Ensemble(model_dir)), 42)


def promote(candidate: str, datasets: list[str], activate: bool = False, tolerance: float = 0.0):
    current_dir = settings().model_dir
    new, old = _test_top1(candidate, datasets), _test_top1(current_dir, datasets)
    passes = (new["top1"] or 0) >= (old["top1"] or 0) - tolerance and (new["auroc"] or 0) >= (old["auroc"] or 0) - 0.01
    decision = {"at": datetime.now(timezone.utc).isoformat(), "candidate": candidate, "current": current_dir,
                "candidate_top1": new["top1"], "current_top1": old["top1"], "candidate_auroc": new["auroc"],
                "current_auroc": old["auroc"], "gate_passed": passes, "activated": False}
    if passes and activate:
        backup = Path(current_dir).with_name(Path(current_dir).name + "_backup_" + datetime.now().strftime("%Y%m%d%H%M%S"))
        shutil.copytree(current_dir, backup)
        shutil.rmtree(current_dir)
        shutil.copytree(candidate, current_dir)
        decision.update(activated=True, backup=str(backup))
    log = ROOT / "data" / "model_promotions.json"
    history = json.loads(log.read_text(encoding="utf-8")) if log.is_file() else []
    log.write_text(json.dumps(history + [decision], indent=2), encoding="utf-8")
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    e = sub.add_parser("export")
    e.add_argument("--source", default=str(ROOT / "data" / "traces.db"))
    e.add_argument("--out", default=str(ROOT / "data" / "dataset_reviewed.db"))
    p = sub.add_parser("promote")
    p.add_argument("--candidate", required=True)
    p.add_argument("--dataset", action="append", required=True)
    p.add_argument("--activate", action="store_true")
    args = parser.parse_args()
    if args.cmd == "status":
        runs = reviewed(Store(str(ROOT / "data" / "traces.db")))
        print(json.dumps({"queued_for_training": len(runs),
                          "verdicts": {v: sum(r["review"]["verdict"] == v for r in runs) for v in ("confirmed", "wrong_step")}}, indent=2))
    elif args.cmd == "export":
        print(json.dumps(export(args.source, args.out), indent=2))
    else:
        if Path(args.candidate).resolve() == Path(settings().model_dir).resolve():
            parser.error("The candidate must be a separate folder (e.g. data/models_v2)")
        print(json.dumps(promote(args.candidate, args.dataset, args.activate), indent=2))


if __name__ == "__main__":
    main()
