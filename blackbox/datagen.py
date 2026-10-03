"""Black Box Trace Dataset generation (PRD section 9).

  python -m blackbox.datagen --provider sandbox --noise 0.07 --tasks-per-family 200
  python -m blackbox.datagen --provider openai  --tasks-per-family 200 --workers 8

1. Natural-language tasks from split-disjoint templates are run by the agent.
2. Successful runs receive 2-3 injected faults (known types 1-9); test-template
   runs also receive one held-out fault (types 10-12) for the unseen-fault split.
3. Runs that fail on their own are labelled by counterfactual replay.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import random
import time

from .agent.templates import generate_tasks
from .config import ROOT
from .engine import Engine, FAULT_CATALOG
from .labeler import label_run
from .llm.client import ChatLLM, SandboxLLM, get_llm
from .llm.usage import BudgetExceeded, UsageLedger

logger = logging.getLogger("blackbox.datagen")
KNOWN = [f for f in FAULT_CATALOG if not f["held_out"]]
HELD_OUT = [f for f in FAULT_CATALOG if f["held_out"]]


def _seed(*parts) -> int:
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _inject_some(engine, run, faults, count, rng, split):
    made = []
    options = [(fault, step) for fault in faults for step in run["steps"] if step["node_type"] in fault["node_types"]]
    rng.shuffle(options)
    used_faults = set()
    for fault, step in options:
        if len(made) >= count:
            break
        if fault["id"] in used_faults:
            continue
        try:
            failed = engine.inject(run["run_id"], step["step_id"], fault["id"])
        except ValueError:
            continue
        failed.update(split=split, dataset_role="injected", task_split=run["split"])
        engine.store.save_run(failed)
        used_faults.add(fault["id"])
        made.append(failed["run_id"])
    return made


def generate(out: str, provider: str = "sandbox", noise: float = 0.07, tasks_per_family: int = 200,
             seed: int = 42, workers: int = 1, label_limit: int | None = None, k: int = 3,
             inject_per_run: tuple[int, int] = (2, 3)) -> dict:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    llm = SandboxLLM(noise) if provider == "sandbox" else get_llm(provider)
    engine = Engine(str(out_path), llm=llm)
    tasks = generate_tasks(tasks_per_family, seed)
    rng = random.Random(seed)
    started = time.time()
    clean_ids, natural_ids, errors = [], [], []

    def run_task(task):
        return engine.run(task["prompt"], task["family"],
                          {**task["params"], "template_id": task["template_id"], "frozen_at": task["frozen_at"]},
                          expose_params=False, seed=_seed(seed, task["template_id"], task["prompt"]),
                          extra={"split": task["split"], "task_split": task["split"], "dataset": out_path.stem})

    def record(task, run):
        role = "clean" if run["success"] else "natural"
        run["dataset_role"] = role
        engine.store.save_run(run)
        (clean_ids if run["success"] else natural_ids).append(run["run_id"])

    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    record(futures[future], future.result())
                except BudgetExceeded:
                    raise
                except Exception as exc:
                    errors.append(str(exc))
                if i % 50 == 0:
                    logger.info("ran %d/%d tasks", i, len(tasks))
    else:
        for i, task in enumerate(tasks, 1):
            try:
                record(task, run_task(task))
            except BudgetExceeded:
                raise
            except Exception as exc:
                errors.append(str(exc))
            if i % 100 == 0:
                logger.info("ran %d/%d tasks", i, len(tasks))
    logger.info("clean=%d natural=%d errors=%d", len(clean_ids), len(natural_ids), len(errors))

    injected = []
    def inject_for(run_id):
        run = engine.store.get_run(run_id)
        local = random.Random(_seed(seed, run_id))
        made = _inject_some(engine, run, KNOWN, local.randint(*inject_per_run), local, run["split"])
        if run["split"] == "test":
            made += _inject_some(engine, run, HELD_OUT, 1, local, "unseen_fault")
        return made

    if workers > 1:
        with ThreadPoolExecutor(workers) as pool:
            for made in pool.map(inject_for, sorted(clean_ids)):
                injected += made
    else:
        for run_id in sorted(clean_ids):
            injected += inject_for(run_id)
    logger.info("injected=%d", len(injected))

    fixer = llm if isinstance(llm, ChatLLM) else None
    to_label = sorted(natural_ids)
    rng.shuffle(to_label)
    if label_limit is not None:
        to_label = to_label[:label_limit]
    labelled, agreement = 0, []
    for i, run_id in enumerate(to_label, 1):
        run = label_run(engine, run_id, k=k, fixer=fixer)
        labelled += run["label_step"] is not None
        truth = next((s["step_id"] for s in run["steps"] if s.get("sim_noise")), None)
        if truth is not None and run["label_step"] is not None:
            agreement.append(truth == run["label_step"])
        if i % 50 == 0:
            logger.info("labelled %d/%d natural failures", i, len(to_label))
    for run_id in natural_ids:
        if run_id not in to_label:
            run = engine.store.get_run(run_id)
            run["label_method"] = "unlabelled"
            engine.store.save_run(run)

    runs = engine.store.list_runs(limit=10 ** 7)
    counts = {}
    for run in runs:
        key = f"{run.get('split')}/{run.get('dataset_role')}"
        counts[key] = counts.get(key, 0) + 1
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "dataset": str(out_path), "provider": llm.provider,
        "model": llm.model, "noise": noise if provider == "sandbox" else None, "seed": seed,
        "tasks": len(tasks), "runs": len(runs), "steps": sum(len(r["steps"]) for r in runs),
        "clean": len(clean_ids), "natural_failures": len(natural_ids), "injected": len(injected),
        "counterfactual_labelled": labelled, "counterfactual_attempted": len(to_label),
        "labeler_agreement_with_simulation": (sum(agreement) / len(agreement)) if agreement else None,
        "errors": len(errors), "error_examples": errors[:5], "counts": counts,
        "billed_cost_usd": round(sum(r.get("cost_usd", 0) for r in runs), 6),
        "elapsed_s": round(time.time() - started, 1),
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    engine.store.close()
    return manifest


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default="sandbox", choices=["sandbox", "openai", "ollama"])
    parser.add_argument("--noise", type=float, default=0.07, help="sandbox mistake rate per LLM call")
    parser.add_argument("--tasks-per-family", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--label-limit", type=int, default=None, help="max natural failures to label")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or str(ROOT / "data" / f"dataset_{args.provider}.db")
    if Path(out).exists():
        parser.error(f"{out} exists; delete it or pass --out")
    manifest = generate(out, args.provider, args.noise, args.tasks_per_family, args.seed, args.workers,
                        args.label_limit, args.k)
    print(json.dumps(manifest, indent=2))
    if args.provider == "openai":
        print(json.dumps(UsageLedger().summary(), indent=2))


if __name__ == "__main__":
    main()
