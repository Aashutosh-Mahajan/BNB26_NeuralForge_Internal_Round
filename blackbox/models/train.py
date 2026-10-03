"""Train M1/M2/M3, tune ensemble weights on validation, save artifacts.

  python -m blackbox.models.train --dataset data/dataset_sandbox.db --out data/models
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import logging
from pathlib import Path
import time

import numpy as np

from ..config import ROOT
from ..features import node_stats
from ..features.semantic import INPUT_DIM, MAX_STEPS, SemanticEncoder, build_matrices
from ..storage import Store
from .ensemble import AE_TEMPERATURE, ModelOutputs, blend, pfail_signals
from .lgbm import TABULAR_NAMES, tabular, train_lightgbm

logger = logging.getLogger("blackbox.train")
SPLITS = ("train", "validation", "test", "unseen_fault")


def load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for path in paths:
        store = Store(path)
        runs += [r for r in store.list_runs(limit=10 ** 7) if r.get("status") in ("PASSED", "FAILED")]
        store.close()
    return runs


def by_split(runs):
    return {split: [r for r in runs if r.get("split") == split] for split in SPLITS}


def has_label(run) -> bool:
    return run.get("success") is False and isinstance(run.get("label_step"), int)


def ranking_metrics(runs, rankings) -> dict:
    top1, top3, rr, near = [], [], [], []
    for run, ranked in zip(runs, rankings):
        if not has_label(run):
            continue
        truth = run["label_step"]
        rank = ranked.index(truth) + 1 if truth in ranked else None
        top1.append(rank == 1)
        top3.append(rank is not None and rank <= 3)
        rr.append(1 / rank if rank else 0.0)
        near.append(bool(ranked) and abs(ranked[0] - truth) <= 1)
    mean = lambda v: float(np.mean(v)) if v else None
    return {"count": len(top1), "top1": mean(top1), "top3": mean(top3), "mrr": mean(rr), "within_one": mean(near)}


def auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def rank_steps(run, scores) -> list[int]:
    ids = [s["step_id"] for s in run["steps"]]
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-scores[i], i))]


def _tensors(encoded, runs, mean, std):
    n = len(encoded)
    x = np.zeros((n, MAX_STEPS, INPUT_DIM), np.float32)
    mask = np.zeros((n, MAX_STEPS), bool)
    y_fail = np.zeros(n, np.float32)
    y_step = np.full(n, -1, np.int64)
    for i, (e, run) in enumerate(zip(encoded, runs)):
        t = min(len(e["matrix"]), MAX_STEPS)
        x[i, :t] = (e["matrix"][:t] - mean) / std
        mask[i, :t] = True
        y_fail[i] = float(run.get("success") is False)
        if has_label(run):
            ids = [s["step_id"] for s in run["steps"]]
            y_step[i] = ids.index(run["label_step"])
    return x, mask, y_fail, y_step


def train_transformer(train, val, val_runs, device, epochs=30, seed=42, config=None, log=None):
    import torch
    from .transformer import StepBlameTransformer, localization_loss
    config = config or {"hidden": 256, "heads": 4, "layers": 4, "max_steps": MAX_STEPS}
    torch.manual_seed(seed)
    model = StepBlameTransformer(INPUT_DIM, config["hidden"], config["heads"], config["layers"], config["max_steps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    tensors = [torch.tensor(a, device=device) for a in train]
    val_tensors = [torch.tensor(a, device=device) for a in val[:2]]
    generator = torch.Generator().manual_seed(seed)
    best, best_state, history = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(tensors[0]), generator=generator)
        losses = []
        for i in range(0, len(order), 64):
            idx = order[i:i + 64].to(device)
            x, mask, y_fail, y_step = (t[idx] for t in tensors)
            loss = localization_loss(model(x, mask), y_fail, y_step, mask, sigma=1.0, blame_weight=1.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss))
        scheduler.step()
        model.eval()
        with torch.no_grad():
            out = model(*val_tensors)
        blame = out["blame"].cpu().numpy()
        rankings = [rank_steps(run, blame[i, :len(run["steps"])]) for i, run in enumerate(val_runs)]
        metric = ranking_metrics(val_runs, rankings)
        roc = auroc(val[2], out["p_fail"].cpu().numpy()) or 0.0
        score = (metric["top1"] or 0) + 0.5 * roc
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)), "val_top1": metric["top1"], "val_auroc": roc})
        if log:
            log(f"M1 epoch {epoch + 1:02d} loss={np.mean(losses):.4f} val_top1={metric['top1']:.3f} val_auroc={roc:.3f}")
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, config, history


def tune_weights(items, runs):
    grid = [w / 20 for w in range(21)]
    best, chosen = None, None
    for w1, w2 in itertools.product(grid, grid):
        w3 = round(1 - w1 - w2, 4)
        # Validation holds known fault types only, so it cannot reward the anomaly model
        # whose job is unseen faults; keep a floor of 0.1 for M3 as a design prior.
        if w3 < 0.1 - 1e-9:
            continue
        weights = {"m1": w1, "m2": w2, "m3": max(w3, 0.0)}
        rankings = [rank_steps(run, blend(item, weights)) for item, run in zip(items, runs)]
        m = ranking_metrics(runs, rankings)
        # Prefer the PRD prior (0.6/0.25/0.15) on ties.
        key = ((m["top1"] or 0), (m["mrr"] or 0), -abs(w1 - 0.6) - abs(w2 - 0.25))
        if best is None or key > best:
            best, chosen = key, weights
    return chosen, {"top1": best[0], "mrr": best[1]}


def fit_stacker(items, runs):
    from sklearn.linear_model import LogisticRegression
    x = np.array([pfail_signals(item) for item in items])
    y = np.array([int(r.get("success") is False) for r in runs])
    if y.min() == y.max():
        return {"coef": [4.0, 0.0, 0.0], "intercept": -2.0}
    model = LogisticRegression(C=1.0, max_iter=1000).fit(x, y)
    return {"coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}


def train_all(dataset_paths, out_dir, epochs=30, seed=42, encoder=None, drop_features=None, log=print,
              save=True, ae_epochs=40):
    import torch
    from .autoencoder import train_autoencoder
    started = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = load_runs(dataset_paths)
    splits = by_split(runs)
    train_runs, val_runs = splits["train"], splits["validation"]
    if not train_runs or not val_runs:
        raise ValueError("The dataset needs train and validation splits; generate it with python -m blackbox.datagen")
    encoder = encoder or SemanticEncoder()
    stats = node_stats([r for r in train_runs if r.get("success")])
    log(f"Encoding {len(train_runs) + len(val_runs)} runs (MiniLM + NLI on {encoder._device()})...")
    t0 = time.time()
    enc_train = build_matrices(train_runs, encoder, stats)
    enc_val = build_matrices(val_runs, encoder, stats)
    log(f"Encoded in {time.time() - t0:.1f}s; semantic status: {encoder.status}")
    drop = drop_features or []
    if drop:
        from ..features import FEATURE_NAMES
        offset = INPUT_DIM - len(FEATURE_NAMES)
        for e in enc_train + enc_val:
            for name in drop:
                e["numeric"][:, FEATURE_NAMES.index(name)] = 0
                e["matrix"][:, offset + FEATURE_NAMES.index(name)] = 0
    steps = np.concatenate([e["matrix"] for e in enc_train])
    mean, std = steps.mean(0), steps.std(0) + 1e-6
    # M2: LightGBM on tabular step features.
    def tab_xy(encoded, runs):
        xs, ys = [], []
        for e, run in zip(encoded, runs):
            if run.get("success") is False and not has_label(run):
                continue
            xs.append(tabular(e["numeric"], e["onehot"]))
            label = run["label_step"] if has_label(run) else None
            ys.append(np.array([float(s["step_id"] == label) for s in run["steps"]], np.float32))
        return np.concatenate(xs), np.concatenate(ys)
    booster = train_lightgbm(*tab_xy(enc_train, train_runs), *tab_xy(enc_val, val_runs), seed=seed)
    log(f"M2 LightGBM trained: {booster.best_iteration} rounds")
    # M3: autoencoder on successful training steps only.
    clean = np.concatenate([(e["matrix"] - mean) / std for e, r in zip(enc_train, train_runs) if r.get("success")])
    autoencoder = train_autoencoder(clean, epochs=ae_epochs, seed=seed, device=device)
    with torch.no_grad():
        errors = autoencoder.anomaly_score(torch.tensor(clean, device=device)).cpu().numpy()
    names = [s["node_name"] for r in train_runs if r.get("success") for s in r["steps"]]
    ae_medians = {name: float(np.median([e for e, n in zip(errors, names) if n == name])) for name in set(names)}
    ae_medians["*"] = float(np.median(errors))
    log(f"M3 autoencoder trained on {len(clean)} successful steps")
    # M1: Transformer.
    train_t = _tensors(enc_train, train_runs, mean, std)
    val_t = _tensors(enc_val, val_runs, mean, std)
    transformer, config, history = train_transformer(train_t, val_t, val_runs, device, epochs, seed, log=log)
    log(f"M1 transformer trained ({sum(p.numel() for p in transformer.parameters()) / 1e6:.2f}M params)")
    outputs = ModelOutputs(transformer, booster, autoencoder, (mean, std), ae_medians, device)
    items = outputs.compute(enc_val, [[s["node_name"] for s in r["steps"]] for r in val_runs])
    weights, val_metric = tune_weights(items, val_runs)
    stacker = fit_stacker(items, val_runs)
    log(f"Ensemble weights {weights} (validation top1={val_metric['top1']:.3f})")
    version = hashlib.sha256(f"{datetime.now(timezone.utc).isoformat()}{dataset_paths}".encode()).hexdigest()[:12]
    meta = {"version": version, "created_at": datetime.now(timezone.utc).isoformat(), "weights": weights,
            "pfail_stacker": stacker, "ae_medians": ae_medians, "ae_temperature": AE_TEMPERATURE,
            "node_stats": stats, "transformer": config, "input_dim": INPUT_DIM, "tabular_features": TABULAR_NAMES,
            "dropped_features": drop, "semantic": encoder.status, "history": history,
            "provenance": {"datasets": [str(p) for p in dataset_paths], "seed": seed, "epochs": epochs,
                           "train_runs": len(train_runs), "validation_runs": len(val_runs),
                           "train_steps": int(len(steps)), "lightgbm_rounds": booster.best_iteration,
                           "fault_types_seen": sorted({r.get("fault_type") for r in train_runs if r.get("fault_type")}),
                           "label_methods": sorted({str(r.get("label_method")) for r in train_runs if r.get("success") is False}),
                           "device": device, "train_seconds": round(time.time() - started, 1)}}
    if save:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(transformer.state_dict(), out / "transformer.pt")
        torch.save(autoencoder.state_dict(), out / "autoencoder.pt")
        booster.save_model(str(out / "lightgbm.txt"), num_iteration=booster.best_iteration)
        np.savez(out / "scaler.npz", mean=mean, std=std)
        (out / "ensemble.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log(f"Saved models to {out} (version {version}) in {time.time() - started:.0f}s")
    return {"meta": meta, "outputs": outputs, "stats": stats}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", action="append", help="dataset .db (repeatable)")
    parser.add_argument("--out", default=str(ROOT / "data" / "models"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    datasets = args.dataset or [str(ROOT / "data" / "dataset_sandbox.db")]
    train_all(datasets, args.out, args.epochs, args.seed, log=logger.info)


if __name__ == "__main__":
    main()
