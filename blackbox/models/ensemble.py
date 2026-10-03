"""Weighted ensemble of M1 (Transformer), M2 (LightGBM) and M3 (autoencoder).

    blame(t) = w1 * M1(t) + w2 * M2(t) + w3 * M3(t)

Each model's scores are first turned into a distribution over the run's steps.
Weights are tuned on the validation split; p_fail is a logistic stacker over
the three models' run-level signals, also fit on validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import threading

import numpy as np

from ..config import settings
from ..features.semantic import INPUT_DIM, MAX_STEPS, SemanticEncoder, build_matrices, shared_encoder
from .lgbm import TABULAR_NAMES, predict as lgb_predict, shap_values, tabular

AE_TEMPERATURE = 1.5
WINDOW_STRIDE = 12


def _softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = (values - values.max()) * temperature
    e = np.exp(z)
    return e / e.sum()


def _normalize(values: np.ndarray) -> np.ndarray:
    total = values.sum()
    return values / total if total > 0 else np.full_like(values, 1 / len(values))


def _sigmoid(x):
    return 1 / (1 + math.exp(-max(-40.0, min(40.0, x))))


class ModelOutputs:
    """Raw per-model outputs for a batch of encoded runs (shared by training and inference)."""

    def __init__(self, transformer, booster, autoencoder, scaler, ae_medians, device="cpu"):
        self.transformer, self.booster, self.autoencoder = transformer, booster, autoencoder
        self.mean, self.std = scaler
        self.ae_medians = ae_medians
        self.device = device

    def _transformer_scores(self, batch, mask, lengths):
        """M1 blame per step. Runs longer than the Transformer's 24-step positional table are
        scored as overlapping 24-step windows (stride 12); each step's blame logit is the
        mean over the windows that cover it, then normalised over the whole run. p_fail is
        the maximum over windows. No retraining is needed."""
        import torch
        blame, pfail = [None] * len(lengths), [0.0] * len(lengths)
        short = [i for i, n in enumerate(lengths) if n <= MAX_STEPS]
        if short:
            x = torch.tensor(batch[short, :MAX_STEPS], device=self.device)
            m = torch.tensor(mask[short, :MAX_STEPS], device=self.device)
            out = self.transformer(x, m)
            probs, fails = out["blame"].float().cpu().numpy(), out["p_fail"].float().cpu().numpy()
            for row, i in enumerate(short):
                blame[i] = probs[row, :lengths[i]].astype(np.float64)
                pfail[i] = float(fails[row])
        for i, n in enumerate(lengths):
            if n <= MAX_STEPS:
                continue
            starts = list(range(0, n - MAX_STEPS + 1, WINDOW_STRIDE))
            if starts[-1] != n - MAX_STEPS:
                starts.append(n - MAX_STEPS)
            x = torch.tensor(np.stack([batch[i, s:s + MAX_STEPS] for s in starts]), device=self.device)
            m = torch.ones((len(starts), MAX_STEPS), dtype=torch.bool, device=self.device)
            out = self.transformer(x, m)
            logits = out["blame_logits"].float().cpu().numpy()
            total, count = np.zeros(n), np.zeros(n)
            for w, s in enumerate(starts):
                # Centre each window's logits so windows are comparable before averaging.
                total[s:s + MAX_STEPS] += logits[w] - logits[w].mean()
                count[s:s + MAX_STEPS] += 1
            blame[i] = _softmax(total / np.maximum(count, 1))
            pfail[i] = float(out["p_fail"].max())
        return blame, pfail

    def compute(self, encoded: list[dict], node_names: list[list[str]]) -> list[dict]:
        import torch
        results = []
        if not encoded:
            return results
        lengths = [len(e["matrix"]) for e in encoded]
        width = max(MAX_STEPS, max(lengths))
        batch = np.zeros((len(encoded), width, INPUT_DIM), np.float32)
        mask = np.zeros((len(encoded), width), bool)
        for i, e in enumerate(encoded):
            batch[i, :lengths[i]] = (e["matrix"] - self.mean) / self.std
            mask[i, :lengths[i]] = True
        with torch.no_grad():
            m1_blame, m1_pfail = (self._transformer_scores(batch, mask, lengths) if self.transformer is not None
                                  else (None, None))
            flat = torch.tensor(batch[mask], device=self.device)
            ae_err = self.autoencoder.anomaly_score(flat).float().cpu().numpy() if self.autoencoder is not None else None
        tab = np.concatenate([tabular(e["numeric"], e["onehot"]) for e in encoded]) if encoded else None
        m2 = lgb_predict(self.booster, tab) if self.booster is not None else None
        cursor = 0
        for i, e in enumerate(encoded):
            n = lengths[i]
            names = node_names[i]
            item = {"n": n}
            if m1_blame is not None:
                item["m1"] = m1_blame[i]
                item["m1_pfail"] = float(m1_pfail[i])
                item["m1_windows"] = int(math.ceil(max(0, n - MAX_STEPS) / WINDOW_STRIDE)) + 1
            if m2 is not None:
                item["m2_raw"] = m2[cursor:cursor + n].astype(np.float64)
                item["m2"] = _normalize(item["m2_raw"])
                item["tab"] = tab[cursor:cursor + n]
            if ae_err is not None:
                errors = ae_err[cursor:cursor + n].astype(np.float64)
                z = np.array([math.log(max(err, 1e-9) / self.ae_medians.get(name, self.ae_medians.get("*", 1.0)))
                              for err, name in zip(errors, names)])
                item["m3_raw"] = z
                item["m3"] = _softmax(z, AE_TEMPERATURE)
            cursor += n
            results.append(item)
        return results


def blend(item: dict, weights: dict) -> np.ndarray:
    total = np.zeros(item["n"])
    used = 0.0
    for key in ("m1", "m2", "m3"):
        if key in item and weights.get(key, 0) > 0:
            total += weights[key] * item[key]
            used += weights[key]
    return total / used if used else np.full(item["n"], 1 / item["n"])


def pfail_signals(item: dict) -> list[float]:
    m2 = item.get("m2_raw")
    return [item.get("m1_pfail", 0.5), float(1 - np.prod(1 - np.clip(m2, 0, 0.999))) if m2 is not None else 0.5,
            float(np.max(item["m3_raw"])) if "m3_raw" in item else 0.0]


class Ensemble:
    def __init__(self, directory: str | Path):
        import lightgbm as lgb
        import torch
        from .autoencoder import StepAutoencoder
        from .transformer import StepBlameTransformer
        self.directory = Path(directory)
        self.meta = json.loads((self.directory / "ensemble.json").read_text(encoding="utf-8"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        config = self.meta["transformer"]
        transformer = StepBlameTransformer(INPUT_DIM, config["hidden"], config["heads"], config["layers"], config["max_steps"])
        transformer.load_state_dict(torch.load(self.directory / "transformer.pt", map_location=self.device))
        transformer.to(self.device).eval()
        autoencoder = StepAutoencoder(INPUT_DIM)
        autoencoder.load_state_dict(torch.load(self.directory / "autoencoder.pt", map_location=self.device))
        autoencoder.to(self.device).eval()
        booster = lgb.Booster(model_file=str(self.directory / "lightgbm.txt"))
        scaler = np.load(self.directory / "scaler.npz")
        self.outputs = ModelOutputs(transformer, booster, autoencoder, (scaler["mean"], scaler["std"]),
                                    self.meta["ae_medians"], self.device)
        self.weights = self.meta["weights"]
        self.stats = self.meta["node_stats"]
        self._lock = threading.Lock()

    @property
    def version(self) -> str:
        return self.meta["version"]

    def score(self, runs: list[dict], encoder: SemanticEncoder | None = None) -> list[dict]:
        encoder = encoder or shared_encoder()
        with self._lock:
            encoded = build_matrices(runs, encoder, self.stats)
            items = self.outputs.compute(encoded, [[s.get("node_name", "") for s in r.get("steps", [])] for r in runs])
        coefficients = self.meta["pfail_stacker"]
        for item, enc in zip(items, encoded):
            item["blame"] = blend(item, self.weights)
            logit = coefficients["intercept"] + sum(c * v for c, v in zip(coefficients["coef"], pfail_signals(item)))
            item["p_fail"] = _sigmoid(logit)
            item["rows"] = enc["rows"]
            item["details"] = enc["details"]
        return items

    def shap(self, item: dict, index: int) -> dict:
        contributions = shap_values(self.outputs.booster, item["tab"][index:index + 1])[0]
        return {name: float(value) for name, value in zip(TABULAR_NAMES, contributions[:-1])}


_ensemble: Ensemble | None = None
_ensemble_error: str | None = None
_load_lock = threading.Lock()


def load_ensemble(directory: str | None = None, reload: bool = False) -> Ensemble | None:
    """The trained ensemble, or None when artifacts are missing (heuristic fallback)."""
    global _ensemble, _ensemble_error
    with _load_lock:
        if _ensemble is not None and not reload:
            return _ensemble
        path = Path(directory or settings().model_dir)
        if not (path / "ensemble.json").is_file():
            _ensemble_error = f"No trained models in {path}. Run python -m blackbox.models.train"
            return None
        try:
            _ensemble = Ensemble(path)
            _ensemble_error = None
        except Exception as exc:
            _ensemble, _ensemble_error = None, f"Model load failed: {exc}"
        return _ensemble


def ensemble_status() -> dict:
    model = load_ensemble()
    if model is None:
        return {"available": False, "reason": _ensemble_error}
    return {"available": True, "version": model.version, "weights": model.weights, "device": model.device,
            "provenance": model.meta.get("provenance", {})}
