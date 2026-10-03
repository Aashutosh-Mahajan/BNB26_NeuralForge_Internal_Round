"""A dependency-free supervised logistic step ranker with saved provenance.

This is a trainable baseline, not LightGBM or the PRD neural ensemble. Labels are
read only by train_ranker; prediction consumes the observable feature matrix.
"""
from __future__ import annotations
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random

from ..features import FEATURE_NAMES, extract_features

# Timing and token counts vary by environment; identity/template strings never enter.
_EXCLUDED = {"latency", "tokens", "reference_divergence", "first_divergence"}


def _sigmoid(value):
    return 1 / (1 + math.exp(-max(-40, min(40, value))))


class LinearStepRanker:
    def __init__(self, artifact: dict):
        if artifact.get("features") != FEATURE_NAMES:
            raise ValueError("Model feature schema does not match this feature extractor.")
        self.artifact = artifact

    def predict(self, run: dict) -> list[dict]:
        features = extract_features(run)
        predictions = []
        for row, detail in zip(features["matrix"], features["details"]):
            normalized = [(value - mean) / scale for value, mean, scale in
                          zip(row, self.artifact["means"], self.artifact["scales"])]
            contributions = {name: weight * value for name, weight, value in
                             zip(FEATURE_NAMES, self.artifact["weights"], normalized)}
            logit = self.artifact["bias"] + sum(contributions.values())
            predictions.append({"step": detail["step"], "node": detail["node"],
                                "score": _sigmoid(logit), "contributions": contributions})
        return sorted(predictions, key=lambda prediction: -prediction["score"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.artifact, indent=2), encoding="utf-8")


def train_ranker(runs: list[dict], *, seed: int = 42, epochs: int = 160) -> LinearStepRanker:
    """Train only on caller-supplied train split; no successful-pair references."""
    randomizer = random.Random(seed)
    examples = []
    for run in runs:
        extracted = extract_features(run)
        failed = run.get("success") is False
        for row, detail in zip(extracted["matrix"], extracted["details"]):
            label = int(failed and detail["step"] == run.get("label_step"))
            examples.append((row, label))
    if not examples or not any(y for _, y in examples):
        raise ValueError("Training needs recorded steps and at least one localized failed run.")
    width = len(FEATURE_NAMES)
    means = [sum(x[i] for x, _ in examples) / len(examples) for i in range(width)]
    scales = [max(1e-4, math.sqrt(sum((x[i] - means[i]) ** 2 for x, _ in examples) / len(examples)))
              for i in range(width)]
    normalized = [([(value - means[i]) / scales[i] for i, value in enumerate(row)], label)
                  for row, label in examples]
    positives = sum(label for _, label in examples)
    positive_weight = min(12.0, (len(examples) - positives) / positives)
    weights, bias = [0.0] * width, 0.0
    for epoch in range(epochs):
        randomizer.shuffle(normalized)
        lr = 0.035 / (1 + epoch / 40)
        for row, label in normalized:
            error = (_sigmoid(bias + sum(w * x for w, x in zip(weights, row))) - label)
            error *= positive_weight if label else 1.0
            bias -= lr * error
            for i, name in enumerate(FEATURE_NAMES):
                if name not in _EXCLUDED:
                    weights[i] -= lr * (error * row[i] + 0.004 * weights[i])
    templates = sorted({str(run.get("template_id", run.get("params", {}).get("template_id", "unknown")))
                        for run in runs})
    artifact = {"kind": "supervised_logistic_step_ranker", "schema_version": 1,
                "features": FEATURE_NAMES, "means": means, "scales": scales,
                "weights": weights, "bias": bias,
                "provenance": {"trained_at": datetime.now(timezone.utc).isoformat(),
                               "seed": seed, "epochs": epochs, "runs": len(runs),
                               "steps": len(examples), "positive_steps": positives,
                               "templates": templates, "references_used": False,
                               "fault_types": sorted({str(r.get("fault_type")) for r in runs if r.get("fault_type")}),
                               "limitations": "Synthetic sandbox only; no LLM or public benchmark validation."}}
    return LinearStepRanker(artifact)
