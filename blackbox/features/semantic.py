"""The 412-dimensional step representation: 384 MiniLM + 8 node one-hot + 20 numeric.

Frozen encoders (all-MiniLM-L6-v2 and the nli-deberta-v3-small cross-encoder) run
on the GPU when available. Vectors and NLI scores are cached by content hash, so
each distinct step text is encoded once.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from pathlib import Path

import numpy as np

from ..config import ROOT
from .extractor import FEATURE_NAMES, extract_features, observable_run

logger = logging.getLogger(__name__)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-small"
EMBED_DIM = 384
NODE_TYPES = ["planner", "router", "tool", "retriever", "memory", "reasoner", "final", "other"]
INPUT_DIM = EMBED_DIM + len(NODE_TYPES) + len(FEATURE_NAMES)  # 412
MAX_STEPS = 24


def _compact(value, limit=600) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return text[:limit]


def step_text(step: dict) -> str:
    """What the step was asked and what it produced (observable fields only)."""
    inputs = step.get("input") if isinstance(step.get("input"), dict) else {}
    head = inputs.get("prompt") or (inputs.get("dependencies", {}).get("planner", {}) or {}).get("constraints") \
        if step.get("node_type") in ("planner", "router") else None
    text = f"[{step.get('node_type')}] {step.get('node_name')}"
    if head:
        text += f" | task: {_compact(head, 300)}"
    return text + f" | output: {_compact(step.get('output'))}"


def nli_text(step: dict) -> str:
    output = step.get("output")
    if isinstance(output, dict):
        parts = [f"{k} is {v}" for k, v in output.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
        if parts:
            return f"{step.get('node_name')}: " + "; ".join(parts)[:300]
    return f"{step.get('node_name')}: {_compact(output, 300)}"


class _Cache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v BLOB)")
        self.lock = threading.Lock()

    def get_many(self, keys):
        found = {}
        with self.lock:
            for i in range(0, len(keys), 500):
                chunk = keys[i:i + 500]
                rows = self.conn.execute(f"SELECT k, v FROM kv WHERE k IN ({','.join('?' * len(chunk))})", chunk).fetchall()
                found.update(rows)
        return found

    def put_many(self, items):
        with self.lock, self.conn:
            self.conn.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", items)


def _key(*parts) -> str:
    return hashlib.sha256("␟".join(parts).encode()).hexdigest()


class SemanticEncoder:
    """Lazy MiniLM + NLI encoders with a persistent cache and a hashing fallback."""

    def __init__(self, cache_path: str | Path | None = None, device: str | None = None, use_nli: bool = True):
        self.cache = _Cache(Path(cache_path or ROOT / "data" / "cache" / "semantic.db"))
        self.device = device
        self.use_nli = use_nli
        self._embedder = None
        self._nli = None
        self.fallback = False
        self._lock = threading.Lock()

    def _device(self):
        if self.device:
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        with self._lock:
            if self._embedder is not None or self.fallback:
                return
            try:
                from sentence_transformers import CrossEncoder, SentenceTransformer
                self._embedder = SentenceTransformer(EMBED_MODEL, device=self._device())
                if self.use_nli:
                    self._nli = CrossEncoder(NLI_MODEL, device=self._device(), max_length=192)
            except Exception as exc:  # Offline or missing package: deterministic fallback.
                logger.warning("Semantic encoders unavailable (%s); using hashed embeddings.", exc)
                self.fallback = True

    @property
    def status(self) -> dict:
        return {"embedder": None if self.fallback else EMBED_MODEL, "nli": None if self.fallback or not self.use_nli else NLI_MODEL,
                "fallback": self.fallback, "device": self._device()}

    @staticmethod
    def _hashed(text: str) -> np.ndarray:
        vector = np.zeros(EMBED_DIM, dtype=np.float32)
        for token in text.lower().split():
            digest = int(hashlib.md5(token.encode()).hexdigest(), 16)
            vector[digest % EMBED_DIM] += 1.0 if (digest >> 9) & 1 else -1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed(self, texts: list[str]) -> np.ndarray:
        self._load()
        tag = "hash" if self.fallback else EMBED_MODEL
        keys = [_key(tag, t) for t in texts]
        found = self.cache.get_many(list(set(keys)))
        missing = sorted({t for t, k in zip(texts, keys) if k not in found})
        if missing:
            if self.fallback:
                vectors = [self._hashed(t) for t in missing]
            else:
                vectors = self._embedder.encode(missing, batch_size=128, normalize_embeddings=True,
                                                convert_to_numpy=True, show_progress_bar=len(missing) > 2000)
            items = [(_key(tag, t), np.asarray(v, dtype=np.float16).tobytes()) for t, v in zip(missing, vectors)]
            self.cache.put_many(items)
            found.update(items)
        return np.stack([np.frombuffer(found[k], dtype=np.float16).astype(np.float32) for k in keys]) \
            if keys else np.zeros((0, EMBED_DIM), np.float32)

    def contradiction(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        """P(contradiction) for (premise, hypothesis) pairs."""
        if not pairs:
            return np.zeros(0, np.float32)
        self._load()
        if self.fallback or self._nli is None:
            return np.zeros(len(pairs), np.float32)
        keys = [_key(NLI_MODEL, a, b) for a, b in pairs]
        found = self.cache.get_many(list(set(keys)))
        missing = sorted({pair for pair, k in zip(pairs, keys) if k not in found})
        if missing:
            scores = self._nli.predict(missing, batch_size=128, apply_softmax=True, show_progress_bar=len(missing) > 2000)
            # Label order for this model: contradiction, entailment, neutral.
            items = [(_key(NLI_MODEL, a, b), np.float32(s[0]).tobytes()) for (a, b), s in zip(missing, scores)]
            self.cache.put_many(items)
            found.update(items)
        return np.array([np.frombuffer(found[k], dtype=np.float32)[0] for k in keys], dtype=np.float32)


def node_one_hot(node_type: str) -> np.ndarray:
    vector = np.zeros(len(NODE_TYPES), np.float32)
    vector[NODE_TYPES.index(node_type) if node_type in NODE_TYPES else -1] = 1.0
    return vector


def _contradictions(run: dict, encoder: SemanticEncoder) -> list[float]:
    steps = run.get("steps", [])
    pairs, owners = [], []
    for i, step in enumerate(steps):
        for child in steps:
            if step.get("step_id") in child.get("parent_step_ids", []):
                pairs.append((nli_text(step), nli_text(child)))
                owners.append(i)
    scores = encoder.contradiction(pairs)
    result = [0.0] * len(steps)
    for owner, score in zip(owners, scores):
        result[owner] = max(result[owner], float(score))
    return result


def build_matrices(runs: list[dict], encoder: SemanticEncoder, stats: dict | None = None) -> list[dict]:
    """Batch-encode many runs. Returns one {matrix [T,412], numeric, details} per run."""
    observed = [observable_run(run) for run in runs]
    texts = [step_text(step) for run in observed for step in run["steps"]]
    embeddings = encoder.embed(texts)
    out, cursor = [], 0
    for raw, run in zip(runs, observed):
        count = len(run["steps"])
        emb = embeddings[cursor:cursor + count]
        cursor += count
        features = extract_features(raw, stats=stats, contradiction=_contradictions(run, encoder))
        numeric = np.array(features["matrix"], dtype=np.float32).reshape(count, len(FEATURE_NAMES))
        onehot = np.stack([node_one_hot(str(s.get("node_type", ""))) for s in run["steps"]]) if count else \
            np.zeros((0, len(NODE_TYPES)), np.float32)
        matrix = np.concatenate([emb, onehot, numeric], axis=1) if count else np.zeros((0, INPUT_DIM), np.float32)
        out.append({"matrix": matrix, "numeric": numeric, "onehot": onehot, "details": features["details"],
                    "rows": features["rows"]})
    return out


_shared: SemanticEncoder | None = None


def shared_encoder() -> SemanticEncoder:
    global _shared
    if _shared is None:
        _shared = SemanticEncoder()
    return _shared
