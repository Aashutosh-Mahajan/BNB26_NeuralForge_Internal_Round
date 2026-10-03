"""M2: LightGBM step ranker on ~20 numeric features + node type, with TreeSHAP reasons."""
from __future__ import annotations

import numpy as np

from ..features import FEATURE_NAMES
from ..features.semantic import NODE_TYPES

TABULAR_NAMES = FEATURE_NAMES + [f"node_{t}" for t in NODE_TYPES]


def tabular(numeric: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    return np.concatenate([numeric, onehot], axis=1).astype(np.float32)


def train_lightgbm(x_train, y_train, x_val, y_val, seed=42, drop: list[str] | None = None):
    import lightgbm as lgb
    x_train, x_val = x_train.copy(), x_val.copy()
    for name in drop or []:
        x_train[:, TABULAR_NAMES.index(name)] = 0
        x_val[:, TABULAR_NAMES.index(name)] = 0
    positives = max(1, int(y_train.sum()))
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 10,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "lambda_l2": 1.0,
              "scale_pos_weight": min(20.0, (len(y_train) - positives) / positives), "seed": seed,
              "verbose": -1, "deterministic": True, "num_threads": 4}
    train = lgb.Dataset(x_train, y_train, feature_name=TABULAR_NAMES)
    valid = lgb.Dataset(x_val, y_val, reference=train)
    booster = lgb.train(params, train, num_boost_round=600, valid_sets=[valid],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    return booster


def predict(booster, x: np.ndarray) -> np.ndarray:
    return booster.predict(x, num_iteration=booster.best_iteration or None)


def shap_values(booster, x: np.ndarray) -> np.ndarray:
    """Exact TreeSHAP contributions (last column is the expected value)."""
    return booster.predict(x, num_iteration=booster.best_iteration or None, pred_contrib=True)
