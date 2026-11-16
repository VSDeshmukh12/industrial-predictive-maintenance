"""Hyper-parameter search spaces (controlled budget: random search, fixed seed)."""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import ParameterSampler

from pm import config

SPACES = {
    "logreg": {"C": loguniform(1e-3, 1e2)},
    "random_forest": {"n_estimators": randint(200, 501), "max_depth": [None, 6, 8, 12, 16],
                      "min_samples_leaf": randint(1, 11), "max_features": ["sqrt", "log2", 0.5, 0.8]},
    "extra_trees": {"n_estimators": randint(200, 501), "max_depth": [None, 6, 8, 12, 16],
                    "min_samples_leaf": randint(1, 11), "max_features": ["sqrt", "log2", 0.5, 0.8]},
    "xgboost": {"n_estimators": randint(100, 501), "learning_rate": loguniform(0.02, 0.3),
                "max_depth": randint(2, 8), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                "min_child_weight": randint(1, 11), "reg_lambda": loguniform(0.1, 20)},
    "lightgbm": {"n_estimators": randint(100, 501), "learning_rate": loguniform(0.02, 0.3),
                 "num_leaves": randint(4, 49), "min_child_samples": randint(5, 61),
                 "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.5, 0.5),
                 "reg_lambda": loguniform(0.1, 20)},
    "catboost": {"iterations": randint(150, 501), "depth": randint(3, 9), "learning_rate": loguniform(0.02, 0.3),
                 "l2_leaf_reg": loguniform(1, 20)},
}


def sample_params(model: str, n: int, seed: int = config.RANDOM_SEED) -> list[dict]:
    out = []
    for p in ParameterSampler(SPACES[model], n_iter=n, random_state=seed):
        out.append({k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()})
    return out
