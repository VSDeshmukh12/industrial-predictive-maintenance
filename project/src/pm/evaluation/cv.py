"""Repeated stratified cross-validation on the TRAINING partition only.

Each fold clones the (full) pipeline, so scaling/encoding (and, in Phase 3,
resampling) are fitted on the fold's training rows only. Out-of-fold scores are
kept for later threshold / calibration analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone

from pm import config
from pm.data.splitting import make_cv_splits
from pm.evaluation.metrics import evaluate_scores
from pm.features.preprocessing import get_target, select_model_inputs


@dataclass
class CVResult:
    name: str
    fold_metrics: pd.DataFrame   # one row per (repeat, fold)
    oof_scores: np.ndarray       # shape (n_repeats, n_rows); every row scored once per repeat
    y: np.ndarray                # shape (n_rows,)


def cross_validate_model(estimator, df_train: pd.DataFrame, name: str,
                         n_splits: int = config.CV_N_SPLITS, n_repeats: int = config.CV_N_REPEATS,
                         seed: int = config.RANDOM_SEED, threshold: float = 0.5) -> CVResult:
    X = select_model_inputs(df_train)
    y = get_target(df_train)
    oof = np.full((n_repeats, len(df_train)), np.nan)
    rows = []
    for repeat, fold, tr, va in make_cv_splits(df_train, n_splits, n_repeats, seed):
        model = clone(estimator)
        model.fit(X.iloc[tr], y.iloc[tr])
        score = model.predict_proba(X.iloc[va])[:, 1]
        oof[repeat, va] = score
        rows.append({
            "model": name, "repeat": repeat, "fold": fold,
            "n_train": len(tr), "n_val": len(va), "pos_val": int(y.iloc[va].sum()),
            **evaluate_scores(y.iloc[va], score, threshold),
        })
    return CVResult(name, pd.DataFrame(rows), oof, y.to_numpy())


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over folds (SD understates true generalisation variance: folds overlap)."""
    id_cols = ["model", "repeat", "fold", "n_train", "n_val", "pos_val"]
    metric_cols = [c for c in fold_metrics.columns if c not in id_cols]
    g = fold_metrics.groupby("model", sort=False)[metric_cols]
    mean, std = g.mean(), g.std(ddof=1)
    out = pd.concat({"mean": mean, "std": std}, axis=1).swaplevel(axis=1).sort_index(axis=1)
    return out


def pooled_oof_metrics(result: CVResult, threshold: float = 0.5) -> pd.DataFrame:
    """One row per repeat: metrics computed on the pooled out-of-fold scores of that repeat."""
    rows = []
    for r in range(result.oof_scores.shape[0]):
        rows.append({"model": result.name, "repeat": r, **evaluate_scores(result.y, result.oof_scores[r], threshold)})
    return pd.DataFrame(rows)
