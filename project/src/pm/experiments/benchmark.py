"""Cached, resumable CV experiments on the TRAINING partition (identical folds everywhere)."""
from __future__ import annotations

import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from pm import config
from pm.data.splitting import make_cv_splits
from pm.evaluation.metrics import evaluate_scores
from pm.features.preprocessing import get_target, select_model_inputs
from pm.models.zoo import build_experiment_pipeline

PHASE3_DIR = config.ARTIFACTS_DIR / "phase3"
CACHE_DIR = PHASE3_DIR / "cache"
ALL_REPEATS = tuple(range(config.CV_N_REPEATS))


def config_id(spec: dict, repeats) -> str:
    blob = json.dumps({**spec, "repeats": list(repeats), "seed": config.RANDOM_SEED}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def fold_assignment(train: pd.DataFrame, repeats=ALL_REPEATS):
    """(len(repeats), n) array of fold ids; identical for every experiment."""
    ids = np.full((len(repeats), len(train)), -1, dtype=np.int8)
    pos = {r: i for i, r in enumerate(repeats)}
    for r, f, tr, va in make_cv_splits(train):
        if r in pos:
            ids[pos[r], va] = f
    return ids


def run_experiment(spec: dict, train: pd.DataFrame, repeats, stage: str, force: bool = False):
    """spec: {model, strategy, feature_set, params?, resampler_params?}. Returns (fold_df, oof, y)."""
    cid = config_id(spec, repeats)
    d = CACHE_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    f_csv, f_npz = d / f"{cid}.csv", d / f"{cid}.npz"
    y = get_target(train).to_numpy()
    if f_csv.exists() and f_npz.exists() and not force:
        return pd.read_csv(f_csv, dtype={"config_id": str}), np.load(f_npz)["oof"], y
    X = select_model_inputs(train)
    ys = pd.Series(y)
    pos = {r: i for i, r in enumerate(repeats)}
    oof = np.full((len(repeats), len(train)), np.nan)
    rows = []
    est = build_experiment_pipeline(spec["model"], spec["strategy"], spec["feature_set"],
                                    spec.get("params"), spec.get("resampler_params"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r, f, tr, va in make_cv_splits(train):
            if r not in pos:
                continue
            t0 = time.time()
            m = clone(est).fit(X.iloc[tr], ys.iloc[tr])
            s = m.predict_proba(X.iloc[va])[:, 1]
            oof[pos[r], va] = s
            rows.append({"config_id": cid, "model": spec["model"], "strategy": spec["strategy"],
                         "feature_set": spec["feature_set"], "repeat": r, "fold": f,
                         "fit_seconds": time.time() - t0, "pos_val": int(ys.iloc[va].sum()),
                         **evaluate_scores(ys.iloc[va], s)})
    df = pd.DataFrame(rows)
    df.to_csv(f_csv, index=False, float_format="%.10g")
    np.savez_compressed(f_npz, oof=oof)
    (d / f"{cid}.json").write_text(json.dumps({**spec, "repeats": list(repeats)}, sort_keys=True, default=str))
    return df, oof, y


def mean_sd(df: pd.DataFrame, col: str = "pr_auc"):
    return float(df[col].mean()), float(df[col].std(ddof=1))
