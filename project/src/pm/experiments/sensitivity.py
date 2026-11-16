"""Grouped / blocked CV schemes to test whether random folds are optimistic.

The CSV was generated sequentially (tool wear resets along UDI; air temperature is a random walk),
so neighbouring rows are related. Schemes (TRAIN partition only):
  udi_block  : GroupKFold(5) on blocks of 400 consecutive UDIs.
  tool_cycle : StratifiedGroupKFold(5) with one group per tool life. Cycle boundaries are inferred from
               TRAIN rows only: a drop in tool wear between consecutive training rows marks a tool change.
UDI is used ONLY to order rows / define groups, never as a feature.
"""
from __future__ import annotations

import hashlib
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from pm.evaluation.metrics import evaluate_scores
from pm.experiments.benchmark import CACHE_DIR
from pm.features.preprocessing import get_target, select_model_inputs
from pm.models.zoo import build_experiment_pipeline

BLOCK = 400


def tool_cycle_ids(train: pd.DataFrame) -> np.ndarray:
    """Cycle id per row of `train` (aligned to its current order), from train rows only."""
    order = np.argsort(train["udi"].to_numpy(), kind="mergesort")
    wear = train["tool_wear_min"].to_numpy()[order]
    cyc_sorted = np.concatenate([[0], np.cumsum(np.diff(wear) < 0)])
    out = np.empty(len(train), dtype=int)
    out[order] = cyc_sorted
    return out


def grouped_folds(train: pd.DataFrame, scheme: str, seeds=(0, 1, 2)):
    """Yield (repeat, fold, train_idx, val_idx)."""
    y = get_target(train).to_numpy()
    if scheme == "udi_block":
        groups = (train["udi"].to_numpy() - 1) // BLOCK
        for f, (tr, va) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(train)), y, groups)):
            yield 0, f, tr, va
    elif scheme == "tool_cycle":
        groups = tool_cycle_ids(train)
        for r, seed in enumerate(seeds):
            sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            for f, (tr, va) in enumerate(sgkf.split(np.zeros(len(train)), y, groups)):
                yield r, f, tr, va
    else:
        raise ValueError(scheme)


def run_grouped(spec: dict, train: pd.DataFrame, scheme: str, stage: str = "sens") -> pd.DataFrame:
    blob = json.dumps({**spec, "scheme": scheme}, sort_keys=True, default=str)
    cid = hashlib.sha1(blob.encode()).hexdigest()[:12]
    d = CACHE_DIR / stage
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{cid}.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"config_id": str})
    X, y = select_model_inputs(train), pd.Series(get_target(train).to_numpy())
    est = build_experiment_pipeline(spec["model"], spec["strategy"], spec["feature_set"], spec.get("params"))
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r, k, tr, va in grouped_folds(train, scheme):
            if y.iloc[va].sum() == 0:
                continue
            m = clone(est).fit(X.iloc[tr], y.iloc[tr])
            s = m.predict_proba(X.iloc[va])[:, 1]
            rows.append({"config_id": cid, "scheme": scheme, "model": spec["model"], "strategy": spec["strategy"],
                         "feature_set": spec["feature_set"], "repeat": r, "fold": k, "n_val": len(va),
                         "pos_val": int(y.iloc[va].sum()), **evaluate_scores(y.iloc[va], s)})
    df = pd.DataFrame(rows)
    df.to_csv(f, index=False, float_format="%.10g")
    return df
