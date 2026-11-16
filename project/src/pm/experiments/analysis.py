"""Turn cached CV experiments into comparison tables (all on the training partition)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pm.evaluation.thresholds import crossfit_threshold_metrics
from pm.experiments.benchmark import CACHE_DIR, fold_assignment

THRESH_COLS = ["recall", "precision", "f1", "f2", "mcc", "false_alarms_per_1000", "threshold"]


def load_stage(stage: str) -> pd.DataFrame:
    """All fold-level rows of a stage (adds params if the spec json exists)."""
    frames = []
    for csv in sorted((CACHE_DIR / stage).glob("*.csv")):
        f = pd.read_csv(csv, dtype={"config_id": str})
        js = csv.with_suffix(".json")
        f["params"] = json.dumps(json.loads(js.read_text()).get("params") or {}, sort_keys=True) if js.exists() else "{}"
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def summarize_stage(stage: str, train: pd.DataFrame, repeats, with_thresholds: bool = True) -> pd.DataFrame:
    """One row per config: PR-AUC etc. (mean, sd over folds) + metrics at cross-fitted F2 thresholds."""
    folds = load_stage(stage)
    ids = fold_assignment(train, tuple(repeats))
    y = train["machine_failure"].to_numpy()
    rows = []
    for cid, g in folds.groupby("config_id", sort=False):
        r = {"config_id": cid, "model": g.model.iloc[0], "strategy": g.strategy.iloc[0],
             "feature_set": g.feature_set.iloc[0], "params": g.params.iloc[0], "n_folds": len(g)}
        for m in ("pr_auc", "roc_auc", "brier"):
            r[m], r[m + "_sd"] = g[m].mean(), g[m].std(ddof=1)
        r["fit_seconds"] = g.fit_seconds.mean()
        if with_thresholds:
            oof = np.load(CACHE_DIR / stage / f"{cid}.npz")["oof"]
            tm = crossfit_threshold_metrics(y, oof, ids)
            for c in THRESH_COLS:
                r[f"cf_{c}"] = tm[c].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def fold_series(folds: pd.DataFrame, config_id: str, col: str = "pr_auc") -> pd.Series:
    g = folds[folds.config_id == config_id].sort_values(["repeat", "fold"])
    return g.set_index(["repeat", "fold"])[col]


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
                                      for v in row) + " |")
    return "\n".join(out)
