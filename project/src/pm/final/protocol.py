"""One-shot hold-out protocol: lock, evaluation, failure-mode breakdown, uncertainty.

This module never fits anything and never fetches the hold-out itself; the evaluation script passes
the already-materialised frame. `evaluate_holdout` only calls `predict_proba` on the frozen bundle.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pm import config
from pm.evaluation.calibration import calibration_metrics
from pm.evaluation.metrics import (confusion_counts, metrics_from_counts, ranking_metrics,
                                   threshold_metrics, wilson_interval)
from pm.final import spec

MODES = ("twf", "hdf", "pwf", "osf", "rnf")
METRIC_KEYS = ("pr_auc", "roc_auc", "brier", "precision", "recall", "f1", "f2", "mcc", "false_alarms_per_1000")


class HoldoutAlreadyEvaluated(RuntimeError):
    """The protected hold-out has already been consumed; a second evaluation is refused."""


def acquire_holdout_lock(path: Path, payload: dict) -> None:
    """Atomically create the lock file. Raises if it exists (hold-out already used)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o444)
    except FileExistsError:
        raise HoldoutAlreadyEvaluated(
            f"{path} exists: the protected hold-out was already evaluated. "
            "It must not be evaluated again (that would turn it into a validation set).") from None
    with os.fdopen(fd, "w") as fh:
        json.dump({**payload, "acquired_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "status": "started"}, fh, indent=2, sort_keys=True)


def _all_metrics(y, p, tau) -> dict:
    out = ranking_metrics(y, p)
    th = threshold_metrics(y, p, tau)
    out.update({k: th[k] for k in th})
    return out


def _stratified_bootstrap(y, p, tau, n_boot: int, seed: int) -> dict:
    """Percentile 95% CIs; resamples positives and negatives separately so every replicate keeps the prevalence."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rows = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        m = _all_metrics(y[idx], p[idx], tau)
        rows.append([m[k] for k in METRIC_KEYS])
    a = np.asarray(rows)
    lo, hi = np.percentile(a, [2.5, 97.5], axis=0)
    return {k: [float(l), float(h)] for k, l, h in zip(METRIC_KEYS, lo, hi)}


def failure_mode_table(df: pd.DataFrame, p: np.ndarray, tau: float) -> pd.DataFrame:
    """Recall at tau per failure mode (labels used for EVALUATION only), with Wilson 95% CIs."""
    y = df[config.TARGET].to_numpy().astype(int)
    alert = p >= tau
    anym = df[list(MODES)].sum(axis=1).to_numpy() > 0
    groups = {m: (df[m].to_numpy() == 1) & (y == 1) for m in MODES}
    groups["no_mode_flag"] = (y == 1) & ~anym
    groups["all_failures"] = y == 1
    n_alerts = int(alert.sum())
    rows = []
    for g, mask in groups.items():
        n = int(mask.sum())
        k = int((alert & mask).sum())
        lo, hi = wilson_interval(k, n) if n else (np.nan, np.nan)
        rows.append({"group": g, "n_failures": n, "detected": k, "missed": n - k,
                     "recall": (k / n) if n else np.nan, "wilson_lo": lo, "wilson_hi": hi,
                     "share_of_alerts": (k / n_alerts) if n_alerts else np.nan,
                     "mean_prob_failures": float(p[mask].mean()) if n else np.nan})
    return pd.DataFrame(rows)


def advisory_table(df: pd.DataFrame, advisory: np.ndarray) -> pd.DataFrame:
    """How the SEPARATE tool-wear advisory relates to TWF labels (descriptive; advisory is not a classifier)."""
    twf = df["twf"].to_numpy() == 1
    rows = []
    for name, flagged in (("advisory (wear >= %g)" % spec.TWF_ADVISORY_WEAR_MIN, advisory != "none"),
                          ("elevated (wear >= %g)" % spec.TWF_ELEVATED_WEAR_MIN, advisory == "elevated")):
        k, n = int((flagged & twf).sum()), int(twf.sum())
        lo, hi = wilson_interval(k, n) if n else (np.nan, np.nan)
        rows.append({"level": name, "rows_flagged": int(flagged.sum()),
                     "flagged_per_1000": 1000.0 * float(flagged.mean()),
                     "twf_total": n, "twf_inside_flagged": k, "twf_recall_of_advisory": k / n if n else np.nan,
                     "wilson_lo": lo, "wilson_hi": hi,
                     "twf_rate_among_flagged": (k / int(flagged.sum())) if flagged.any() else np.nan})
    return pd.DataFrame(rows)


def evaluate_holdout(bundle, holdout_df: pd.DataFrame, *, n_boot: int = 2000, seed: int = config.RANDOM_SEED) -> dict:
    """Score the frozen bundle on `holdout_df`. No fitting, no threshold search, no re-calibration."""
    y = holdout_df[config.TARGET].to_numpy().astype(int)
    tau = bundle.threshold
    assess = bundle.assess(holdout_df)
    p = assess["failure_probability"].to_numpy()
    metrics = _all_metrics(y, p, tau)
    counts = confusion_counts(y, (p >= tau).astype(int))
    cal = calibration_metrics(y, p)
    ci = _stratified_bootstrap(y, p, tau, n_boot, seed)
    n_alert, n_pos = int((p >= tau).sum()), int(y.sum())
    k_tp = counts["tp"]
    ci["recall_wilson"] = list(wilson_interval(k_tp, n_pos))
    ci["precision_wilson"] = list(wilson_interval(k_tp, n_alert)) if n_alert else [np.nan, np.nan]
    modes = failure_mode_table(holdout_df, p, tau)
    adv = advisory_table(holdout_df, assess["tool_wear_advisory"].to_numpy())
    by_type = []
    for t in config.TYPE_CATEGORIES:
        m = (holdout_df["type"] == t).to_numpy()
        c = confusion_counts(y[m], (p[m] >= tau).astype(int))
        mm = metrics_from_counts(c)
        by_type.append({"type": t, "rows": int(m.sum()), "failures": int(y[m].sum()), **c,
                        "recall": mm["recall"], "precision": mm["precision"]})
    return {"threshold": tau, "n_rows": int(len(y)), "n_failures": n_pos, "prevalence": float(y.mean()),
            "no_skill_pr_auc": float(y.mean()), "metrics": metrics, "confusion_matrix": counts,
            "calibration": cal, "bootstrap_95ci": ci, "bootstrap": {"n": n_boot, "seed": seed, "type": "stratified percentile"},
            "failure_modes": modes, "tool_wear_advisory": adv, "by_type": pd.DataFrame(by_type),
            "predictions": assess}
