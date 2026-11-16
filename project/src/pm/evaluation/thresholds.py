"""Decision-threshold analysis on out-of-fold (OOF) scores.

Provisional objective: maximise F2 (recall weighted 2x precision). To avoid
optimistic reporting, `crossfit_threshold_metrics` selects each fold's threshold
on the OOF scores of the OTHER folds of the same repeat and applies it to the fold.
(Caveat: those OOF scores came from models that saw the held-out fold in training,
so a small residual optimism remains; the protected hold-out in Phase 4 is the
unbiased check.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pm.evaluation.metrics import threshold_metrics


def f2_by_threshold(y, s):
    """Return (thresholds, f2) for predict-positive-if score >= threshold, over all unique scores."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    ss, yy = s[order], y[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    last = np.r_[ss[1:] != ss[:-1], True]            # only cut between distinct scores
    tp, fp, thr = tp[last], fp[last], ss[last]
    fn = y.sum() - tp
    denom = 5 * tp + 4 * fn + fp
    f2 = np.divide(5 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return thr, f2


def best_f2_threshold(y, s) -> tuple[float, float]:
    thr, f2 = f2_by_threshold(y, s)
    i = int(np.argmax(f2))
    return float(thr[i]), float(f2[i])


def crossfit_threshold_metrics(y, oof, fold_ids) -> pd.DataFrame:
    """One row per (repeat, fold): metrics at a threshold chosen on the other folds."""
    y = np.asarray(y).astype(int)
    rows = []
    for r in range(oof.shape[0]):
        for k in np.unique(fold_ids[r]):
            tr, va = fold_ids[r] != k, fold_ids[r] == k
            thr, _ = best_f2_threshold(y[tr], oof[r][tr])
            rows.append({"repeat": r, "fold": int(k), "threshold": thr,
                         **threshold_metrics(y[va], oof[r][va], thr)})
    return pd.DataFrame(rows)


def tradeoff_table(y, oof, thresholds=None) -> pd.DataFrame:
    """Precision/recall trade-off averaged over repeats (pooled OOF per repeat)."""
    y = np.asarray(y).astype(int)
    if thresholds is None:
        thresholds = np.unique(np.r_[np.linspace(0.02, 0.98, 49), 0.5])
    rows = []
    for t in thresholds:
        per = pd.DataFrame([threshold_metrics(y, oof[r], t) for r in range(oof.shape[0])])
        m = per.mean()
        rows.append({"threshold": float(t), **{k: float(m[k]) for k in
                     ("recall", "precision", "f1", "f2", "mcc", "false_alarms_per_1000", "alerts_per_1000", "fn", "fp", "tp")}})
    return pd.DataFrame(rows)


def operating_points_for_recall(y, oof, targets=(0.80, 0.85, 0.90, 0.95)) -> pd.DataFrame:
    """For each recall target: the highest threshold meeting it (max precision) on pooled OOF, averaged over repeats."""
    y = np.asarray(y).astype(int)
    rows = []
    for tgt in targets:
        vals = []
        for r in range(oof.shape[0]):
            s = oof[r]
            thr_c = np.unique(s)[::-1]                      # descending
            best = None
            for t in thr_c:
                m = threshold_metrics(y, s, t)
                if m["recall"] >= tgt:
                    best = (t, m); break
            if best:
                vals.append({"threshold": best[0], **{k: best[1][k] for k in ("recall", "precision", "f2", "false_alarms_per_1000")}})
        if vals:
            rows.append({"recall_target": tgt, **pd.DataFrame(vals).mean().to_dict()})
    return pd.DataFrame(rows)
