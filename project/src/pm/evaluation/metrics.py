"""Imbalance-aware evaluation metrics.

Conventions
-----------
* Positive class (1) = machine failure.
* Undefined ratios (e.g. precision when nothing is predicted positive) are 0.0.
* PR-AUC is average precision; its no-skill baseline equals the prevalence.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score,
)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def metrics_from_counts(c: dict[str, int]) -> dict[str, float]:
    tn, fp, fn, tp = c["tn"], c["fp"], c["fn"], c["tp"]
    n = tn + fp + fn + tp
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    f2 = _safe_div(5 * precision * recall, 4 * precision + recall)
    specificity = _safe_div(tn, tn + fp)
    mcc_den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_den)
    return {
        **c,
        "accuracy": _safe_div(tp + tn, n),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": _safe_div(fp, fp + tn),
        "f1": f1,
        "f2": f2,
        "mcc": mcc,
        "alerts_per_1000": 1000.0 * _safe_div(tp + fp, n),
        "false_alarms_per_1000": 1000.0 * _safe_div(fp, n),
    }


def threshold_metrics(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    """Decision-rule metrics: predict failure when score >= threshold."""
    y_score = np.asarray(y_score, dtype=float)
    return metrics_from_counts(confusion_counts(y_true, (y_score >= threshold).astype(int)))


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    """Best achievable recall over all thresholds with precision >= min_precision (0 if none)."""
    precision, recall, _ = precision_recall_curve(np.asarray(y_true), np.asarray(y_score))
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def ranking_metrics(y_true, y_score) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.min() == y_true.max():
        raise ValueError("y_true contains a single class; ranking metrics are undefined")
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, np.clip(y_score, 0.0, 1.0))),
        "prevalence": float(y_true.mean()),
    }


def evaluate_scores(y_true, y_score, threshold: float = 0.5,
                    precision_floors: tuple[float, ...] = (0.5, 0.7)) -> dict[str, float]:
    """Threshold-free + threshold-based metrics in one flat dict."""
    out = ranking_metrics(y_true, y_score)
    for pf in precision_floors:
        out[f"recall_at_precision_{int(round(pf * 100))}"] = recall_at_precision(y_true, y_score, pf)
    th = threshold_metrics(y_true, y_score, threshold)
    out.update({f"{k}@{threshold:g}": v for k, v in th.items()})
    return out
