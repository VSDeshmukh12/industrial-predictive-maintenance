"""Pre-registered model-selection rule.

1. Primary metric: mean PR-AUC over identical CV folds.
2. Among candidates whose mean PR-AUC is within 1 standard error (paired differences
   vs the best, same folds) of the best, choose the SIMPLEST by `complexity`.
   (Folds overlap, so the SE is optimistic; the rule is a tie-breaker, not a test.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_diff_stats(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    d = (a - b).to_numpy()
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def select_parsimonious(fold_scores: dict[str, pd.Series], complexity: dict[str, int], se_mult: float = 1.0):
    """fold_scores: candidate -> per-fold PR-AUC (aligned index). Returns (winner, table)."""
    means = {k: v.mean() for k, v in fold_scores.items()}
    best = max(means, key=means.get)
    rows = []
    for k, v in fold_scores.items():
        diff, se = (0.0, 0.0) if k == best else paired_diff_stats(fold_scores[best], v)
        rows.append({"candidate": k, "mean_pr_auc": means[k], "diff_vs_best": diff, "paired_se": se,
                     "within_1se": (diff <= se_mult * se) or k == best, "complexity": complexity[k]})
    tab = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False).reset_index(drop=True)
    pool = tab[tab["within_1se"]]
    winner = pool.sort_values(["complexity", "mean_pr_auc"], ascending=[True, False]).iloc[0]["candidate"]
    return winner, tab


GATE_TOLERANCE = 0.03   # fixed validation-gate tolerance (must not be relaxed to obtain a passing model)


def grouped_gate(random_pr_auc: float, udi_block_pr_auc: float, tool_cycle_pr_auc: float,
                 tol: float = GATE_TOLERANCE) -> bool:
    """True iff PR-AUC drops by no more than `tol` under BOTH grouped CV schemes vs random CV."""
    return (random_pr_auc - udi_block_pr_auc <= tol) and (random_pr_auc - tool_cycle_pr_auc <= tol)
