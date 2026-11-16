"""Untuned baseline estimators (Phase 2). No resampling, no tuning.

These exist to set a floor and to validate the evaluation machinery. Their
hyper-parameters are conventional defaults, deliberately NOT optimised.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from pm import config
from pm.features.preprocessing import build_pipeline


def get_baselines(seed: int = config.RANDOM_SEED) -> dict:
    return {
        "dummy_prior": build_pipeline(DummyClassifier(strategy="prior")),
        "logreg": build_pipeline(LogisticRegression(max_iter=2000, random_state=seed)),
        "logreg_balanced": build_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        # Trees do not need scaling; skip it so the tree sees raw units.
        "tree_d4_balanced": build_pipeline(
            DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, class_weight="balanced",
                                   random_state=seed), scale_numeric=False),
    }
