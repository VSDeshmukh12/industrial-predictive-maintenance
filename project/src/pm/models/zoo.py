"""Model zoo, resampling-strategy plumbing and leakage-safe experiment pipelines.

Pipeline = leakage_guard -> [resample] -> engineer -> preprocess -> model.
Resampling only runs at fit time on the training fold. Class weighting is
implemented uniformly via balanced sample weights computed from the training fold.
"""
from __future__ import annotations

import warnings

from catboost import CatBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
import numpy as np

from pm import config
from pm.features.engineering import FeatureEngineer
from pm.features.preprocessing import LeakageGuard
from pm.models.resampling import RawSpaceResampler

STRATEGIES = ("none", "class_weight", "smote", "smotenc", "smote_enn", "smotenc_enn")
# Complexity order used by the parsimony rule (simplest first).
STRATEGY_COMPLEXITY = {s: i for i, s in enumerate(STRATEGIES)}
RESAMPLING = ("smote", "smotenc", "smote_enn", "smotenc_enn")

MODELS = ("logreg", "decision_tree", "random_forest", "extra_trees", "xgboost", "lightgbm",
          "catboost", "balanced_rf", "svm_rbf")


class BalancedSampleWeight(ClassifierMixin, BaseEstimator):
    """Fit `estimator` with 'balanced' sample weights computed from the training fold."""

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, sample_weight=compute_sample_weight("balanced", np.asarray(y)))
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


def default_params(name: str) -> dict:
    return {
        "logreg": dict(C=1.0),
        "decision_tree": dict(max_depth=4, min_samples_leaf=20),
        "random_forest": dict(n_estimators=300, min_samples_leaf=1, max_features="sqrt"),
        "extra_trees": dict(n_estimators=300, min_samples_leaf=1, max_features="sqrt"),
        "xgboost": dict(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=1, reg_lambda=1.0),
        "lightgbm": dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=0.0),
        "catboost": dict(iterations=300, depth=6, learning_rate=0.1, l2_leaf_reg=3.0),
        "balanced_rf": dict(n_estimators=300, min_samples_leaf=1),
        "svm_rbf": dict(C=1.0, gamma="scale"),
    }[name]


def make_base_estimator(name: str, params: dict | None = None, seed: int = config.RANDOM_SEED):
    p = {**default_params(name), **(params or {})}
    if name == "logreg":
        return LogisticRegression(max_iter=3000, random_state=seed, **p)
    if name == "decision_tree":
        return DecisionTreeClassifier(random_state=seed, **p)
    if name == "random_forest":
        return RandomForestClassifier(n_jobs=1, random_state=seed, **p)
    if name == "extra_trees":
        return ExtraTreesClassifier(n_jobs=1, random_state=seed, **p)
    if name == "xgboost":
        return XGBClassifier(tree_method="hist", n_jobs=1, random_state=seed, verbosity=0,
                             eval_metric="logloss", **p)
    if name == "lightgbm":
        return LGBMClassifier(subsample_freq=1, n_jobs=1, random_state=seed, verbose=-1, **p)
    if name == "catboost":
        return CatBoostClassifier(thread_count=1, random_seed=seed, verbose=0,
                                  allow_writing_files=False, **p)
    if name == "balanced_rf":
        return BalancedRandomForestClassifier(n_jobs=1, random_state=seed,
                                              sampling_strategy="all", replacement=True,
                                              bootstrap=False, **p)
    if name == "svm_rbf":
        # `SVC(probability=True)` is deprecated in scikit-learn 1.9; use Platt scaling explicitly.
        return CalibratedClassifierCV(SVC(kernel="rbf", random_state=seed, **p),
                                      method="sigmoid", cv=3, ensemble=False)
    raise KeyError(name)


def build_experiment_preprocessor() -> ColumnTransformer:
    """Scale every numeric column (raw + engineered); one-hot `type` with fixed categories."""
    return ColumnTransformer(
        [("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
         ("cat", OneHotEncoder(categories=[list(config.TYPE_CATEGORIES)], handle_unknown="error",
                               sparse_output=False), list(config.CATEGORICAL_FEATURES))],
        remainder="drop", verbose_feature_names_out=False)


def build_experiment_pipeline(model: str, strategy: str = "none", feature_set: str = "raw",
                              params: dict | None = None, resampler_params: dict | None = None,
                              seed: int = config.RANDOM_SEED) -> ImbPipeline:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    if model == "balanced_rf" and strategy != "none":
        raise ValueError("balanced_rf balances internally; use strategy='none'")
    est = make_base_estimator(model, params, seed)
    if strategy == "class_weight":
        est = BalancedSampleWeight(est)
    steps = [("leakage_guard", LeakageGuard())]
    if strategy in RESAMPLING:
        steps.append(("resample", RawSpaceResampler(method=strategy, random_state=seed,
                                                    **(resampler_params or {}))))
    steps += [("engineer", FeatureEngineer(feature_set)),
              ("preprocess", build_experiment_preprocessor()),
              ("model", est)]
    return ImbPipeline(steps)
