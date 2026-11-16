"""Leakage-safe feature selection and preprocessing.

Design
------
* `select_model_inputs` is the single door from the full table to model inputs.
* `LeakageGuard` is the FIRST step of every pipeline, so the same protection
  holds at training time, in cross-validation, and at app inference time.
* Scaling / encoding live inside the pipeline, so they are fitted only on the
  training portion of each fold (no preprocessing leakage).
* The target and the failure-mode labels are returned by separate functions
  and are never part of X.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pm import config


class LeakageError(ValueError):
    """A forbidden (leaky or identifier) column reached a model input."""


def check_features(columns) -> None:
    """Raise LeakageError unless `columns` is exactly the allowed input set."""
    cols = list(columns)
    leaked = sorted(set(cols) & config.FORBIDDEN_FEATURES)
    if leaked:
        raise LeakageError(f"Forbidden columns in model input: {leaked}")
    unexpected = sorted(set(cols) - set(config.MODEL_FEATURES))
    if unexpected:
        raise LeakageError(f"Columns not in the approved feature list: {unexpected}")
    missing = sorted(set(config.MODEL_FEATURES) - set(cols))
    if missing:
        raise LeakageError(f"Missing required model inputs: {missing}")
    if len(cols) != len(set(cols)):
        raise LeakageError("Duplicate columns in model input")


def select_model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Return X = approved features only (copy, fixed column order)."""
    missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
    if missing:
        raise LeakageError(f"Missing required model inputs: {missing}")
    X = df.loc[:, list(config.MODEL_FEATURES)].copy()
    check_features(X.columns)
    return X


def get_target(df: pd.DataFrame) -> pd.Series:
    return df[config.TARGET].astype(int).copy()


def get_mode_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Failure-mode labels for the OPTIONAL secondary analysis only.

    These must never be used as inputs to the main model.
    """
    return df.loc[:, list(config.MODE_COLUMNS)].astype(int).copy()


class LeakageGuard(BaseEstimator, TransformerMixin):
    """Pass-through transformer that enforces the approved input columns."""

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise LeakageError("LeakageGuard requires a pandas DataFrame with named columns")
        check_features(X.columns)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise LeakageError("LeakageGuard requires a pandas DataFrame with named columns")
        check_features(X.columns)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Numeric scaling (optional) + fixed-category one-hot of `type`.

    Categories are fixed ({L, M, H}) so the encoding never depends on the data
    seen during fit, and unseen categories raise instead of silently mapping.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric, list(config.NUMERIC_FEATURES)),
            ("cat", OneHotEncoder(categories=[list(config.TYPE_CATEGORIES)],
                                  handle_unknown="error", sparse_output=False),
             list(config.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator, scale_numeric: bool = True) -> Pipeline:
    """guard -> preprocess -> estimator. The estimator is cloned (never mutated)."""
    return Pipeline([
        ("leakage_guard", LeakageGuard()),
        ("preprocess", build_preprocessor(scale_numeric)),
        ("model", clone(estimator)),
    ])
