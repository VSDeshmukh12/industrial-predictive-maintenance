"""Feature sets for the raw-vs-engineered ablation.

All derived features are deterministic functions of the six approved RAW inputs,
so they cannot introduce target leakage. They run INSIDE the pipeline, after any
resampling, so synthetic samples always satisfy the formulas exactly.

Sets
----
raw         : the six approved inputs only.
paper       : raw + the three features defined in the research paper
              (temp_diff, power = torque*rpm [not watts], wear_rate = wear/rpm).
physics     : raw + delta_t, power_w (watts, torque*rpm*2*pi/60), overstrain (wear*torque).
rule_aware  : physics + overstrain_ratio and power_margin, which embed the
              generator thresholds documented by UCI (OSF limits by Type; power
              band 3500-9000 W). OPTIMISTIC REFERENCE ONLY: these encode the
              synthetic generator's rules and would not exist for a real plant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from pm import config

OSF_LIMIT = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}   # minNm, UCI documentation
POWER_BAND_W = (3_500.0, 9_000.0)                           # W, UCI documentation

DERIVED = {
    "raw": (),
    "paper": ("temp_diff", "power_paper", "wear_rate"),
    "physics": ("delta_t", "power_w", "overstrain"),
    "rule_aware": ("delta_t", "power_w", "overstrain", "overstrain_ratio", "power_margin"),
}
FEATURE_SETS = tuple(DERIVED)


def _compute(X: pd.DataFrame, name: str) -> pd.Series:
    air, proc = X["air_temp_k"], X["process_temp_k"]
    rpm, tq, wear = X["rot_speed_rpm"], X["torque_nm"], X["tool_wear_min"]
    if name in ("temp_diff", "delta_t"):
        return proc - air
    if name == "power_paper":
        return tq * rpm
    if name == "wear_rate":
        return wear / rpm
    if name == "power_w":
        return tq * rpm * 2.0 * np.pi / 60.0
    if name == "overstrain":
        return wear * tq
    if name == "overstrain_ratio":
        return wear * tq / X["type"].map(OSF_LIMIT).astype(float)
    if name == "power_margin":
        p = tq * rpm * 2.0 * np.pi / 60.0
        return np.minimum(p - POWER_BAND_W[0], POWER_BAND_W[1] - p)
    raise KeyError(name)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Append the derived columns of `feature_set` to the raw inputs (stateless)."""

    def __init__(self, feature_set: str = "raw"):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        if self.feature_set not in DERIVED:
            raise ValueError(f"Unknown feature_set {self.feature_set!r}; choose from {FEATURE_SETS}")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        out = X.copy()
        for name in DERIVED[self.feature_set]:
            out[name] = _compute(X, name).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(list(config.MODEL_FEATURES) + list(DERIVED[self.feature_set]), dtype=object)
