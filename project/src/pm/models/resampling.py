"""Resampling in RAW feature space with correct categorical handling.

Why a custom sampler?
  * Engineered features (e.g. power = torque * rpm) are NONLINEAR in the raw
    inputs; interpolating them would give synthetic rows that violate the formulas.
    So we resample the six raw inputs and compute engineered features afterwards.
  * `type` is categorical. Interpolating one-hot dummies yields invalid fractional
    categories. Each method below ends with an exact category in {L, M, H}.
  * Distances need comparable scales, so numeric columns are standardised
    internally (fitted on the training rows passed in) and mapped back.

Methods
  smote       : SMOTE on [scaled numeric + one-hot type]; type decoded by argmax.
  smotenc     : SMOTENC on [scaled numeric + ordinal type]; type = neighbour mode.
  smote_enn   : `smote` then Edited Nearest Neighbours (cleans both classes), as in the paper.
  smotenc_enn : `smotenc` then ENN computed on the one-hot representation (an
                extension: a categorically-correct version of SMOTE-ENN).
Used only at fit time (imblearn pipelines skip samplers at predict time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.base import BaseEstimator

from pm import config

METHODS = ("smote", "smotenc", "smote_enn", "smotenc_enn")
_CATS = list(config.TYPE_CATEGORIES)


class RawSpaceResampler(BaseEstimator):
    def __init__(self, method: str = "smote", sampling_strategy="auto", k_neighbors: int = 5,
                 enn_neighbors: int = 3, random_state: int = config.RANDOM_SEED):
        self.method = method
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.enn_neighbors = enn_neighbors
        self.random_state = random_state

    # imblearn's Pipeline looks for fit_resample
    def fit_resample(self, X: pd.DataFrame, y):
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        num_cols = list(config.NUMERIC_FEATURES)
        num = X[num_cols].to_numpy(dtype=float)
        mu, sd = num.mean(axis=0), num.std(axis=0)
        sd[sd == 0] = 1.0
        zn = (num - mu) / sd
        code = X["type"].map({c: i for i, c in enumerate(_CATS)}).to_numpy()
        if np.isnan(code.astype(float)).any():
            raise ValueError("`type` contains values outside {L, M, H}")
        code = code.astype(int)
        y = np.asarray(y).astype(int)
        onehot = np.eye(len(_CATS))[code]

        if self.method in ("smote", "smote_enn"):
            Z = np.hstack([zn, onehot])
            Zr, yr = SMOTE(sampling_strategy=self.sampling_strategy, k_neighbors=self.k_neighbors,
                           random_state=self.random_state).fit_resample(Z, y)
            zn_r = Zr[:, :len(num_cols)]
            code_r = Zr[:, len(num_cols):].argmax(axis=1)          # decode fractional dummies
        else:
            Z = np.hstack([zn, code[:, None].astype(float)])
            Zr, yr = SMOTENC(categorical_features=[len(num_cols)], sampling_strategy=self.sampling_strategy,
                             k_neighbors=self.k_neighbors, random_state=self.random_state).fit_resample(Z, y)
            zn_r = Zr[:, :len(num_cols)]
            code_r = np.rint(Zr[:, len(num_cols)]).astype(int)

        if self.method.endswith("_enn"):
            Zenn = np.hstack([zn_r, np.eye(len(_CATS))[code_r]])
            enn = EditedNearestNeighbours(sampling_strategy="all", n_neighbors=self.enn_neighbors)
            _, yr = enn.fit_resample(Zenn, yr)
            keep = enn.sample_indices_
            zn_r, code_r = zn_r[keep], code_r[keep]

        out = pd.DataFrame(zn_r * sd + mu, columns=num_cols)
        out["type"] = pd.Categorical.from_codes(code_r, categories=_CATS).astype(str)
        out = out[list(config.MODEL_FEATURES)]
        return out, pd.Series(np.asarray(yr).astype(int), name=config.TARGET)
