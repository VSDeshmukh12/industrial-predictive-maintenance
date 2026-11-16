"""Probability-calibration metrics and calibration wrappers."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

EPS = 1e-12


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    """ECE with equal-FREQUENCY bins (equal-width bins are nearly empty at 3% prevalence)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return float(sum(len(b) / len(y) * abs(y[b].mean() - p[b].mean()) for b in bins if len(b)))


def calibration_slope_intercept(y, p) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b*logit(p). Perfect calibration: slope 1, intercept 0."""
    z = _logit(np.asarray(p, float)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y).astype(int))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def calibration_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    slope, intercept = calibration_slope_intercept(y, p)
    return {"brier": float(np.mean((p - y) ** 2)), "log_loss": float(log_loss(y, p)),
            "ece": expected_calibration_error(y, p), "cal_slope": slope, "cal_intercept": intercept,
            "mean_pred": float(p.mean()), "prevalence": float(y.mean())}


def reliability_bins(y, p, n_bins: int = 10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.array_split(np.argsort(p), n_bins)
    return (np.array([p[b].mean() for b in bins]), np.array([y[b].mean() for b in bins]))


def prior_shift(p, pi_train: float, pi_true: float):
    """Bayes prior-shift correction of probabilities from a model trained at prior pi_train."""
    odds = np.asarray(p, float) / (1 - np.asarray(p, float) + EPS)
    odds = odds * (pi_true / (1 - pi_true)) / (pi_train / (1 - pi_train))
    return odds / (1 + odds)


class PriorShiftCorrected(ClassifierMixin, BaseEstimator):
    """Wrap a pipeline trained on an (effectively) balanced prior pi_train and correct its
    probabilities to the natural prevalence of the training fold (known from y at fit time)."""

    def __init__(self, estimator=None, pi_train: float = 0.5):
        self.estimator = estimator
        self.pi_train = pi_train

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.pi_true_ = float(np.mean(np.asarray(y)))
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        p1 = prior_shift(self.estimator_.predict_proba(X)[:, 1], self.pi_train, self.pi_true_)
        return np.column_stack([1 - p1, p1])


def calibrated(estimator, method: str, cv: int = 3):
    """Sigmoid (Platt) or isotonic calibration fitted on held-out sub-folds of the training data
    (natural prevalence, never resampled)."""
    return CalibratedClassifierCV(estimator, method=method, cv=cv, ensemble=True)
