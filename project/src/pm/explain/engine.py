"""SHAP explanations for the FROZEN Phase 4 Random Forest, plus the assembled per-record explanation.

Method: shap.TreeExplainer, feature_perturbation="tree_path_dependent" (exact TreeSHAP, deterministic, no background
data), output = the forest's probability of failure. For every row:  base_value + sum(SHAP) = probability  (checked).
The 11 model-matrix columns are shown as 9 features (the three one-hot `type` columns are summed: SHAP is additive)
and, additionally, as 4 physical-driver groups (also additive). Attribution is NOT causation (see notices).

The engine never fits, re-tunes, re-calibrates or changes the threshold; the bundle is used read-only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from pm import config
from pm.explain import advisory, engineering, notices, rules
from pm.final import spec

FEATURES = ("air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min",
            "delta_t", "power_w", "overstrain", "type")
UNITS = {"air_temp_k": "K", "process_temp_k": "K", "rot_speed_rpm": "rpm", "torque_nm": "Nm", "tool_wear_min": "min",
         "delta_t": "K", "power_w": "W", "overstrain": "min·Nm", "type": ""}
LABELS = {"air_temp_k": "air temperature", "process_temp_k": "process temperature", "rot_speed_rpm": "rotational speed",
          "torque_nm": "torque", "tool_wear_min": "tool wear", "delta_t": "ΔT (process − air)", "power_w": "mechanical power",
          "overstrain": "wear × torque", "type": "product type"}
GROUPS = {
    "thermal": ("air_temp_k", "process_temp_k", "delta_t"),
    "speed_torque_power": ("rot_speed_rpm", "torque_nm", "power_w"),
    "wear_overstrain": ("tool_wear_min", "overstrain"),
    "product_type": ("type",),
}
ADDITIVITY_TOL = 1e-9


@dataclass
class ShapResult:
    base_value: float
    probability: np.ndarray        # (n,)
    values: np.ndarray             # (n, 9) in FEATURES order
    display: pd.DataFrame          # (n, 9) feature values in original units (type as letter)
    groups: np.ndarray             # (n, 4) in GROUPS order
    additivity_error: np.ndarray   # (n,)


class ExplanationEngine:
    def __init__(self, bundle):
        if abs(bundle.threshold - spec.THRESHOLD) > 1e-12:
            raise ValueError("bundle threshold differs from the frozen provisional threshold 0.365")
        self.bundle = bundle
        p = bundle.pipeline
        self._guard, self._eng, self._pre = p.named_steps["leakage_guard"], p.named_steps["engineer"], p.named_steps["preprocess"]
        self.rf = p.named_steps["model"]
        self._cols = [str(c) for c in self._pre.get_feature_names_out()]
        self.explainer = shap.TreeExplainer(self.rf, feature_perturbation="tree_path_dependent")
        ev = np.asarray(self.explainer.expected_value, dtype=float).ravel()
        self.base_value = float(ev[-1])
        self.threshold = float(bundle.threshold)

    # ------------------------------------------------------------------ SHAP
    def _design(self, X: pd.DataFrame):
        e = self._eng.transform(self._guard.transform(X))
        M = np.asarray(self._pre.transform(e), dtype=float)
        return e, M

    def shap(self, df: pd.DataFrame) -> ShapResult:
        X = self.bundle._inputs(df)                      # validates: six inputs, type in L/M/H, plausible ranges, no NaN
        e, M = self._design(X)
        sv = np.asarray(self.explainer.shap_values(M, check_additivity=False))
        sv = sv[..., 1] if sv.ndim == 3 else sv
        prob = self.rf.predict_proba(M)[:, 1]
        cols = {c: i for i, c in enumerate(self._cols)}
        vals = np.column_stack([sv[:, cols[f]] if f != "type" else sv[:, [cols[c] for c in cols if c.startswith("type_")]].sum(axis=1)
                                for f in FEATURES])
        grp = np.column_stack([vals[:, [FEATURES.index(f) for f in fs]].sum(axis=1) for fs in GROUPS.values()])
        disp = pd.DataFrame({f: (X["type"].to_numpy() if f == "type" else e[f].to_numpy()) for f in FEATURES})
        err = self.base_value + vals.sum(axis=1) - prob
        if np.abs(err).max() > ADDITIVITY_TOL:
            raise AssertionError(f"SHAP additivity violated (max error {np.abs(err).max():.3g})")
        return ShapResult(self.base_value, prob, vals, disp, grp, err)

    def global_attribution(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Mean |SHAP| per feature over the rows given (the Phase 5 scripts pass the TRAINING partition only)."""
        r = self.shap(train_df)
        a = np.abs(r.values).mean(axis=0)
        out = pd.DataFrame({"feature": FEATURES, "label": [LABELS[f] for f in FEATURES], "mean_abs_shap": a,
                            "mean_shap": r.values.mean(axis=0), "share_of_total_abs": a / a.sum()})
        return out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def group_attribution(self, train_df: pd.DataFrame) -> pd.DataFrame:
        r = self.shap(train_df)
        a = np.abs(r.groups).mean(axis=0)
        return pd.DataFrame({"group": list(GROUPS), "members": [", ".join(v) for v in GROUPS.values()],
                             "mean_abs_shap": a, "share_of_total_abs": a / a.sum()}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------ assembled explanation
    def explain(self, df: pd.DataFrame, top_k: int = 5) -> list[dict]:
        raw = self.bundle._inputs(df)                     # ValueError for missing columns / bad type / out-of-range / NaN
        r = self.shap(raw)
        out = []
        for i in range(len(raw)):
            row = raw.iloc[i]
            p = float(r.probability[i])
            feats = [{"feature": f, "label": LABELS[f], "value": (str(r.display.at[i, f]) if f == "type" else float(r.display.at[i, f])),
                      "unit": UNITS[f], "shap": float(r.values[i, j])} for j, f in enumerate(FEATURES)]
            feats.sort(key=lambda d: -abs(d["shap"]))
            cond = rules.evaluate_row(row)
            alert = p >= self.threshold
            met = cond["summary"] == "documented_condition_met"
            consistency = ("rule_and_alert" if met and alert else "rule_only" if met else "alert_only" if alert else "neither")
            out.append({
                "inputs": {k: (row[k] if k == "type" else float(row[k])) for k in config.MODEL_FEATURES},
                "engineering": engineering.engineering_record(row),
                "model": {"failure_probability": p, "threshold": self.threshold, "threshold_status": "provisional",
                          "alert": bool(alert), "output_kind": "classifier output (Random Forest probability, uncalibrated)"},
                "attribution": {"method": "TreeSHAP (path-dependent), probability space", "base_value": r.base_value,
                                "additivity_error": float(r.additivity_error[i]), "top": feats[:top_k], "features": feats,
                                "groups": {g: float(r.groups[i, k]) for k, g in enumerate(GROUPS)},
                                "kind": "model attribution, not causation"},
                "conditions": cond,
                "model_vs_rules": consistency,
                "tool_wear_advisory": advisory.tool_wear_advisory(float(row["tool_wear_min"])).to_dict(),
                "notices": dict(notices.ALL),
            })
        return out
