"""Read-only data and model access for the dashboard. No Streamlit imports: everything here is unit-testable.

* The dataset is loaded through the SHA-256-checked loader (never modified).
* Partition labels come from the persisted split file. The consumed hold-out is NOT materialised through
  the protected hold-out loader; rows are only tagged with their partition so the UI can say whether a prediction is in-sample.
* The bundle is loaded with `load_bundle` (model SHA-256 + frozen-spec hash verified). The ExplanationEngine
  refuses any bundle whose threshold differs from 0.365.
* Scoring is inference only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from pm import config
from pm.data.loader import load_dataset, sha256_of
from pm.explain import advisory, engineering, rules
from pm.explain.engine import ExplanationEngine
from pm.final import spec
from pm.final.bundle import Bundle, load_bundle

P3 = config.ARTIFACTS_DIR / "phase3"
P4 = config.ARTIFACTS_DIR / "phase4"
P5 = config.ARTIFACTS_DIR / "phase5"
BUNDLE_DIR = P4 / "final_model"
PARTITION_TRAIN = "training"
PARTITION_HOLDOUT = "hold-out"
RULE_COLS = ("hdf", "pwf", "osf")
RAW = list(config.MODEL_FEATURES)


# ------------------------------------------------------------------------------------------------ records
def load_records() -> pd.DataFrame:
    """All 10,000 actual records (canonical names) with a `partition` column. Labels stay in the frame for DISPLAY only."""
    df = load_dataset()
    split = json.loads(config.SPLIT_PATH.read_text())
    if split["dataset_sha256"] != sha256_of(config.RAW_DATA_PATH):
        raise RuntimeError("split file does not belong to this dataset")
    train = set(split["train_udi"])
    df = df.copy()
    df["partition"] = np.where(df["udi"].isin(train), PARTITION_TRAIN, PARTITION_HOLDOUT)
    return df


def load_system() -> tuple[Bundle, ExplanationEngine]:
    bundle = load_bundle(BUNDLE_DIR)
    return bundle, ExplanationEngine(bundle)


# ------------------------------------------------------------------------------------------------ scoring
def score_records(bundle: Bundle, df: pd.DataFrame) -> pd.DataFrame:
    """Frozen-model scoring + documented rules + TWF advisory for many rows. Inputs are the six raw columns only.

    Added columns: failure_probability, failure_alert (>= 0.365), delta_t_k, mechanical_power_w, wear_torque_min_nm,
    hdf_status/pwf_status/osf_status, rule_summary, tool_wear_advisory.
    """
    missing = [c for c in RAW if c not in df.columns]
    if missing:
        raise ValueError(f"cannot score: missing input column(s) {missing}")
    if len(df) == 0:
        raise ValueError("cannot score: no rows")
    out = df.copy().reset_index(drop=True)
    X = out[RAW]
    bad = [c for c in config.NUMERIC_FEATURES if not pd.api.types.is_numeric_dtype(X[c])]
    if bad:
        raise ValueError(f"cannot score: non-numeric values in {bad}")
    a = bundle.assess(X)                                  # validates inputs; probability, alert, advisory
    out["failure_probability"] = a["failure_probability"].to_numpy()
    out["failure_alert"] = a["failure_alert"].to_numpy()
    out["tool_wear_advisory"] = a["tool_wear_advisory"].to_numpy()
    e = engineering.engineering_table(X)
    for c in e.columns:
        out[c] = e[c].to_numpy()
    r = rules.rules_table(X)
    for c in RULE_COLS:
        out[f"{c}_status"] = r[c].to_numpy()
    met = (r == "condition_met").any(axis=1).to_numpy()
    near = (r == "approaching").any(axis=1).to_numpy()
    out["rule_summary"] = np.where(met, "condition_met", np.where(near, "approaching", "none"))
    return out


def failure_mode_label(row) -> str:
    modes = [m.upper() for m in config.MODE_COLUMNS if int(row[m]) == 1]
    return "none" if not int(row[config.TARGET]) else (", ".join(modes) if modes else "unlabelled")


# ------------------------------------------------------------------------------------------------ reference ranges
def training_ranges(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Observed min/max of the six inputs in the TRAINING partition (extrapolation warnings for manual input)."""
    t = df[df["partition"] == PARTITION_TRAIN]
    return {c: (float(t[c].min()), float(t[c].max())) for c in config.NUMERIC_FEATURES}


def input_defaults(df: pd.DataFrame) -> dict:
    t = df[df["partition"] == PARTITION_TRAIN]
    d = {c: float(t[c].median()) for c in config.NUMERIC_FEATURES}
    d["type"] = "M"
    return d


def extrapolation_warnings(record: dict, ranges: dict[str, tuple[float, float]]) -> list[str]:
    w = []
    for c, (lo, hi) in ranges.items():
        v = float(record[c])
        if v < lo or v > hi:
            w.append(f"{c} = {v:g} lies outside the range seen in the training data ({lo:g} to {hi:g}); the model has no experience there.")
    dt = float(record["process_temp_k"]) - float(record["air_temp_k"])
    if dt <= 0:
        w.append("Process temperature is not above air temperature (ΔT ≤ 0 K); such combinations do not occur in the dataset.")
    return w


# ------------------------------------------------------------------------------------------------ saved Phase 4 / 5 artifacts
def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def global_shap_tables() -> dict[str, pd.DataFrame]:
    """Training-partition global attribution saved in Phase 5 (nine features, four driver groups, per failure mode)."""
    return {"features": _csv(P5 / "global_shap_features.csv"), "groups": _csv(P5 / "global_shap_groups.csv"),
            "by_mode": _csv(P5 / "global_shap_by_failure_mode.csv")}


def global_shap_training_values() -> dict:
    """Per-row SHAP of the 8,000 training rows (Phase 5 .npz) for beeswarm-style plots."""
    z = np.load(P5 / "global_shap_train.npz")
    return {k: z[k] for k in z.files}


def global_shap_summary() -> dict:
    return json.loads((P5 / "global_shap_summary.json").read_text())


def advisory_tables() -> dict[str, pd.DataFrame]:
    return {"bands": _csv(P5 / "advisory_bands_training.csv"), "by_wear_bin": _csv(P5 / "twf_rate_by_wear_bin_training.csv"),
            "rule_status": _csv(P5 / "rule_status_failure_rates.csv")}


def phase4_report() -> dict:
    """Phase 4 final hold-out result (REPORTING ONLY, already published; the hold-out is consumed)."""
    m = json.loads((P4 / "holdout_metrics.json").read_text())
    return {"metrics": m, "failure_modes": _csv(P4 / "holdout_failure_modes.csv"), "by_type": _csv(P4 / "holdout_by_type.csv")}


def system_info(bundle: Bundle, records: pd.DataFrame) -> dict:
    mf = bundle.manifest
    return {
        "model": f"{spec.MODEL} (frozen; {spec.STRATEGY} resampling; {spec.FEATURE_SET} features; uncalibrated)",
        "params": dict(spec.PARAMS), "threshold": bundle.threshold, "threshold_status": "provisional (F2 objective)",
        "model_sha256": mf["model_sha256"], "frozen_spec_sha256": mf["frozen_spec_sha256"],
        "dataset_sha256": sha256_of(config.RAW_DATA_PATH),
        "dataset_sha256_ok": sha256_of(config.RAW_DATA_PATH) == config.RAW_DATA_SHA256,
        "rows": int(len(records)), "train_rows": int((records["partition"] == PARTITION_TRAIN).sum()),
        "holdout_rows": int((records["partition"] == PARTITION_HOLDOUT).sum()),
        "advisory_bands_min": (advisory.ADVISORY_MIN, advisory.ELEVATED_MIN), "rules_margin": rules.MARGIN,
    }
