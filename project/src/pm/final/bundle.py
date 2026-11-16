"""Final model bundle: train on the full training partition, save, load, predict.

The bundle directory holds
  model_pipeline.joblib   fitted pipeline: leakage_guard -> engineer(physics) -> preprocess -> RandomForest
  manifest.json           frozen spec, threshold, feature definitions, hashes, environment metadata
  feature_definitions.json, environment.json, requirements.lock.txt   (human-readable copies)

`load_bundle` verifies the model file SHA-256 against the manifest before unpickling and refuses
to run on a mismatch. Inference accepts ONLY the six approved raw inputs (the pipeline's leakage
guard rejects identifiers, the label and failure-mode flags).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from pm import config
from pm.data.loader import sha256_of
from pm.features.engineering import DERIVED
from pm.features.preprocessing import check_features, select_model_inputs
from pm.final import spec

SYNTHETIC_NOTICE = ("AI4I 2020 is a SYNTHETIC dataset (UCI, Matzka 2020). Scores measure recovery of the data "
                    "generator's rules, not predictive capability on real machines.")
TWF_NOTICE = ("Tool-wear (TWF) failures are randomised in the generator; the classifier cannot predict them "
              "reliably. The tool-wear advisory only says the tool is inside the documented 200-240 min "
              "replacement window. It is a maintenance heuristic, not a failure prediction.")


class BundleIntegrityError(RuntimeError):
    pass


def build_pipeline():
    """The frozen Phase 3 experiment pipeline, unchanged (same builder the CV experiments used)."""
    from pm.models.zoo import build_experiment_pipeline      # local import: heavy deps only needed to train
    return build_experiment_pipeline(spec.MODEL, spec.STRATEGY, spec.FEATURE_SET, spec.PARAMS)


def train_final(train_df: pd.DataFrame):
    """Fit the frozen pipeline on ALL rows of the training partition."""
    X = select_model_inputs(train_df)
    y = train_df[config.TARGET].astype(int).reset_index(drop=True)
    return build_pipeline().fit(X.reset_index(drop=True), y)


# ----------------------------------------------------------------------------------- metadata
def feature_definitions(pipeline) -> dict:
    rf = pipeline.named_steps["model"]
    names = [str(n) for n in pipeline.named_steps["preprocess"].get_feature_names_out()]
    imp = dict(zip(names, (float(v) for v in rf.feature_importances_)))
    return {
        "raw_inputs": {
            "air_temp_k": "Air temperature [K]", "process_temp_k": "Process temperature [K]",
            "rot_speed_rpm": "Rotational speed [rpm]", "torque_nm": "Torque [Nm]",
            "tool_wear_min": "Tool wear [min]", "type": "Product quality variant, one of L/M/H",
        },
        "type_categories": list(config.TYPE_CATEGORIES),
        "derived_physics_features": {
            "delta_t": "process_temp_k - air_temp_k  [K]",
            "power_w": "torque_nm * rot_speed_rpm * 2*pi/60  [W]",
            "overstrain": "tool_wear_min * torque_nm  [min*Nm]",
        },
        "never_model_inputs": sorted(config.FORBIDDEN_FEATURES),
        "model_matrix_columns_in_order": names,
        "scaling": "StandardScaler on all numeric columns (raw + derived), fitted on the training partition",
        "rf_impurity_importance_training_fit": imp,
        "tool_wear_advisory_inputs": ["tool_wear_min"],
    }


def environment_metadata() -> dict:
    pins = {}
    req = config.PROJECT_ROOT / "requirements.txt"
    for line in req.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            n, v = line.strip().split("==")
            try:
                pins[n] = metadata.version(n)
            except metadata.PackageNotFoundError:
                pins[n] = None
    mism = {n: {"installed": pins[n], "pinned": ln.split("==")[1]}
            for ln in req.read_text().splitlines() if ln.strip() and not ln.startswith("#")
            for n in [ln.split("==")[0]] if pins[n] != ln.strip().split("==")[1]}
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "installed_versions": pins, "pin_mismatches": mism,
        "requirements_txt_sha256": sha256_of(req),
    }


def training_data_fingerprint(train_df: pd.DataFrame) -> dict:
    X = select_model_inputs(train_df)
    h = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([train_df[["udi"]], X, train_df[[config.TARGET]]], axis=1),
                                                  index=False).to_numpy().tobytes()).hexdigest()
    return {"rows": int(len(train_df)), "positives": int(train_df[config.TARGET].sum()),
            "udi_min": int(train_df["udi"].min()), "udi_max": int(train_df["udi"].max()), "content_hash": h}


def save_bundle(pipeline, out_dir: Path, *, train_df: pd.DataFrame, split_path: Path = config.SPLIT_PATH) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / spec.MODEL_FILE
    joblib.dump(pipeline, model_path, compress=3)
    fdefs = feature_definitions(pipeline)
    env = environment_metadata()
    manifest = {
        "bundle_schema_version": 1,
        "phase": 4,
        "frozen_spec": spec.frozen_dict(), "frozen_spec_sha256": spec.spec_sha256(),
        "threshold": {"value": spec.THRESHOLD, "objective": spec.THRESHOLD_OBJECTIVE, "rule": spec.THRESHOLD_RULE,
                      "status": "provisional; set from training-only cross-validation, never from hold-out results"},
        "model_file": spec.MODEL_FILE, "model_sha256": sha256_of(model_path),
        "dataset": {"file": "data/raw/ai4i2020.csv", "sha256": sha256_of(config.RAW_DATA_PATH),
                    "pinned_sha256": config.RAW_DATA_SHA256},
        "split_file_sha256": sha256_of(split_path),
        "training_data": training_data_fingerprint(train_df),
        "feature_definitions": fdefs,
        "environment": env,
        "notices": {"synthetic_data": SYNTHETIC_NOTICE, "twf": TWF_NOTICE},
    }
    (out_dir / spec.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (out_dir / "feature_definitions.json").write_text(json.dumps(fdefs, indent=2))
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (out_dir / "requirements.lock.txt").write_text(
        "\n".join(f"{n}=={v}" for n, v in sorted(env["installed_versions"].items())) + "\n")
    return manifest


# ----------------------------------------------------------------------------------- inference
@dataclass
class Bundle:
    pipeline: object
    manifest: dict

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"]["value"])

    def _inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in config.MODEL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        X = df.loc[:, list(config.MODEL_FEATURES)].reset_index(drop=True)
        check_features(X.columns)
        if not X["type"].isin(config.TYPE_CATEGORIES).all():
            raise ValueError(f"type must be one of {config.TYPE_CATEGORIES}")
        for col, (lo, hi) in config.PLAUSIBLE_RANGES.items():
            if not X[col].between(lo, hi).all() or X[col].isna().any():
                raise ValueError(f"{col} outside plausible range [{lo}, {hi}] or missing")
        return X

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Uncalibrated RF probability of failure."""
        return self.pipeline.predict_proba(self._inputs(df))[:, 1]

    def assess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Two SEPARATE outputs: classifier alert, and the tool-wear advisory."""
        X = self._inputs(df)
        p = self.pipeline.predict_proba(X)[:, 1]
        wear = X["tool_wear_min"].to_numpy()
        adv = np.where(wear >= spec.TWF_ELEVATED_WEAR_MIN, "elevated",
                       np.where(wear >= spec.TWF_ADVISORY_WEAR_MIN, "advisory", "none"))
        return pd.DataFrame({"failure_probability": p, "failure_alert": p >= self.threshold,
                             "tool_wear_advisory": adv})


def load_bundle(bundle_dir: Path) -> Bundle:
    bundle_dir = Path(bundle_dir)
    manifest = json.loads((bundle_dir / spec.MANIFEST_FILE).read_text())
    model_path = bundle_dir / manifest["model_file"]
    actual = sha256_of(model_path)
    if actual != manifest["model_sha256"]:
        raise BundleIntegrityError(f"{model_path} SHA-256 {actual} != manifest {manifest['model_sha256']}")
    if manifest["frozen_spec_sha256"] != spec.spec_sha256():
        raise BundleIntegrityError("Frozen spec in the code differs from the one the bundle was built with")
    pipe = joblib.load(model_path)
    return Bundle(pipe, manifest)
