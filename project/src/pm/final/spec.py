"""The FROZEN Phase 4 specification.

Everything here was fixed by Phase 3 (training-partition cross-validation only) and approved
before the hold-out was touched. Nothing in this module may be changed using hold-out results.
`assert_matches_phase3` cross-checks the constants against `artifacts/phase3/decisions.json`
so a silent edit of either side is detected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pm import config

MODEL = "random_forest"
STRATEGY = "none"                       # no resampling
FEATURE_SET = "physics"
PARAMS = {"max_depth": 6, "max_features": 0.8, "min_samples_leaf": 6, "n_estimators": 390}
CALIBRATION = "none|uncalibrated"       # raw RF probabilities
THRESHOLD = 0.365                       # provisional, F2 objective; predict failure if p >= THRESHOLD
THRESHOLD_OBJECTIVE = "F2 (provisional)"
THRESHOLD_RULE = "alert if failure_probability >= threshold"

# Tool-wear advisory: SEPARATE from the classifier. Fixed a priori from the UCI documentation
# (tool-wear failures occur at random between 200 and 240 min of tool wear; 220 = window midpoint).
# It is NOT fitted or tuned on any data, and is not a failure prediction.
TWF_ADVISORY_WEAR_MIN = 200.0           # "advisory": tool has entered the documented 200-240 min window
TWF_ELEVATED_WEAR_MIN = 220.0           # "elevated": past the window midpoint

MODEL_FILE = "model_pipeline.joblib"
MANIFEST_FILE = "manifest.json"


def frozen_dict() -> dict:
    return {
        "model": MODEL, "strategy": STRATEGY, "feature_set": FEATURE_SET, "params": dict(PARAMS),
        "calibration": CALIBRATION, "threshold": THRESHOLD, "threshold_objective": THRESHOLD_OBJECTIVE,
        "random_seed": config.RANDOM_SEED,
        "twf_advisory": {"advisory_wear_min": TWF_ADVISORY_WEAR_MIN, "elevated_wear_min": TWF_ELEVATED_WEAR_MIN,
                         "role": "separate tool-wear advisory; not a classifier output"},
    }


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_dict(), sort_keys=True).encode()).hexdigest()


def assert_matches_phase3(decisions_path: Path | None = None) -> None:
    """Raise AssertionError if the constants differ from the approved Phase 3 decisions."""
    p = decisions_path or (config.ARTIFACTS_DIR / "phase3" / "decisions.json")
    d = json.loads(Path(p).read_text())
    f, t = d["final"], d["threshold"]
    assert f["model"] == MODEL and f["strategy"] == STRATEGY and f["feature_set"] == FEATURE_SET, f
    assert f["params"] == PARAMS, (f["params"], PARAMS)
    assert d["calibration"]["chosen_variant"] == CALIBRATION, d["calibration"]
    assert abs(t["tau"] - THRESHOLD) < 1e-12 and t["objective"] == THRESHOLD_OBJECTIVE, t
