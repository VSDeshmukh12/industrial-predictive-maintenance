"""Central, immutable project configuration.

Everything that defines the data contract (column names, which columns are
allowed as model inputs, seeds, split parameters) lives here so that the
rest of the code never hard-codes it.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"
# SHA-256 of the original, unmodified CSV. The loader refuses any other file.
RAW_DATA_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PHASE2_DIR = ARTIFACTS_DIR / "phase2"
SPLIT_PATH = ARTIFACTS_DIR / "splits" / "holdout_seed42.json"

# ---------------------------------------------------------------- randomness
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_N_SPLITS = 5
CV_N_REPEATS = 5

# ------------------------------------------------------------------- schema
# Original CSV header -> canonical snake_case name. Canonical names contain no
# brackets/spaces (XGBoost/LightGBM reject some of those characters).
COLUMN_MAP: dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}
RAW_COLUMNS: tuple[str, ...] = tuple(COLUMN_MAP)

ID_COLUMNS: tuple[str, ...] = ("udi", "product_id")
TARGET: str = "machine_failure"
MODE_COLUMNS: tuple[str, ...] = ("twf", "hdf", "pwf", "osf", "rnf")
NUMERIC_FEATURES: tuple[str, ...] = (
    "air_temp_k",
    "process_temp_k",
    "rot_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)
CATEGORICAL_FEATURES: tuple[str, ...] = ("type",)
TYPE_CATEGORIES: tuple[str, ...] = ("L", "M", "H")

# The ONLY columns a main-model pipeline may receive.
MODEL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
# Columns that must never reach the main model (identifiers, label, and
# failure-mode labels, which fully determine the label).
FORBIDDEN_FEATURES: frozenset[str] = frozenset(ID_COLUMNS + MODE_COLUMNS + (TARGET,))

# Generous physical-plausibility bounds (violations are errors, not outliers).
PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "air_temp_k": (250.0, 350.0),
    "process_temp_k": (250.0, 400.0),
    "rot_speed_rpm": (1.0, 10_000.0),
    "torque_nm": (0.0, 500.0),
    "tool_wear_min": (0.0, 1_000.0),
}

# Counts stated in the dataset documentation (UCI, DOI 10.24432/C5HS5C),
# used only as an informational cross-check against the CSV.
DOCUMENTED_MODE_COUNTS: dict[str, int] = {"twf": 51, "hdf": 115, "pwf": 95, "osf": 98}
