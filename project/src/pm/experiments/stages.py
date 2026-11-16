"""Shared helpers for the Phase 3 stage scripts (train partition ONLY)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

from pm import config
from pm.data.loader import load_dataset, sha256_of
from pm.data.splitting import get_or_create_split, get_train_data
from pm.experiments.benchmark import PHASE3_DIR
from pm.models.zoo import MODELS, STRATEGIES
from pm.features.engineering import FEATURE_SETS

SCREEN_REPEATS = (0, 1)          # screening + tuning folds
CONFIRM_REPEATS = (2, 3, 4)      # confirmation folds (never used for choosing)
ELIGIBLE_FEATURE_SETS = ("raw", "paper", "physics")   # rule_aware = optimistic reference only
DECISIONS_PATH = PHASE3_DIR / "decisions.json"
FEATURE_COMPLEXITY = {"raw": 0, "paper": 1, "physics": 2, "rule_aware": 3}


def load_train() -> pd.DataFrame:
    df = load_dataset()
    split = get_or_create_split(df, sha256_of(config.RAW_DATA_PATH))
    return get_train_data(df, split)      # hold-out partition is never materialised in Phase 3


def log(msg: str, fh=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def load_decisions() -> dict:
    return json.loads(DECISIONS_PATH.read_text()) if DECISIONS_PATH.exists() else {}


def save_decisions(d: dict):
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    DECISIONS_PATH.write_text(json.dumps(d, indent=2, default=str))


def stage1_specs():
    for fs in FEATURE_SETS:
        for m in MODELS:
            if m == "balanced_rf":
                strategies = ("none",)
            elif m == "svm_rbf":
                strategies = ("none", "class_weight")      # SMOTE variants skipped: 3x inner CV cost
            else:
                strategies = STRATEGIES
            for s in strategies:
                yield {"model": m, "strategy": s, "feature_set": fs}
