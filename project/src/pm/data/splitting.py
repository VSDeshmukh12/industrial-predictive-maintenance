"""Reproducible, persisted, stratified hold-out split.

* Stratified on (type x machine_failure) so both class balance and the Type mix
  are preserved (the H stratum has only 21 positives overall).
* The split is persisted as a list of UDI values with the dataset hash. The saved
  file, not the RNG, is the source of truth, so results do not depend on library
  versions. UDI is used only as a bookkeeping key, never as a feature.
* The test partition is protected: it can only be materialised by calling
  `get_test_data(..., confirm_final_evaluation=True)`. All development work
  (CV, tuning, threshold selection) uses `get_train_data` only.
"""
from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from pm import config
from pm.data.loader import sha256_of

SCHEMA_VERSION = 1


class SplitIntegrityError(RuntimeError):
    pass


@dataclass(frozen=True)
class Split:
    train_udi: tuple[int, ...]
    test_udi: tuple[int, ...]
    meta: dict


def make_strata(df: pd.DataFrame) -> pd.Series:
    """Composite stratification label, e.g. 'L_0', 'H_1'."""
    return (df["type"].astype(str) + "_" + df[config.TARGET].astype(int).astype(str)).rename("strata")


def _summaries(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "positives": int(df[config.TARGET].sum()),
        "prevalence": round(float(df[config.TARGET].mean()), 6),
        "by_type": {t: {"rows": int((df["type"] == t).sum()),
                        "positives": int(df.loc[df["type"] == t, config.TARGET].sum())}
                    for t in config.TYPE_CATEGORIES},
    }


def create_split(df: pd.DataFrame, dataset_sha256: str, seed: int = config.RANDOM_SEED,
                 test_size: float = config.TEST_SIZE) -> Split:
    if df["udi"].duplicated().any():
        raise SplitIntegrityError("udi must be unique to be used as a split key")
    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=test_size, stratify=make_strata(df),
        random_state=seed, shuffle=True,
    )
    train_df, test_df = df.iloc[np.sort(train_idx)], df.iloc[np.sort(test_idx)]
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "seed": seed,
        "test_size": test_size,
        "stratified_on": "type x machine_failure",
        "train": _summaries(train_df),
        "test": _summaries(test_df),
        "created_with": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
    }
    return Split(tuple(int(u) for u in sorted(train_df["udi"])),
                 tuple(int(u) for u in sorted(test_df["udi"])), meta)


def save_split(split: Split, path: Path | str = config.SPLIT_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(split.meta, train_udi=list(split.train_udi), test_udi=list(split.test_udi))
    path.write_text(json.dumps(payload, indent=1))
    return path


def verify_split(split: Split, df: pd.DataFrame, dataset_sha256: str) -> None:
    """Raise SplitIntegrityError unless the split is valid for this exact dataset."""
    if split.meta.get("dataset_sha256") != dataset_sha256:
        raise SplitIntegrityError("Split was created for a different dataset (hash mismatch)")
    tr, te = set(split.train_udi), set(split.test_udi)
    if len(tr) != len(split.train_udi) or len(te) != len(split.test_udi):
        raise SplitIntegrityError("Duplicate UDIs inside a partition")
    if tr & te:
        raise SplitIntegrityError(f"{len(tr & te)} rows appear in both train and test")
    all_udi = set(int(u) for u in df["udi"])
    if tr | te != all_udi:
        raise SplitIntegrityError("Train + test do not cover exactly the dataset's rows")
    for name, udis in (("train", tr), ("test", te)):
        part = df[df["udi"].isin(udis)]
        if _summaries(part) != split.meta[name]:
            raise SplitIntegrityError(f"{name} summary does not match the stored metadata")


def load_split(df: pd.DataFrame, dataset_sha256: str, path: Path | str = config.SPLIT_PATH) -> Split:
    raw = json.loads(Path(path).read_text())
    split = Split(tuple(raw.pop("train_udi")), tuple(raw.pop("test_udi")), raw)
    verify_split(split, df, dataset_sha256)
    return split


def get_or_create_split(df: pd.DataFrame, dataset_sha256: str, path: Path | str = config.SPLIT_PATH) -> Split:
    """Load the persisted split if present (verifying it), else create and persist it."""
    path = Path(path)
    if path.exists():
        return load_split(df, dataset_sha256, path)
    split = create_split(df, dataset_sha256)
    verify_split(split, df, dataset_sha256)
    save_split(split, path)
    return split


def get_train_data(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    """Development partition (original row order, reset index)."""
    return df[df["udi"].isin(set(split.train_udi))].reset_index(drop=True)


def get_test_data(df: pd.DataFrame, split: Split, *, confirm_final_evaluation: bool = False) -> pd.DataFrame:
    """Hold-out partition. Use ONCE, for the final model, after all choices are frozen."""
    if not confirm_final_evaluation:
        raise PermissionError(
            "The hold-out test set is reserved for the single final evaluation. "
            "Pass confirm_final_evaluation=True only when all modelling choices are frozen."
        )
    return df[df["udi"].isin(set(split.test_udi))].reset_index(drop=True)


def make_cv_splits(df_train: pd.DataFrame, n_splits: int = config.CV_N_SPLITS,
                   n_repeats: int = config.CV_N_REPEATS, seed: int = config.RANDOM_SEED):
    """Repeated stratified K-fold on the composite strata. Yields (repeat, fold, train_idx, val_idx)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, va) in enumerate(rskf.split(np.zeros(len(df_train)), make_strata(df_train))):
        yield i // n_splits, i % n_splits, tr, va
