"""Read-only, integrity-checked access to the original dataset."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from pm import config


class DatasetIntegrityError(RuntimeError):
    """The dataset on disk is not the original, unmodified file."""


def sha256_of(path: Path | str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def load_raw(path: Path | str = config.RAW_DATA_PATH, verify_checksum: bool = True) -> pd.DataFrame:
    """Load the CSV exactly as shipped (original headers). Never writes to disk."""
    path = Path(path)
    if verify_checksum:
        actual = sha256_of(path)
        if actual != config.RAW_DATA_SHA256:
            raise DatasetIntegrityError(
                f"{path} has SHA-256 {actual}, expected {config.RAW_DATA_SHA256}. "
                "The original dataset must not be modified."
            )
    return pd.read_csv(path)


def canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    """Rename original headers to canonical snake_case names (returns a copy).

    Raises KeyError listing any missing original columns.
    """
    missing = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return df.rename(columns=config.COLUMN_MAP)[list(config.COLUMN_MAP.values())].copy()


def load_dataset(path: Path | str = config.RAW_DATA_PATH, verify_checksum: bool = True) -> pd.DataFrame:
    """Integrity-checked load + canonical column names."""
    return canonicalize(load_raw(path, verify_checksum))
