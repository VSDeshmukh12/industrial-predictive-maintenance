"""Transparent engineering calculations from the six raw inputs (units shown)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FORMULAS = {
    "delta_t_k": "ΔT [K] = process temperature [K] − air temperature [K]",
    "mechanical_power_w": "P [W] = 2π × speed [rpm] × torque [Nm] / 60",
    "wear_torque_min_nm": "wear × torque [min·Nm] = tool wear [min] × torque [Nm]",
}
UNITS = {"delta_t_k": "K", "mechanical_power_w": "W", "wear_torque_min_nm": "min·Nm"}


def delta_t_k(air_temp_k, process_temp_k):
    return np.asarray(process_temp_k, dtype=float) - np.asarray(air_temp_k, dtype=float)


def mechanical_power_w(rot_speed_rpm, torque_nm):
    return 2.0 * math.pi * np.asarray(rot_speed_rpm, dtype=float) * np.asarray(torque_nm, dtype=float) / 60.0


def wear_torque_indicator(tool_wear_min, torque_nm):
    return np.asarray(tool_wear_min, dtype=float) * np.asarray(torque_nm, dtype=float)


def engineering_table(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised: three engineered quantities for every row (index preserved)."""
    return pd.DataFrame({
        "delta_t_k": delta_t_k(df["air_temp_k"], df["process_temp_k"]),
        "mechanical_power_w": mechanical_power_w(df["rot_speed_rpm"], df["torque_nm"]),
        "wear_torque_min_nm": wear_torque_indicator(df["tool_wear_min"], df["torque_nm"]),
    }, index=df.index)


def engineering_record(row) -> dict:
    """One row -> {quantity: {value, unit, formula}} for reports."""
    t = engineering_table(pd.DataFrame([dict(row)])).iloc[0]
    return {k: {"value": float(t[k]), "unit": UNITS[k], "formula": FORMULAS[k]} for k in FORMULAS}
