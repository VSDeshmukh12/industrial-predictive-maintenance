"""Machine-condition interpretation from transparent, documented rules.

Source of the three rules: the UCI AI4I 2020 dataset documentation (Matzka 2020), which describes how the
SYNTHETIC generator produces failure labels:
  HDF  heat dissipation : ΔT < 8.6 K AND speed < 1380 rpm
  PWF  power            : power < 3500 W OR power > 9000 W
  OSF  overstrain       : wear × torque > 11000 (L) / 12000 (M) / 13000 (H) min·Nm
(TWF is covered by the separate advisory; RNF is random and has no rule.)

"approaching" is an INTERPRETIVE CONVENTION of this project, not part of the documentation: the rule is not met but
every sub-condition is met or within MARGIN (10 %) of its boundary. The margin is fixed here, not fitted.
Comparisons use float64 exactly as the generator's labels do (see the report: at ΔT = 8.6 K the label depends on
floating-point round-off, so that single boundary is numerically ambiguous).
These rules are NOT model inputs and are not real-world engineering limits.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from pm.explain import engineering, notices

MARGIN = 0.10
HDF_DT_MAX_K, HDF_RPM_MAX = 8.6, 1380.0
PWF_MIN_W, PWF_MAX_W = 3500.0, 9000.0
OSF_LIMIT = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}
SOURCE = "UCI AI4I 2020 documentation (Matzka 2020): rules of the synthetic data generator"
STATUSES = ("condition_met", "approaching", "not_met")


def rules_table(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised status of the three rules for every row: columns hdf/pwf/osf with STATUSES values."""
    e = engineering.engineering_table(df)
    dt, p, ov, rpm = e["delta_t_k"].to_numpy(), e["mechanical_power_w"].to_numpy(), e["wear_torque_min_nm"].to_numpy(), df["rot_speed_rpm"].to_numpy(float)
    lim = df["type"].map(OSF_LIMIT).to_numpy(float)
    if np.isnan(lim).any():
        raise ValueError("type must be one of L/M/H")
    hdf_met = (dt < HDF_DT_MAX_K) & (rpm < HDF_RPM_MAX)
    hdf_near = (dt < HDF_DT_MAX_K * (1 + MARGIN)) & (rpm < HDF_RPM_MAX * (1 + MARGIN))
    pwf_met = (p < PWF_MIN_W) | (p > PWF_MAX_W)
    pwf_near = (p < PWF_MIN_W * (1 + MARGIN)) | (p > PWF_MAX_W * (1 - MARGIN))
    osf_met = ov > lim
    osf_near = ov >= lim * (1 - MARGIN)

    def status(met, near):
        return np.where(met, "condition_met", np.where(near, "approaching", "not_met"))
    return pd.DataFrame({"hdf": status(hdf_met, hdf_near), "pwf": status(pwf_met, pwf_near),
                         "osf": status(osf_met, osf_near)}, index=df.index)


@dataclass(frozen=True)
class RuleResult:
    code: str
    name: str
    status: str
    rule: str
    evidence: str
    source: str = SOURCE

    def to_dict(self):
        return asdict(self)


_NAMES = {"hdf": "Heat dissipation (HDF)", "pwf": "Power band (PWF)", "osf": "Overstrain (OSF)"}


def evaluate_row(row) -> dict:
    """Interpret ONE record. Returns rule results, an overall summary and the mandatory rules notice."""
    r = dict(row)
    df = pd.DataFrame([r])
    st = rules_table(df).iloc[0]
    e = engineering.engineering_table(df).iloc[0]
    lim = OSF_LIMIT[r["type"]]
    rule_text = {"hdf": f"ΔT < {HDF_DT_MAX_K} K and speed < {HDF_RPM_MAX:.0f} rpm",
                 "pwf": f"power < {PWF_MIN_W:.0f} W or > {PWF_MAX_W:.0f} W",
                 "osf": f"wear × torque > {lim:.0f} min·Nm for type {r['type']}"}
    evidence = {"hdf": f"ΔT = {e['delta_t_k']:.2f} K, speed = {float(r['rot_speed_rpm']):.0f} rpm",
                "pwf": f"power = {e['mechanical_power_w']:.0f} W",
                "osf": f"wear × torque = {e['wear_torque_min_nm']:.0f} min·Nm ({100 * e['wear_torque_min_nm'] / lim:.0f}% of the type-{r['type']} limit)"}
    results = [RuleResult(c, _NAMES[c], str(st[c]), rule_text[c], evidence[c]) for c in ("hdf", "pwf", "osf")]
    met = [x.code for x in results if x.status == "condition_met"]
    near = [x.code for x in results if x.status == "approaching"]
    if met:
        summary = "documented_condition_met"
        text = "Documented synthetic-generator failure condition met: " + ", ".join(c.upper() for c in met) + "."
    elif near:
        summary = "approaching"
        text = "No documented condition met; approaching (within 10% of the boundary): " + ", ".join(c.upper() for c in near) + "."
    else:
        summary = "no_condition_met_or_approaching"
        text = "No documented failure condition is met or approached. This does not rule out tool-wear or random failures."
    return {"results": [x.to_dict() for x in results], "summary": summary, "summary_text": text,
            "margin_convention": f"'approaching' = within {int(MARGIN * 100)}% of a boundary (project convention, not from the documentation)",
            "source": SOURCE, "notice": notices.RULES}
