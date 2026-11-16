"""Dataset validation. Produces a structured report; never mutates the data.

Severity semantics
------------------
error   : the data violates the contract; downstream code must not run.
warning : a data-quality / documentation issue a human must be aware of.
info    : descriptive fact worth recording (not a problem by itself).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pm import config

ERROR, WARNING, INFO = "error", "warning", "info"


@dataclass
class Finding:
    check: str
    severity: str
    message: str
    count: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    findings: list[Finding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def add(self, check, severity, message, count=None, **details):
        self.findings.append(Finding(check, severity, message, count, details))

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == WARNING]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "summary": self.summary,
            "findings": [asdict(f) for f in self.findings],
        }

    def to_markdown(self) -> str:
        lines = ["# Dataset validation report", ""]
        lines.append(f"**Passed:** {self.passed}  |  errors: {len(self.errors)}  |  warnings: {len(self.warnings)}")
        lines += ["", "## Summary", ""]
        lines += [f"- {k}: {v}" for k, v in self.summary.items()]
        lines += ["", "## Findings", "", "| Severity | Check | Message | Count |", "|---|---|---|---|"]
        order = {ERROR: 0, WARNING: 1, INFO: 2}
        for f in sorted(self.findings, key=lambda f: order[f.severity]):
            lines.append(f"| {f.severity} | {f.check} | {f.message} | {'' if f.count is None else f.count} |")
        return "\n".join(lines) + "\n"


class DatasetValidationError(ValueError):
    pass


def raise_on_errors(report: ValidationReport) -> None:
    if not report.passed:
        msgs = "; ".join(f"[{f.check}] {f.message}" for f in report.errors)
        raise DatasetValidationError(f"Dataset validation failed: {msgs}")


def validate_dataset(df: pd.DataFrame, expected_rows: int | None = 10_000) -> ValidationReport:
    """Validate a canonical-column DataFrame (see loader.canonicalize)."""
    r = ValidationReport()
    expected_cols = list(config.COLUMN_MAP.values())

    # 1. Columns -----------------------------------------------------------
    missing = [c for c in expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in expected_cols]
    if missing:
        r.add("columns", ERROR, f"Missing columns: {missing}", len(missing))
        return r  # nothing further is meaningful
    r.add("columns", INFO, f"All {len(expected_cols)} expected columns present", len(expected_cols))
    if extra:
        r.add("columns_extra", WARNING, f"Unexpected extra columns: {extra}", len(extra))

    n = len(df)
    r.summary["n_rows"] = n
    r.summary["n_columns"] = len(df.columns)
    if expected_rows is not None and n != expected_rows:
        r.add("row_count", WARNING, f"Expected {expected_rows} rows, found {n}", n)

    # 2. Dtypes ------------------------------------------------------------
    num_cols = list(config.NUMERIC_FEATURES) + [config.TARGET] + list(config.MODE_COLUMNS) + ["udi"]
    bad_dtype = [c for c in num_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if bad_dtype:
        r.add("dtypes", ERROR, f"Non-numeric dtype in numeric columns: {bad_dtype}", len(bad_dtype))
        return r

    # 3. Missing values ----------------------------------------------------
    n_missing = int(df[expected_cols].isna().sum().sum())
    if n_missing:
        per_col = df[expected_cols].isna().sum()
        r.add("missing_values", ERROR, "Missing values present", n_missing,
              per_column=per_col[per_col > 0].to_dict())
        return r
    r.add("missing_values", INFO, "No missing values", 0)

    # 4. Identifiers -------------------------------------------------------
    for col in config.ID_COLUMNS:
        d = int(df[col].duplicated().sum())
        if d:
            r.add(f"unique_{col}", ERROR, f"{col} is not unique", d)
        else:
            r.add(f"unique_{col}", INFO, f"{col} unique per row (pure identifier)", 0)

    # 5. Type column -------------------------------------------------------
    bad_type = ~df["type"].isin(config.TYPE_CATEGORIES)
    if bad_type.any():
        r.add("type_values", ERROR, f"`type` outside {list(config.TYPE_CATEGORIES)}", int(bad_type.sum()),
              values=sorted(map(str, df.loc[bad_type, "type"].unique())))
    else:
        r.add("type_values", INFO, "`type` values are within {L, M, H}", 0)
    prefix_mismatch = df["product_id"].astype(str).str[0] != df["type"].astype(str)
    if prefix_mismatch.any():
        r.add("product_id_prefix", ERROR, "Product ID first letter != Type", int(prefix_mismatch.sum()))
    else:
        r.add("product_id_prefix", INFO, "Product ID prefix always equals Type (redundant identifier)", 0)

    # 6. Binary columns ----------------------------------------------------
    for col in (config.TARGET,) + tuple(config.MODE_COLUMNS):
        vals = set(pd.unique(df[col]))
        if not vals <= {0, 1}:
            r.add(f"binary_{col}", ERROR, f"{col} is not binary", None, values=sorted(map(str, vals)))

    # 7. Physical plausibility --------------------------------------------
    for col, (lo, hi) in config.PLAUSIBLE_RANGES.items():
        out = (df[col] < lo) | (df[col] > hi)
        if out.any():
            r.add(f"range_{col}", ERROR, f"{col} outside plausible range [{lo}, {hi}]", int(out.sum()),
                  observed_min=float(df[col].min()), observed_max=float(df[col].max()))
    if not any(f.check.startswith("range_") for f in r.findings):
        r.add("ranges", INFO, "All numeric features within generous plausibility bounds", 0)
    nonpos_dt = int((df["process_temp_k"] <= df["air_temp_k"]).sum())
    if nonpos_dt:
        r.add("delta_t_positive", WARNING, "process temperature <= air temperature", nonpos_dt)
    else:
        r.add("delta_t_positive", INFO, "process temperature > air temperature in all rows", 0)
    for col in config.NUMERIC_FEATURES:
        if df[col].nunique() <= 1:
            r.add(f"constant_{col}", ERROR, f"{col} is constant", None)

    # 8. Target / failure-mode consistency ---------------------------------
    modes = df[list(config.MODE_COLUMNS)]
    y = df[config.TARGET]
    any_mode = modes.sum(axis=1) > 0
    n_modes = modes.sum(axis=1)
    mf1_no_mode = int(((y == 1) & ~any_mode).sum())
    if mf1_no_mode:
        r.add("failure_without_mode", WARNING,
              "Machine failure = 1 but no failure-mode flag set (label inconsistency)", mf1_no_mode)
    mf0_with_mode = (y == 0) & any_mode
    if mf0_with_mode.any():
        by_mode = modes[mf0_with_mode].sum().to_dict()
        r.add("mode_without_failure", WARNING,
              "Machine failure = 0 although a failure-mode flag is set", int(mf0_with_mode.sum()),
              by_mode={k: int(v) for k, v in by_mode.items()})
    non_rnf = [m for m in config.MODE_COLUMNS if m != "rnf"]
    nonrnf_no_fail = int(((y == 0) & (df[non_rnf].sum(axis=1) > 0)).sum())
    if nonrnf_no_fail:
        r.add("nonrandom_mode_without_failure", WARNING,
              "Non-RNF mode flagged while Machine failure = 0", nonrnf_no_fail)
    rnf = df["rnf"] == 1
    r.add("rnf_label_agreement", INFO if not rnf.any() else WARNING,
          f"RNF=1 rows: {int(rnf.sum())}; of which Machine failure=1: {int(((df['rnf']==1)&(y==1)).sum())}",
          int(rnf.sum()), rnf_with_failure=int(((df["rnf"] == 1) & (y == 1)).sum()))
    r.add("multi_mode_rows", INFO, "Rows with more than one failure mode (overlapping labels)",
          int((n_modes > 1).sum()), distribution={int(k): int(v) for k, v in n_modes.value_counts().sort_index().items()})
    r.summary["failures_explained_by_a_mode"] = int(((y == 1) & any_mode).sum())

    # 9. Class balance -----------------------------------------------------
    pos = int(y.sum())
    prevalence = pos / n
    r.summary.update(n_failures=pos, n_normal=n - pos, prevalence=round(prevalence, 6),
                     imbalance_ratio=round((n - pos) / max(pos, 1), 2))
    r.add("class_balance", INFO, f"Failure prevalence {prevalence:.4%} ({pos} positives)", pos,
          imbalance_ratio=round((n - pos) / max(pos, 1), 2))
    by_type = df.groupby("type")[config.TARGET].agg(["size", "sum", "mean"])
    r.summary["by_type"] = {t: {"rows": int(row["size"]), "failures": int(row["sum"]),
                                "rate": round(float(row["mean"]), 4)} for t, row in by_type.iterrows()}
    thin = by_type[by_type["sum"] < 30]
    if len(thin):
        r.add("thin_strata", WARNING, "Type stratum with < 30 positives (unstable estimates)",
              int(len(thin)), types=list(thin.index))

    # 10. Duplicates in feature space -------------------------------------
    dup_feat = int(df.duplicated(subset=list(config.MODEL_FEATURES)).sum())
    r.add("duplicate_feature_rows", INFO if dup_feat == 0 else WARNING,
          "Rows with identical model-input values", dup_feat)

    # 11. Documented generator rules vs CSV (informational) ----------------
    power_w = df["torque_nm"] * df["rot_speed_rpm"] * 2 * np.pi / 60
    delta_t = df["process_temp_k"] - df["air_temp_k"]
    osf_thr = df["type"].map({"L": 11_000, "M": 12_000, "H": 13_000})
    rules = {
        "pwf": ((power_w < 3500) | (power_w > 9000)).astype(int),
        "hdf": ((delta_t < 8.6) & (df["rot_speed_rpm"] < 1380)).astype(int),
        "osf": ((df["tool_wear_min"] * df["torque_nm"]) > osf_thr).astype(int),
    }
    agreement = {k: float((v.to_numpy() == df[k].to_numpy()).mean()) for k, v in rules.items()}
    r.summary["rule_agreement"] = {k: round(v, 6) for k, v in agreement.items()}
    for k, v in agreement.items():
        r.add(f"rule_{k}", INFO if v == 1.0 else WARNING,
              f"{k.upper()} flag agrees with the documented generator rule on {v:.2%} of rows "
              "(thresholds from the UCI documentation; empirical check only)",
              int(round((1.0 - v) * n)))

    # 12. Documentation cross-check ---------------------------------------
    observed = {m: int(df[m].sum()) for m in config.DOCUMENTED_MODE_COUNTS}
    mismatch = {m: (config.DOCUMENTED_MODE_COUNTS[m], observed[m])
                for m in observed if observed[m] != config.DOCUMENTED_MODE_COUNTS[m]}
    if mismatch:
        r.add("documented_counts", WARNING,
              "Failure-mode counts differ from the dataset documentation (documented, observed)",
              len(mismatch), mismatch={k: list(v) for k, v in mismatch.items()})
    else:
        r.add("documented_counts", INFO, "Mode counts match the documentation", 0)
    zero_wear = int((df["tool_wear_min"] == 0).sum())
    r.add("tool_wear_zero", INFO, "Rows with tool wear = 0 (tool replacement events)", zero_wear)
    return r
