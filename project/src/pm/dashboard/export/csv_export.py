"""CSV reports. Every row/column carries its provenance so actual data, hypothetical inputs and model outputs cannot be confused."""
from __future__ import annotations

import io

import pandas as pd

from pm.explain import notices

RECORD_COLUMNS = ["section", "item", "value", "unit", "provenance", "note"]


def record_report_rows(expl: dict, source_kind: str, label: str, recorded: dict | None = None, partition: str | None = None) -> pd.DataFrame:
    """Long-format table for ONE record."""
    inp_prov = "ACTUAL DATA" if source_kind == "actual" else "HYPOTHETICAL INPUT"
    units = {"air_temp_k": "K", "process_temp_k": "K", "rot_speed_rpm": "rpm", "torque_nm": "Nm", "tool_wear_min": "min", "type": ""}
    rows: list[tuple] = [("record", "source", label, "", inp_prov, f"partition: {partition}" if partition else "values typed by the user")]
    for k, v in expl["inputs"].items():
        rows.append(("inputs", k, v, units[k], inp_prov, ""))
    if recorded is not None:
        rows.append(("recorded_outcome", "machine_failure", recorded["machine_failure"], "0/1", "ACTUAL DATA", "recorded label; never a model input"))
        for m in ("twf", "hdf", "pwf", "osf", "rnf"):
            rows.append(("recorded_outcome", m, recorded[m], "0/1", "ACTUAL DATA", "recorded failure-mode flag; never a model input"))
    for k, d in expl["engineering"].items():
        rows.append(("engineering", k, d["value"], d["unit"], "ENGINEERING CALCULATION", d["formula"]))
    m = expl["model"]
    rows += [("model", "failure_probability", m["failure_probability"], "probability", "MODEL PREDICTION", m["output_kind"]),
             ("model", "alert_threshold", m["threshold"], "probability", "MODEL PREDICTION", "provisional; unchanged since Phase 3"),
             ("model", "alert", int(m["alert"]), "0/1", "MODEL PREDICTION", "screening flag, not a diagnosis")]
    for r in expl["conditions"]["results"]:
        rows.append(("documented_rules", r["code"], r["status"], "", "DOCUMENTED RULE", f"{r['rule']} | {r['evidence']}"))
    rows.append(("documented_rules", "summary", expl["conditions"]["summary"], "", "DOCUMENTED RULE", expl["conditions"]["summary_text"]))
    rows.append(("documented_rules", "model_vs_rules", expl["model_vs_rules"], "", "DOCUMENTED RULE vs MODEL PREDICTION", ""))
    a = expl["tool_wear_advisory"]
    rows.append(("tool_wear_advisory", "level", a["level"], "", "TWF ADVISORY", "maintenance heuristic, not a failure prediction; not a probability"))
    attr = expl["attribution"]
    rows.append(("shap", "average_model_output", attr["base_value"], "probability", "MODEL ATTRIBUTION", "base value"))
    for f in attr["features"]:
        rows.append(("shap_feature", f["feature"], f["shap"], "probability points", "MODEL ATTRIBUTION", f"value = {f['value']} {f['unit']}".strip()))
    for g, v in attr["groups"].items():
        rows.append(("shap_group", g, v, "probability points", "MODEL ATTRIBUTION", "sum of member features"))
    for k, t in notices.ALL.items():
        rows.append(("notice", k, t, "", "NOTICE", ""))
    return pd.DataFrame(rows, columns=RECORD_COLUMNS)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")


def record_report_csv(expl, source_kind, label, recorded=None, partition=None) -> bytes:
    return to_csv_bytes(record_report_rows(expl, source_kind, label, recorded, partition))


def records_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Scored dataset rows with provenance-prefixed columns (actual_*, calc_*, model_*, rule_*, advisory_*)."""
    d = pd.DataFrame({"udi": scored["udi"], "product_id": scored["product_id"], "partition": scored["partition"]})
    for c in ("type", "air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min"):
        d[f"actual_{c}"] = scored[c]
    for c in ("machine_failure", "twf", "hdf", "pwf", "osf", "rnf"):
        d[f"actual_label_{c}"] = scored[c]
    for c in ("delta_t_k", "mechanical_power_w", "wear_torque_min_nm"):
        d[f"calc_{c}"] = scored[c]
    d["model_failure_probability"] = scored["failure_probability"]
    d["model_alert_at_0.365_provisional"] = scored["failure_alert"].astype(int)
    for c in ("hdf", "pwf", "osf"):
        d[f"rule_{c}_status"] = scored[f"{c}_status"]
    d["rule_summary"] = scored["rule_summary"]
    d["advisory_tool_wear_level"] = scored["tool_wear_advisory"]
    return d
