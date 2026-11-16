"""4. Engineering Intelligence and condition rules."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pm.dashboard import charts, components as ui, errors, provenance as pv
from pm.dashboard.context import AppContext, advisory_tables, explain_record
from pm.explain import notices, rules

TITLE, ICON = "Engineering & Rules", "🔧"
CONSISTENCY = {"rule_and_alert": "The documented rule and the model alert agree.",
               "rule_only": "A documented rule is met but the model output is below the alert level (the model does not always match the rules).",
               "alert_only": "The model output is at or above the alert level although no documented rule is met.",
               "neither": "Neither a documented rule nor the model alert is triggered."}


def render(ctx: AppContext) -> None:
    ui.hero("Engineering Intelligence", "Transparent calculations and documented machine-condition rules, kept separate from the model's output.")
    ui.synthetic_banner()
    src = ui.source_picker(ctx, "eng")
    expl = explain_record(ctx, src.record)
    e, cond, m = expl["engineering"], expl["conditions"], expl["model"]

    ui.section("Engineering calculations", pv.CALC, "Computed from the six inputs; the same formulas the frozen model's physics features use.")
    ui.kpis([("ΔT", f"{e['delta_t_k']['value']:.2f} K", e["delta_t_k"]["formula"]), ("Mechanical power", f"{e['mechanical_power_w']['value']:,.0f} W", e["mechanical_power_w"]["formula"]),
             ("Wear × torque", f"{e['wear_torque_min_nm']['value']:,.0f} min·Nm", e["wear_torque_min_nm"]["formula"])])

    ui.section("Documented condition rules", pv.RULE)
    ui.notice(notices.RULES, pv.RULE.colour)
    st.markdown(f"**{cond['summary_text']}**")
    st.dataframe(pd.DataFrame([{"Condition": r["name"], "Status": r["status"].replace("_", " ").upper(), "Documented boundary": r["rule"], "This record": r["evidence"]} for r in cond["results"]]),
                 hide_index=True, width="stretch")
    st.plotly_chart(charts.rule_proximity(cond["results"], e, src.record), width="stretch", config=ui.PLOT_CONFIG)
    st.caption(f"{cond['margin_convention']}. 'Approaching' is a descriptive proximity label, not a risk score. "
               "At exactly ΔT = 8.6 K the HDF outcome depends on floating-point round-off in the generator, so it should not be read as meaningful.")

    ui.section("Rules versus the model", pv.PREDICTION)
    st.write(f"Model output {m['failure_probability']:.3f} → {'ALERT' if m['alert'] else 'no alert'} (provisional level {m['threshold']:.3f}). " + CONSISTENCY[expl["model_vs_rules"]])
    st.caption("No rule match does not rule out failure: tool-wear and random failures follow no rule.")

    with st.expander("Evidence: what recorded failure rates look like by rule status (training partition, descriptive)"):
        with errors.guarded("Rule-status evidence (Phase 5)"):
            t = advisory_tables()["rule_status"].copy()
            t["failure_rate"] = (t["failure_rate"] * 100).round(2).astype(str) + "%"
            st.dataframe(t[["rule", "status", "rows", "failures", "failure_rate"]], hide_index=True, width="stretch")
            st.caption("Failures among 'approaching' rows come from other modes; 'approaching' for HDF shows essentially no lift over 'not met'. It is a proximity label, not a risk score.")
    ui.footer(ctx)
