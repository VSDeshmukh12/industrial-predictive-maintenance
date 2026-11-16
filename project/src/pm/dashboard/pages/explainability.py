"""5. Global and local SHAP explanations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pm.dashboard import charts, components as ui, errors, provenance as pv
from pm.dashboard.context import AppContext, explain_record, global_tables, global_training_values

TITLE, ICON = "SHAP Explanations", "🔍"


def render(ctx: AppContext) -> None:
    ui.hero("Explainable AI (SHAP)", "How the frozen Random Forest responds to its inputs — globally over the training data, and locally for one record.")
    ui.attribution_banner()
    ui.synthetic_banner()
    tab_g, tab_l = st.tabs(["Global (training partition)", "Local (one record)"])
    with tab_g:
        with errors.guarded("Global SHAP artifacts (Phase 5)"):
            ui.section("Global attribution", pv.ATTRIBUTION, "Mean |SHAP| over the 8,000 training rows (Phase 5, exact TreeSHAP). Explanations of the frozen model, not evidence of performance.")
            t = global_tables()
            c1, c2 = st.columns([1.3, 1])
            c1.plotly_chart(charts.global_feature_bar(t["features"]), width="stretch", config=ui.PLOT_CONFIG)
            c2.plotly_chart(charts.global_group_bar(t["groups"]), width="stretch", config=ui.PLOT_CONFIG)
            tv = global_training_values()
            st.plotly_chart(charts.shap_beeswarm(tv["values"], tv["features"], tv["display"]), width="stretch", config=ui.PLOT_CONFIG)
            st.caption("Correlated and derived inputs share credit (ΔT with the two temperatures; power with speed and torque), so the nine individual features and the four driver groups should be read together.")
            with st.expander("Table: nine features and four groups"):
                f = t["features"][["label", "mean_abs_shap", "share_of_total_abs"]].copy()
                f["share_of_total_abs"] = (f["share_of_total_abs"] * 100).round(1)
                st.dataframe(f.rename(columns={"label": "feature", "mean_abs_shap": "mean |SHAP|", "share_of_total_abs": "share (%)"}), hide_index=True, width="stretch")
                g = t["groups"][["group", "members", "mean_abs_shap", "share_of_total_abs"]].copy()
                g["share_of_total_abs"] = (g["share_of_total_abs"] * 100).round(1)
                st.dataframe(g.rename(columns={"mean_abs_shap": "mean |SHAP|", "share_of_total_abs": "share (%)"}), hide_index=True, width="stretch")
    with tab_l:
        src = ui.source_picker(ctx, "shap")
        expl = explain_record(ctx, src.record)
        a, m = expl["attribution"], expl["model"]
        ui.section("Local attribution", pv.ATTRIBUTION)
        st.write(f"Average model output **{a['base_value']:.3f}** → this record **{m['failure_probability']:.3f}** "
                 f"(sum of nine contributions = {m['failure_probability'] - a['base_value']:+.3f}; additivity error {a['additivity_error']:.1e}).")
        c1, c2 = st.columns([1.6, 1])
        c1.plotly_chart(charts.shap_waterfall(expl), width="stretch", config=ui.PLOT_CONFIG)
        c2.plotly_chart(charts.group_contrib_bar(a["groups"]), width="stretch", config=ui.PLOT_CONFIG)
        st.dataframe(pd.DataFrame([{"feature": f["label"], "value": f"{f['value']:,.4g}" if not isinstance(f["value"], str) else f["value"], "unit": f["unit"], "SHAP (prob. points)": round(f["shap"], 4)} for f in a["features"]]),
                     hide_index=True, width="stretch")
        ui.attribution_banner()
    ui.footer(ctx)
