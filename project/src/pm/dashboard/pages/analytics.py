"""7. Machine Analytics and visualisations (actual data, and model output versus recorded outcome)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pm.dashboard import charts, components as ui, provenance as pv
from pm.dashboard.context import AppContext

TITLE, ICON = "Machine Analytics", "📊"
FEATS = {"air_temp_k": "Air temperature (K)", "process_temp_k": "Process temperature (K)", "rot_speed_rpm": "Rotational speed (rpm)", "torque_nm": "Torque (Nm)",
         "tool_wear_min": "Tool wear (min)", "delta_t_k": "ΔT (K)", "mechanical_power_w": "Mechanical power (W)", "wear_torque_min_nm": "Wear × torque (min·Nm)"}
PAIRS = {"Torque vs speed": ("rot_speed_rpm", "torque_nm"), "ΔT vs speed (heat-dissipation region)": ("rot_speed_rpm", "delta_t_k"),
         "Power vs tool wear": ("tool_wear_min", "mechanical_power_w"), "Wear × torque vs tool wear": ("tool_wear_min", "wear_torque_min_nm"), "Process vs air temperature": ("air_temp_k", "process_temp_k")}


def select(rec: pd.DataFrame, partition: str, types: list[str]) -> pd.DataFrame:
    d = rec if partition == "all" else rec[rec["partition"] == partition]
    return d[d["type"].isin(types)]


def render(ctx: AppContext) -> None:
    ui.hero("Machine Analytics", "Distributions, operating points, failure patterns and model-versus-outcome views.")
    ui.synthetic_banner()
    c = st.columns(2)
    partition = c[0].selectbox("Partition", ["all", "training", "hold-out"], key="an_part")
    types = c[1].multiselect("Product type", ["L", "M", "H"], default=["L", "M", "H"], key="an_types")
    d = select(ctx.records, partition, types)
    if d.empty:
        st.info("No records selected.")
        return
    st.caption(f"{len(d):,} records · {int(d['machine_failure'].sum())} recorded failures. Plots of inputs and outcomes are ACTUAL data; plots using probabilities are MODEL PREDICTIONS.")
    t1, t2, t3, t4, t5 = st.tabs(["Distributions", "Operating points", "Failure patterns", "Correlations", "Model vs outcome"])
    with t1:
        ui.section("Feature distributions by recorded outcome", pv.ACTUAL)
        f = st.selectbox("Feature", list(FEATS), format_func=FEATS.get, key="an_feat")
        st.plotly_chart(charts.distribution(d, f, FEATS[f]), width="stretch", config=ui.PLOT_CONFIG)
    with t2:
        ui.section("Operating points", pv.ACTUAL)
        p = st.selectbox("View", list(PAIRS), key="an_pair")
        x, y = PAIRS[p]
        st.plotly_chart(charts.scatter_operating_points(d, x, y, FEATS[x], FEATS[y]), width="stretch", config=ui.PLOT_CONFIG)
    with t3:
        ui.section("Failure patterns", pv.ACTUAL)
        a, b = st.columns(2)
        a.plotly_chart(charts.failure_mode_counts(d), width="stretch", config=ui.PLOT_CONFIG)
        b.plotly_chart(charts.failure_rate_by_type(d), width="stretch", config=ui.PLOT_CONFIG)
        st.plotly_chart(charts.wear_failure_rate(d), width="stretch", config=ui.PLOT_CONFIG)
    with t4:
        ui.section("Correlations", pv.ACTUAL)
        st.plotly_chart(charts.correlation_heatmap(d), width="stretch", config=ui.PLOT_CONFIG)
        st.caption("Correlation describes association in synthetic data, not cause and effect.")
    with t5:
        ui.section("Model output versus recorded outcome", pv.PREDICTION)
        if partition != "hold-out":
            ui.notice("Training rows are in-sample for the model; metrics on them are optimistic. Choose the hold-out partition for out-of-sample values (consumed in Phase 4; display only).", "#B45309")
        st.plotly_chart(charts.probability_by_outcome(d, ctx.threshold), width="stretch", config=ui.PLOT_CONFIG)
        tp = int((d["failure_alert"] & (d["machine_failure"] == 1)).sum()); fp = int((d["failure_alert"] & (d["machine_failure"] == 0)).sum())
        fn = int((~d["failure_alert"] & (d["machine_failure"] == 1)).sum()); tn = int((~d["failure_alert"] & (d["machine_failure"] == 0)).sum())
        st.dataframe(pd.DataFrame({"": ["model alert", "no model alert"], "recorded failure": [tp, fn], "no recorded failure": [fp, tn]}), hide_index=True, width="stretch")
        ui.threshold_banner()
    ui.footer(ctx)
