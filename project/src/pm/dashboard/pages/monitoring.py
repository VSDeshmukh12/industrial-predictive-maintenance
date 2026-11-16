"""2. Machine Monitoring on actual dataset records."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pm.dashboard import charts, components as ui, provenance as pv
from pm.dashboard.context import AppContext, explain_record

TITLE, ICON = "Machine Monitoring", "📡"
SORTS = {"Model probability (high → low)": ("failure_probability", False), "Tool wear (high → low)": ("tool_wear_min", False), "UDI (low → high)": ("udi", True)}


def filter_records(rec: pd.DataFrame, partition: str, types: list[str], outcome: str, alert: str, rule: str, advisory: str, query: str) -> pd.DataFrame:
    d = rec
    if partition != "all":
        d = d[d["partition"] == partition]
    d = d[d["type"].isin(types)]
    if outcome != "all":
        d = d[d["machine_failure"] == (1 if outcome == "recorded failure" else 0)]
    if alert != "all":
        d = d[d["failure_alert"] == (alert == "model alert")]
    if rule != "all":
        d = d[d["rule_summary"] == rule]
    if advisory != "all":
        d = d[d["tool_wear_advisory"] == advisory]
    q = query.strip()
    if q:
        d = d[d["product_id"].str.contains(q, case=False, regex=False) | d["udi"].astype(str).eq(q)]
    return d


def render(ctx: AppContext) -> None:
    ui.hero("Machine Monitoring", "Browse recorded machine records; compare the recorded outcome with the frozen model, the documented rules and the TWF advisory.")
    ui.synthetic_banner()
    ui.notice("Rows below are <b>actual</b> dataset records. The recorded failure label is shown for comparison only; it is never an input to the model. "
              "Training-partition rows are in-sample for the model; hold-out rows are out-of-sample but already consumed (display only).", pv.ACTUAL.colour)
    rec = ctx.records
    with st.expander("Filters", expanded=True):
        c = st.columns(4)
        partition = c[0].selectbox("Partition", ["hold-out", "training", "all"], help="hold-out: model output is out-of-sample (consumed in Phase 4). training: in-sample.")
        types = c[1].multiselect("Product type", ["L", "M", "H"], default=["L", "M", "H"])
        outcome = c[2].selectbox("Recorded outcome (actual)", ["all", "recorded failure", "no recorded failure"])
        alert = c[3].selectbox("Model alert (prediction)", ["all", "model alert", "no model alert"])
        c = st.columns(4)
        rule = c[0].selectbox("Documented rules", ["all", "condition_met", "approaching", "none"])
        advis = c[1].selectbox("TWF advisory", ["all", "none", "advisory", "elevated"])
        query = c[2].text_input("Search UDI or Product ID", "")
        sort = c[3].selectbox("Sort by", list(SORTS))
    d = filter_records(rec, partition, types, outcome, alert, rule, advis, query)
    col, asc = SORTS[sort]
    d = d.sort_values(col, ascending=asc)
    st.session_state["monitor_filter_desc"] = (f"partition={partition}; types={','.join(types)}; outcome={outcome}; alert={alert}; rules={rule}; advisory={advis}; search='{query}'")
    st.session_state["monitor_filter_key"] = (partition, tuple(types), outcome, alert, rule, advis, query)

    n = len(d)
    a = int(d["failure_alert"].sum()); f = int(d["machine_failure"].sum())
    ui.kpis([("Records in selection", f"{n:,}", "ACTUAL"), ("Recorded failures", f"{f:,}", "ACTUAL label"), ("Model alerts", f"{a:,}", f"probability ≥ {ctx.threshold:.3f} · PREDICTION"),
             ("Alerts on recorded failures", f"{int((d['failure_alert'] & (d['machine_failure'] == 1)).sum()):,}", "agreement"),
             ("Advisory / elevated", f"{int((d['tool_wear_advisory'] == 'advisory').sum())} / {int((d['tool_wear_advisory'] == 'elevated').sum())}", "TWF heuristic")])
    if n == 0:
        st.info("No records match the filters.")
        ui.footer(ctx)
        return
    view = d.head(500).assign(partition_=d["partition"].head(500))
    st.dataframe(
        pd.DataFrame({"UDI": view["udi"], "Product ID": view["product_id"], "Type": view["type"], "Partition": view["partition"],
                      "ACTUAL · recorded failure": view["machine_failure"].map({0: "no", 1: "yes"}), "ACTUAL · modes": view.apply(lambda r: _modes(r), axis=1),
                      "PREDICTION · probability": view["failure_probability"], "PREDICTION · alert": view["failure_alert"].map({True: "ALERT", False: "–"}),
                      "RULE · summary": view["rule_summary"].str.replace("_", " "), "TWF advisory": view["tool_wear_advisory"],
                      "Tool wear (min)": view["tool_wear_min"], "ΔT (K)": view["delta_t_k"].round(2), "Power (W)": view["mechanical_power_w"].round(0)}),
        hide_index=True, width="stretch", height=320,
        column_config={"PREDICTION · probability": st.column_config.ProgressColumn("PREDICTION · probability", min_value=0.0, max_value=1.0, format="%.3f")})
    st.caption(f"Showing the first {min(500, n)} of {n:,} records in the chosen order. Download the full selection on the Reports page.")

    ui.section("Inspect one record")
    ids = view["udi"].astype(int).tolist()
    c1, c2 = st.columns([2, 1])
    pick = c1.selectbox("Record from the table", ids, format_func=lambda u: f"UDI {u}", key="mon_pick")
    direct = c2.number_input("…or open a UDI directly (0 = use the table)", 0, 10000, 0, 1, key="mon_direct")
    udi = int(direct) if direct else int(pick)
    st.session_state["sel_udi"] = udi
    st.session_state["src_kind"] = "Actual dataset record"
    show_record(ctx, udi)
    ui.footer(ctx)


def _modes(r) -> str:
    from pm.dashboard.data_access import failure_mode_label
    return failure_mode_label(r)


def show_record(ctx: AppContext, udi: int) -> None:
    src = ui.record_from_udi(ctx, udi)
    expl = explain_record(ctx, src.record)
    m, adv, cond = expl["model"], expl["tool_wear_advisory"], expl["conditions"]
    st.markdown(f"### {src.label} &nbsp; {ui.badge(pv.ACTUAL)}", unsafe_allow_html=True)
    if src.partition == "training":
        ui.notice("This row was in the model's training data. The model output is <b>in-sample</b> and looks better than it would on unseen data.", "#B45309")
    else:
        ui.notice("This row is in the Phase 4 hold-out partition (consumed; display only).", "#475569")
    c1, c2, c3 = st.columns([1.1, 1.2, 1.1])
    with c1:
        ui.section("Recorded", pv.ACTUAL)
        i = expl["inputs"]
        st.dataframe(pd.DataFrame({"input": ["Type", "Air temperature (K)", "Process temperature (K)", "Rotational speed (rpm)", "Torque (Nm)", "Tool wear (min)"],
                                   "value": [i["type"], i["air_temp_k"], i["process_temp_k"], i["rot_speed_rpm"], i["torque_nm"], i["tool_wear_min"]]}).astype(str), hide_index=True, width="stretch")
        r = src.recorded
        st.markdown(f"**Recorded outcome:** {'FAILURE' if r['machine_failure'] else 'no failure'}" + (f" · mode(s): {r['modes']}" if r["machine_failure"] else ""))
    with c2:
        ui.section("Model output", pv.PREDICTION)
        st.plotly_chart(charts.probability_gauge(m["failure_probability"], m["threshold"]), width="stretch", config=ui.PLOT_CONFIG)
        st.markdown(ui.chip("alert" if m["alert"] else "no_alert"), unsafe_allow_html=True)
    with c3:
        ui.section("Rules and advisory (separate from the model)")
        for rr in cond["results"]:
            st.markdown(f"{rr['name']} &nbsp; {ui.chip(rr['status'])}", unsafe_allow_html=True)
        st.markdown(f"<br>{ui.badge(pv.ADVISORY)} &nbsp; {ui.chip(adv['level'])}", unsafe_allow_html=True)
        st.caption(adv["message"])
    st.caption(cond["summary_text"])
    with st.expander("Local SHAP attribution for this record", expanded=False):
        st.plotly_chart(charts.shap_waterfall(expl), width="stretch", config=ui.PLOT_CONFIG)
        ui.attribution_banner()
