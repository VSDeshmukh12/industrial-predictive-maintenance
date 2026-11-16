"""3. Manual Failure Prediction (HYPOTHETICAL inputs)."""
from __future__ import annotations

import streamlit as st

from pm import config
from pm.dashboard import charts, components as ui, data_access as da, provenance as pv
from pm.dashboard.context import AppContext, explain_record

TITLE, ICON = "Manual Prediction", "🧮"
LIMITS = {"air_temp_k": (250.0, 350.0, 0.1, "Air temperature (K)"), "process_temp_k": (250.0, 400.0, 0.1, "Process temperature (K)"),
          "rot_speed_rpm": (1.0, 10000.0, 1.0, "Rotational speed (rpm)"), "torque_nm": (0.0, 500.0, 0.1, "Torque (Nm)"),
          "tool_wear_min": (0.0, 1000.0, 1.0, "Tool wear (min)")}
KEYS = {c: f"in_{c}" for c in config.NUMERIC_FEATURES} | {"type": "in_type"}
PRESETS = {
    "Typical operating point (training medians)": None,
    "Heat-dissipation scenario (ΔT < 8.6 K, speed < 1380 rpm)": dict(air_temp_k=302.0, process_temp_k=310.0, rot_speed_rpm=1300.0, torque_nm=50.0, tool_wear_min=100.0, type="M"),
    "High-power scenario (P > 9000 W)": dict(air_temp_k=298.0, process_temp_k=308.5, rot_speed_rpm=1800.0, torque_nm=55.0, tool_wear_min=60.0, type="L"),
    "Overstrain scenario (wear × torque above the type limit)": dict(air_temp_k=298.0, process_temp_k=308.5, rot_speed_rpm=1400.0, torque_nm=62.0, tool_wear_min=200.0, type="L"),
    "Late tool wear (inside the 200–240 min window)": dict(air_temp_k=298.0, process_temp_k=308.5, rot_speed_rpm=1500.0, torque_nm=40.0, tool_wear_min=225.0, type="M"),
}


def _apply(values: dict) -> None:
    for k, v in values.items():
        st.session_state[KEYS[k]] = v


def _on_preset(ctx: AppContext) -> None:
    v = PRESETS[st.session_state["preset_choice"]]
    _apply(v if v else ctx.defaults)


def _on_load_udi(ctx: AppContext) -> None:
    src = ui.record_from_udi(ctx, int(st.session_state["load_udi"]))
    _apply(src.record)


def render(ctx: AppContext) -> None:
    ui.hero("Manual Failure Prediction", "Enter a hypothetical operating point and see what the frozen model, the documented rules and the TWF advisory say.")
    ui.notice("Everything on this page is a <b>hypothetical input</b>. No machine was measured and no outcome exists. The output is a <b>model prediction</b> about the values you typed, "
              "not a statement about any real plant.", pv.HYPOTHETICAL.colour, "Hypothetical.")
    ui.synthetic_banner()
    for k, key in KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = (st.session_state.get("manual_record") or ctx.defaults)[k]

    ui.section("1 · Choose a starting point", pv.HYPOTHETICAL)
    c1, c2, c3 = st.columns([2.2, 1.2, 1])
    c1.selectbox("Example scenario (hypothetical)", list(PRESETS), key="preset_choice", on_change=_on_preset, args=(ctx,))
    c2.number_input("Copy values from actual UDI", 1, 10000, 3237, 1, key="load_udi")
    c3.markdown("<div style='height:1.9rem'></div>", unsafe_allow_html=True)
    c3.button("Copy values", on_click=_on_load_udi, args=(ctx,), help="Copies the six recorded inputs into the form. After copying, the values are hypothetical (editable, nothing recorded).")

    ui.section("2 · Operating point", pv.HYPOTHETICAL)
    cols = st.columns(3)
    rec: dict = {}
    names = list(LIMITS)
    for n, col in zip(names, [cols[0], cols[1], cols[2], cols[0], cols[1]]):
        lo, hi, step, label = LIMITS[n]
        rec[n] = col.number_input(label, lo, hi, step=step, key=KEYS[n], format="%.1f" if step < 1 else "%.0f")
    rec["type"] = cols[2].selectbox("Product type", list(config.TYPE_CATEGORIES), key=KEYS["type"],
                                    help="Product quality variant: L (low), M (medium), H (high). It changes the overstrain limit of the documented rule.")
    st.session_state["manual_record"] = dict(rec)
    st.session_state["src_kind"] = "Hypothetical input (from Manual Prediction)"

    for w in da.extrapolation_warnings(rec, ctx.ranges):
        st.warning(w)
    try:
        expl = explain_record(ctx, rec)
    except ValueError as e:
        st.error(f"These inputs are not accepted by the frozen model: {e}")
        return
    m, adv, cond, e = expl["model"], expl["tool_wear_advisory"], expl["conditions"], expl["engineering"]

    ui.section("3 · Model prediction", pv.PREDICTION)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.plotly_chart(charts.probability_gauge(m["failure_probability"], m["threshold"]), width="stretch", config=ui.PLOT_CONFIG)
    with c2:
        st.markdown(f"### {ui.chip('alert' if m['alert'] else 'no_alert')} &nbsp; p = {m['failure_probability']:.3f}", unsafe_allow_html=True)
        st.write(f"The frozen Random Forest assigns this hypothetical operating point a failure probability of **{m['failure_probability']:.3f}** "
                 f"({'at or above' if m['alert'] else 'below'} the provisional alert level {m['threshold']:.3f}). The probability is uncalibrated; the alert is a screening flag, not a diagnosis.")
        ui.threshold_banner()

    ui.section("4 · Engineering calculations and documented rules", pv.CALC)
    ui.kpis([("ΔT", f"{e['delta_t_k']['value']:.2f} K", "process − air"), ("Mechanical power", f"{e['mechanical_power_w']['value']:,.0f} W", "2π·rpm·Nm / 60"),
             ("Wear × torque", f"{e['wear_torque_min_nm']['value']:,.0f} min·Nm", "overstrain indicator")])
    for r in cond["results"]:
        st.markdown(f"**{r['name']}** &nbsp; {ui.chip(r['status'])} &nbsp; <span style='color:#64748B'>{r['evidence']}</span>", unsafe_allow_html=True)
    st.caption(cond["summary_text"] + " Rules are documented synthetic-generator rules, not model outputs.")

    ui.section("5 · Tool-wear advisory (separate)", pv.ADVISORY)
    st.markdown(f"{ui.chip(adv['level'])} &nbsp; {adv['message']}", unsafe_allow_html=True)
    ui.twf_banner()

    ui.section("6 · Why did the model say that? (attribution)", pv.ATTRIBUTION)
    c1, c2 = st.columns([1.6, 1])
    c1.plotly_chart(charts.shap_waterfall(expl), width="stretch", config=ui.PLOT_CONFIG)
    c2.plotly_chart(charts.group_contrib_bar(expl["attribution"]["groups"]), width="stretch", config=ui.PLOT_CONFIG)
    ui.attribution_banner()
    st.caption("Changing a value above and watching the output move describes this trained model's behaviour, not what would happen to a real machine.")
    ui.footer(ctx)
