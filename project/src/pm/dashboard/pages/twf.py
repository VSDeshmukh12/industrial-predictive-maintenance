"""6. Separate TWF maintenance advisory."""
from __future__ import annotations

import streamlit as st

from pm.dashboard import charts, components as ui, errors, provenance as pv
from pm.dashboard.context import AppContext, advisory_tables, explain_record, phase4_report

TITLE, ICON = "TWF Advisory", "🛠️"


def render(ctx: AppContext) -> None:
    ui.hero("Tool-Wear (TWF) Maintenance Advisory", "A separate heuristic about where tool wear lies relative to the documented 200–240 min window.")
    ui.twf_banner()
    ui.synthetic_banner()
    src = ui.source_picker(ctx, "twf")
    expl = explain_record(ctx, src.record)
    adv, m = expl["tool_wear_advisory"], expl["model"]
    wear = adv["tool_wear_min"]

    ui.section("Advisory for this record", pv.ADVISORY)
    st.markdown(f"### {ui.chip(adv['level'])} &nbsp; tool wear {wear:g} min", unsafe_allow_html=True)
    st.plotly_chart(charts.twf_wear_strip(wear), width="stretch", config=ui.PLOT_CONFIG)
    st.write(adv["message"])
    st.caption("Bands (project convention, fixed a priori from the dataset documentation, never fitted): below 200 min none · 200–219 advisory · 220 or more elevated. It is not a probability and not a classifier output.")

    ui.section("Classifier output, shown separately", pv.PREDICTION)
    st.write(f"Model failure probability {m['failure_probability']:.3f} → {'ALERT' if m['alert'] else 'no alert'}. The advisory above does not depend on this number, and this number does not "
             "reliably reflect tool-wear failures.")
    if adv["level"] != "none" and not m["alert"]:
        st.info("The advisory flags tool wear while the classifier does not alert — expected, because the classifier cannot reliably detect TWF failures.")

    ui.section("Evidence and limitations", pv.ACTUAL, "Recorded TWF labels of the synthetic dataset (training partition) and the one-time Phase 4 hold-out result.")
    with errors.guarded("TWF evidence tables (Phase 4/5)"):
        t = advisory_tables()
        b = t["bands"].copy()
        b["twf_rate"] = (b["twf_rate"] * 100).round(2).astype(str) + "%"
        b["share_of_all_twf"] = (b["share_of_all_twf"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(b[["level", "rows", "twf", "twf_rate", "share_of_all_twf", "failures_any_mode"]].rename(
            columns={"rows": "training rows", "twf": "recorded TWF", "twf_rate": "TWF rate", "share_of_all_twf": "share of all TWF", "failures_any_mode": "failures (any mode)"}), hide_index=True, width="stretch")
        st.plotly_chart(charts.twf_rate_by_wear(t["by_wear_bin"]), width="stretch", config=ui.PLOT_CONFIG)
        fm = phase4_report()["failure_modes"]
        tw = fm[fm["group"] == "twf"].iloc[0]
        st.markdown(f"**Phase 4 hold-out:** the classifier alerted on {int(tw['detected'])} of {int(tw['n_failures'])} recorded TWF failures. The advisory bands have low precision: most records in the "
                    "advisory band show no recorded TWF, and TWF labels in the generator are random within the window.")
    ui.footer(ctx)
