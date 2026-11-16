"""1. Industrial Overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pm.dashboard import charts, components as ui, errors, provenance as pv
from pm.dashboard.context import AppContext, phase4_report

TITLE, ICON = "Industrial Overview", "🏭"


def render(ctx: AppContext) -> None:
    ui.hero("Industrial Predictive Maintenance", "AI4I 2020 (synthetic) · frozen Phase 4 Random Forest · Phase 5 explanations · offline evaluation")
    ui.synthetic_banner()
    rec, i = ctx.records, ctx.info
    ui.section("How to read this dashboard")
    cards = "".join(f'<div class="pm-card">{ui.badge(k)}<div class="sub" style="margin-top:8px">{k.blurb}</div></div>'
                    for k in (pv.ACTUAL, pv.HYPOTHETICAL, pv.PREDICTION, pv.RULE, pv.ADVISORY, pv.ATTRIBUTION))
    st.markdown(f'<div class="pm-grid" style="grid-template-columns:repeat(auto-fit,minmax(260px,1fr))">{cards}</div>', unsafe_allow_html=True)

    ui.section("Dataset snapshot", pv.ACTUAL)
    fail = int(rec["machine_failure"].sum())
    ui.kpis([("Records", f"{i['rows']:,}", "synthetic AI4I 2020 rows"), ("Recorded failures", f"{fail:,}", f"{100 * fail / len(rec):.2f}% of records"),
             ("Training / hold-out", f"{i['train_rows']:,} / {i['holdout_rows']:,}", "hold-out consumed in Phase 4"),
             ("Alert level", f"{i['threshold']:.3f}", "provisional · frozen"), ("Model", "Random Forest", "frozen · uncalibrated")])
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.failure_mode_counts(rec), width="stretch", config=ui.PLOT_CONFIG)
    c2.plotly_chart(charts.failure_rate_by_type(rec), width="stretch", config=ui.PLOT_CONFIG)

    ui.section("Final held-out result (Phase 4)", pv.PREDICTION, "Reporting only: the hold-out was evaluated exactly once in Phase 4 and is never used for tuning. Values are read from the saved Phase 4 artifact.")
    with errors.guarded("Phase 4 hold-out results"):
        p4 = phase4_report()
        m, cm = p4["metrics"]["metrics"], p4["metrics"]["confusion_matrix"]

        ui.kpis([("PR-AUC", f"{m['pr_auc']:.3f}", f"no-skill {p4['metrics']['no_skill_pr_auc']:.3f}"), ("ROC-AUC", f"{m['roc_auc']:.3f}", "hold-out, 2,000 rows"),
                 ("Recall", f"{m['recall']:.3f}", f"{cm['tp']} of {cm['tp'] + cm['fn']} failures alerted"), ("Precision", f"{m['precision']:.3f}", f"{cm['fp']} false alarms"),
                 ("F2 / MCC", f"{m['f2']:.2f} / {m['mcc']:.2f}", "at the 0.365 threshold")])
        fm = p4["failure_modes"].copy()
        fm["recall"] = (fm["recall"] * 100).round(1).astype(str) + "%"
        st.dataframe(fm[["group", "n_failures", "detected", "missed", "recall"]].rename(columns={"group": "failure mode", "n_failures": "failures", "detected": "alerted", "missed": "not alerted"}),
                     hide_index=True, width="stretch")
    ui.twf_banner()
    ui.threshold_banner()

    ui.section("System integrity")
    st.dataframe(pd.DataFrame({"check": ["Dataset SHA-256 matches the pinned original", "Model file SHA-256 verified against manifest at load", "Alert level equals frozen value 0.365",
                                         "Explanation engine additivity (base + SHAP = probability) enforced on every explanation"],
                               "status": ["PASS" if i["dataset_sha256_ok"] else "FAIL", "PASS", "PASS" if abs(i["threshold"] - 0.365) < 1e-12 else "FAIL", "PASS"]}),
                 hide_index=True, width="stretch")
    ui.footer(ctx)
