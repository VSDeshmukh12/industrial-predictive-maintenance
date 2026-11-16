"""8. Downloadable CSV / PDF reports."""
from __future__ import annotations

import streamlit as st

from pm.dashboard import components as ui, errors
from pm.dashboard.context import AppContext, explain_record, global_tables
from pm.dashboard.export import csv_export
from pm.dashboard.pages.analytics import select

try:                                              # PDF support is optional at runtime: CSV reports keep working without reportlab
    from pm.dashboard.export import pdf_export
except ImportError as _e:                         # pragma: no cover
    pdf_export, _PDF_ERR = None, _e

TITLE, ICON = "Reports", "📄"


def _pdf_button(label: str, build, file_name: str, key: str) -> None:
    if pdf_export is None:
        st.button(label + " (unavailable)", disabled=True, width="stretch", key=key, help="PDF support needs reportlab: pip install -r requirements-dashboard.txt")
        return
    try:
        data = build()
    except Exception as e:                          # a report failure must never take the page down
        st.button(label + " (failed)", disabled=True, width="stretch", key=key)
        st.error(f"The PDF could not be generated: {type(e).__name__}: {e}")
        return
    st.download_button(label, data, file_name, "application/pdf", width="stretch", key=key)


def render(ctx: AppContext) -> None:
    ui.hero("Reports", "Download single-record reports and dataset tables. Every file labels what is actual data, hypothetical input, model prediction, rule or advisory.")
    ui.synthetic_banner()
    if pdf_export is None:
        st.warning("PDF export is unavailable because the `reportlab` package is not installed; CSV downloads still work.")
    t1, t2 = st.tabs(["Single record", "Fleet / dataset"])
    with t1:
        src = ui.source_picker(ctx, "rep")
        expl = explain_record(ctx, src.record)
        stem = f"record_{src.udi}" if src.udi else "record_hypothetical"
        c1, c2 = st.columns(2)
        c1.download_button("⬇ CSV report", csv_export.record_report_csv(expl, src.kind, src.label, src.recorded, src.partition), f"{stem}_report.csv", "text/csv", width="stretch")
        with c2:
            _pdf_button("⬇ PDF report", lambda: pdf_export.record_pdf(expl, src.kind, src.label, ctx.info, src.recorded, src.partition), f"{stem}_report.pdf", "rep_pdf_record")
        st.caption("Contains: inputs with provenance, model probability and alert, engineering calculations, documented rules, separate TWF advisory, SHAP (nine features, four groups) and the mandatory notices.")
    with t2:
        c = st.columns(2)
        partition = c[0].selectbox("Partition", ["hold-out", "training", "all"], key="rep_part")
        types = c[1].multiselect("Product type", ["L", "M", "H"], default=["L", "M", "H"], key="rep_types")
        d = select(ctx.records, partition, types)
        desc = f"Partition: {partition}; product types: {', '.join(types) or 'none'}; {len(d):,} records."
        st.write(desc)
        if d.empty:
            st.info("No records selected.")
        c1, c2, c3 = st.columns(3)
        if not d.empty:
            c1.download_button("⬇ Records CSV", csv_export.to_csv_bytes(csv_export.records_table(d)), "records_scored.csv", "text/csv", width="stretch")
            with c2:
                _pdf_button("⬇ Fleet summary PDF", lambda: pdf_export.fleet_pdf(d, desc, ctx.info), "fleet_summary.pdf", "rep_pdf_fleet")
        with c3:
            shap_csv = errors.safe("Global SHAP table (Phase 5)", lambda: csv_export.to_csv_bytes(global_tables()["features"]))
            if shap_csv is not None:
                st.download_button("⬇ Global SHAP CSV", shap_csv, "global_shap_features.csv", "text/csv", width="stretch")
        st.caption("Records CSV columns are prefixed by provenance: actual_*, calc_*, model_*, rule_*, advisory_*. Training rows are in-sample for the model.")
    ui.footer(ctx)
