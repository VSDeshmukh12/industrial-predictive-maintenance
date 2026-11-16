"""Reusable widgets: provenance badges, notices, KPI cards, status chips and the shared record-source picker."""
from __future__ import annotations

from dataclasses import dataclass

import streamlit as st

from pm.dashboard import provenance as pv
from pm.dashboard.context import AppContext
from pm.explain import notices

STATUS_COLOURS = {"condition_met": "#C2410C", "approaching": "#B45309", "not_met": "#0F766E", "none": "#0F766E", "advisory": "#B45309",
                  "elevated": "#C2410C", "alert": "#C2410C", "no_alert": "#0F766E"}
STATUS_TEXT = {"condition_met": "CONDITION MET", "approaching": "APPROACHING", "not_met": "NOT MET", "none": "NO ADVISORY", "advisory": "ADVISORY",
               "elevated": "ELEVATED", "alert": "ALERT", "no_alert": "NO ALERT"}


PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False}      # the toolbar overlaps chart titles on phones


def badge(kind: pv.Kind) -> str:
    return f'<span class="pm-badge" style="background:{kind.colour}" title="{kind.blurb}">{kind.label}</span>'


def chip(status: str, text: str | None = None) -> str:
    return f'<span class="pm-chip" style="background:{STATUS_COLOURS.get(status, "#475569")}">{text or STATUS_TEXT.get(status, status.upper())}</span>'


def section(title: str, kind: pv.Kind | None = None, note: str | None = None) -> None:
    b = badge(kind) if kind else ""
    st.markdown(f'<div class="pm-section"><span class="t">{title}</span>{b}</div>', unsafe_allow_html=True)
    if note:
        st.caption(note)


def notice(text: str, colour: str = "#64748B", title: str | None = None) -> None:
    t = f"<b>{title}</b> " if title else ""
    st.markdown(f'<div class="pm-notice" style="--c:{colour}">{t}{text}</div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str]]) -> None:
    cards = "".join(f'<div class="pm-card"><div class="lbl">{l}</div><div class="val">{v}</div><div class="sub">{s}</div></div>' for l, v, s in items)
    st.markdown(f'<div class="pm-grid">{cards}</div>', unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="pm-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def synthetic_banner() -> None:
    notice(notices.SYNTHETIC + " " + notices.SCOPE, "#B91C1C", "Synthetic data.")


def attribution_banner() -> None:
    st.caption("Reading the charts: red bars raise and blue bars lower the model output; the bars add up exactly from the average output to this row's output.")
    notice(notices.ATTRIBUTION_NOT_CAUSATION, pv.ATTRIBUTION.colour, "Attribution, not causation.")


def twf_banner() -> None:
    notice("The tool-wear advisory is a <b>maintenance heuristic, not a failure prediction</b>. " + notices.TWF_ADVISORY, pv.ADVISORY.colour, "TWF advisory.")


def threshold_banner() -> None:
    notice(notices.THRESHOLD_PROVISIONAL, pv.PREDICTION.colour, "Provisional threshold.")


def footer(ctx: AppContext) -> None:
    i = ctx.info
    st.markdown(f'<div class="pm-foot">Frozen model SHA-256 {i["model_sha256"][:12]}… · alert level {i["threshold"]:.3f} (provisional) · AI4I 2020 is synthetic · '
                f'no real-time prediction, no remaining-useful-life estimate, no real-world validation · attribution is not causation.</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------------------------------------ record source
@dataclass
class Source:
    kind: str                 # provenance.SOURCE_ACTUAL | SOURCE_MANUAL
    record: dict              # the six model inputs
    label: str
    udi: int | None = None
    product_id: str | None = None
    partition: str | None = None
    recorded: dict | None = None     # recorded outcome (ACTUAL only; display, never a model input)


def record_from_udi(ctx: AppContext, udi: int) -> Source:
    r = ctx.records[ctx.records["udi"] == udi].iloc[0]
    from pm.dashboard.data_access import RAW, failure_mode_label
    rec = {k: (r[k] if k == "type" else float(r[k])) for k in RAW}
    recorded = {"machine_failure": int(r["machine_failure"]), "modes": failure_mode_label(r),
                **{m: int(r[m]) for m in ("twf", "hdf", "pwf", "osf", "rnf")}}
    return Source(pv.SOURCE_ACTUAL, rec, f"UDI {int(udi)} · {r['product_id']}", int(udi), str(r["product_id"]), str(r["partition"]), recorded)


def manual_record(ctx: AppContext) -> dict:
    d = st.session_state.get("manual_record") or dict(ctx.defaults)
    return dict(d)


def source_picker(ctx: AppContext, prefix: str, allow_manual: bool = True) -> Source:
    """Shared chooser: an ACTUAL dataset record (by UDI) or the HYPOTHETICAL values from the Manual Prediction page."""
    opts = ["Actual dataset record"] + (["Hypothetical input (from Manual Prediction)"] if allow_manual else [])
    cur = st.session_state.get("src_kind", opts[0])
    cur = cur if cur in opts else opts[0]
    c1, c2 = st.columns([2, 1])
    with c1:
        choice = st.radio("Record source", opts, index=opts.index(cur), horizontal=True, key=f"{prefix}_src",
                          help="Actual = a recorded row of the synthetic dataset. Hypothetical = the values you set on the Manual Prediction page.")
    st.session_state["src_kind"] = choice
    if choice == opts[0]:
        with c2:
            udi = st.number_input("UDI (1–10000)", 1, 10000, int(st.session_state.get("sel_udi", 3237)), 1, key=f"{prefix}_udi")
        st.session_state["sel_udi"] = int(udi)
        src = record_from_udi(ctx, int(udi))
        st.markdown(f"{badge(pv.ACTUAL)}&nbsp; inputs for **{src.label}** are a recorded row ({src.partition} partition).", unsafe_allow_html=True)
        if src.partition == "training":
            st.caption("This row was part of the model's training data, so the model output below is an in-sample value and flatters the model.")
        else:
            st.caption("This row is in the Phase 4 hold-out partition (consumed in Phase 4; it is shown here for display only and never used for tuning).")
        return src
    rec = manual_record(ctx)
    st.markdown(f"{badge(pv.HYPOTHETICAL)}&nbsp; inputs are the values currently set on the <i>Manual Prediction</i> page; no machine was measured.", unsafe_allow_html=True)
    return Source(pv.SOURCE_MANUAL, rec, "Hypothetical input")
