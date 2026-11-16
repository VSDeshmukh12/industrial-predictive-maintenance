"""PDF reports (reportlab). DejaVu Sans (shipped with matplotlib) is registered so Δ, ≥, − and × render correctly."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from pm.explain import notices

_FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(_FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")

INK = colors.HexColor("#1E293B")
LINE = colors.HexColor("#CBD5E1")
_ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_ss["Title"], fontName="DV-B", fontSize=17, leading=21, textColor=INK, alignment=0, spaceAfter=2)
H2 = ParagraphStyle("H2", parent=_ss["Heading2"], fontName="DV-B", fontSize=11.5, leading=14, textColor=INK, spaceBefore=9, spaceAfter=3)
BODY = ParagraphStyle("B", parent=_ss["BodyText"], fontName="DV", fontSize=8.6, leading=11.4, textColor=INK)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=7.4, leading=9.6, textColor=colors.HexColor("#475569"))
CELL = ParagraphStyle("C", parent=BODY, fontSize=8, leading=10)
BANNER = ParagraphStyle("BN", parent=BODY, fontSize=8, leading=10.4, backColor=colors.HexColor("#FEF2F2"), borderColor=colors.HexColor("#B91C1C"),
                        borderWidth=0.8, borderPadding=5, spaceBefore=4, spaceAfter=6)


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _p(s, st=CELL):
    return Paragraph(_esc(s), st)


def _table(data, widths, header=True):
    t = Table([[c if not isinstance(c, str) else _p(c) for c in r] for r in data], colWidths=widths, repeatRows=1 if header else 0)
    sty = [("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if header:
        sty += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0"))]
    t.setStyle(TableStyle(sty))
    return t


def _shap_png(expl: dict) -> io.BytesIO:
    feats = sorted(expl["attribution"]["features"], key=lambda d: abs(d["shap"]))
    fig, ax = plt.subplots(figsize=(6.6, 3.0), dpi=170)
    names = [f"{f['label']} = {f['value'] if isinstance(f['value'], str) else format(f['value'], ',.4g')} {f['unit']}".strip() for f in feats]
    ax.barh(names, [f["shap"] for f in feats], color=["#C2410C" if f["shap"] >= 0 else "#1D4ED8" for f in feats])
    ax.axvline(0, color="#1E293B", lw=0.8)
    ax.set_xlabel("SHAP contribution to the model output (probability points)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    ax.set_title("Model attribution (red raises, blue lowers the model output) — not causation", fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    b = io.BytesIO()
    fig.savefig(b, format="png")
    plt.close(fig)
    b.seek(0)
    return b


def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 6.8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(15 * mm, 9 * mm, "AI4I 2020 is synthetic · offline evaluation · not real-time, not RUL, no real-world validation · attribution is not causation")
    canvas.drawRightString(195 * mm, 9 * mm, f"page {doc.page}")
    canvas.restoreState()


def _build(story) -> bytes:
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm, bottomMargin=16 * mm,
                      title="Predictive maintenance report", author="Predictive Maintenance dashboard (Phase 6)").build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buf.getvalue()


def _stamp(info: dict) -> Paragraph:
    t = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return _p(f"Generated {t} · frozen model SHA-256 {info['model_sha256'][:16]}… · alert level {info['threshold']:.3f} (provisional) · dataset SHA-256 {info['dataset_sha256'][:16]}…", SMALL)


# ------------------------------------------------------------------------------------------------ single record
def record_pdf(expl: dict, source_kind: str, label: str, info: dict, recorded: dict | None = None, partition: str | None = None) -> bytes:
    actual = source_kind == "actual"
    m, a, c, adv, e = expl["model"], expl["attribution"], expl["conditions"], expl["tool_wear_advisory"], expl["engineering"]
    S = [Paragraph("Machine record report", H1), _stamp(info), Spacer(1, 3)]
    S.append(Paragraph(_esc(notices.SYNTHETIC + " " + notices.SCOPE), BANNER))
    S.append(Paragraph("1. Record and provenance", H2))
    S.append(_p(f"Source: {'ACTUAL DATA — a recorded row of the synthetic dataset' if actual else 'HYPOTHETICAL INPUT — values entered by the user; nothing was measured'}. {label}"
                + (f" (partition: {partition})" if partition else ""), BODY))
    units = {"air_temp_k": "K", "process_temp_k": "K", "rot_speed_rpm": "rpm", "torque_nm": "Nm", "tool_wear_min": "min", "type": ""}
    rows = [["Input", "Value", "Unit"]] + [[k, str(v), units[k]] for k, v in expl["inputs"].items()]
    S.append(Spacer(1, 3)); S.append(_table(rows, [60 * mm, 40 * mm, 30 * mm]))
    if recorded is not None:
        S.append(Spacer(1, 3))
        S.append(_p(f"Recorded outcome (ACTUAL DATA; label only, never a model input): machine_failure = {recorded['machine_failure']}; recorded mode(s): {recorded['modes']}.", BODY))
        if partition == "training":
            S.append(_p("This row was in the model's training data: the model output below is in-sample and optimistic.", SMALL))
    S.append(Paragraph("2. Model prediction (frozen Random Forest classifier)", H2))
    S.append(_p(f"MODEL PREDICTION: estimated failure probability {m['failure_probability']:.3f} (uncalibrated). Provisional alert level {m['threshold']:.3f} → "
                f"{'ALERT' if m['alert'] else 'no alert'}. The alert is a screening flag, not a diagnosis.", BODY))
    S.append(Paragraph("3. Engineering calculations", H2))
    S.append(_table([["Quantity", "Value", "Formula"]] + [[k, f"{d['value']:,.0f} {d['unit']}" if abs(d["value"]) >= 1000 else f"{d['value']:.4g} {d['unit']}", d["formula"]] for k, d in e.items()], [45 * mm, 35 * mm, 100 * mm]))
    S.append(Paragraph("4. Documented condition rules (synthetic-generator rules; not model outputs)", H2))
    S.append(_p(c["summary_text"], BODY)); S.append(Spacer(1, 3))
    S.append(_table([["Rule", "Status", "Boundary", "Evidence"]] + [[r["name"], r["status"].replace("_", " "), r["rule"], r["evidence"]] for r in c["results"]], [38 * mm, 25 * mm, 55 * mm, 62 * mm]))
    S.append(_p(c["margin_convention"], SMALL))
    S.append(_p(notices.RULES, SMALL))
    S.append(Paragraph("5. Tool-wear (TWF) advisory — separate from the classifier", H2))
    S.append(_p(f"TWF ADVISORY level: {adv['level'].upper()} (tool wear {adv['tool_wear_min']:g} min). {adv['message']}", BODY))
    S.append(_p("This is a maintenance heuristic, not a failure prediction and not a probability. " + notices.TWF_ADVISORY, SMALL))
    S.append(Paragraph("6. Model attribution (SHAP)", H2))
    S.append(_p(f"Average model output {a['base_value']:.3f}; this record {m['failure_probability']:.3f}. Contributions of the nine features and four driver groups add up exactly "
                f"(additivity error {a['additivity_error']:.1e}).", BODY))
    S.append(KeepTogether([Image(_shap_png(expl), width=165 * mm, height=75 * mm)]))
    S.append(_table([["Driver group", "Contribution"]] + [[g.replace("_", " "), f"{v:+.4f}"] for g, v in a["groups"].items()], [70 * mm, 40 * mm]))
    S.append(_p(notices.ATTRIBUTION_NOT_CAUSATION, SMALL))
    S.append(Paragraph("7. Limitations", H2))
    for k in ("synthetic_data", "threshold_provisional", "scope"):
        S.append(_p("• " + notices.ALL[k], SMALL))
    return _build(S)


# ------------------------------------------------------------------------------------------------ fleet summary
def fleet_pdf(scored: pd.DataFrame, description: str, info: dict, top_n: int = 15) -> bytes:
    n = len(scored)
    alerts = int(scored["failure_alert"].sum())
    fails = int(scored["machine_failure"].sum())
    both = int((scored["failure_alert"] & (scored["machine_failure"] == 1)).sum())
    S = [Paragraph("Fleet summary report", H1), _stamp(info), Spacer(1, 3), Paragraph(_esc(notices.SYNTHETIC + " " + notices.SCOPE), BANNER)]
    S.append(Paragraph("1. Selection", H2)); S.append(_p(description, BODY))
    parts = scored["partition"].value_counts().to_dict()
    if parts.get("training"):
        S.append(_p("Rows from the training partition are in-sample for the model (optimistic). Hold-out rows were used once for the Phase 4 evaluation and are shown for display only.", SMALL))
    S.append(Paragraph("2. Counts", H2))
    S.append(_table([["Quantity", "Count", "Provenance"],
                     ["Records in selection", f"{n:,}", "ACTUAL DATA"], ["Recorded failures", f"{fails:,}", "ACTUAL DATA (label)"],
                     ["Records at/above provisional alert level (0.365)", f"{alerts:,}", "MODEL PREDICTION"],
                     ["Alerts on recorded failures", f"{both:,}", "MODEL PREDICTION vs ACTUAL DATA"],
                     ["Recorded failures without alert", f"{fails - both:,}", "MODEL PREDICTION vs ACTUAL DATA"],
                     ["Alerts on records without recorded failure (false alarms)", f"{alerts - both:,}", "MODEL PREDICTION vs ACTUAL DATA"],
                     ["Documented condition met (any of HDF/PWF/OSF)", f"{int((scored['rule_summary'] == 'condition_met').sum()):,}", "DOCUMENTED RULE"],
                     ["Tool-wear advisory: advisory / elevated", f"{int((scored['tool_wear_advisory'] == 'advisory').sum()):,} / {int((scored['tool_wear_advisory'] == 'elevated').sum()):,}", "TWF ADVISORY (heuristic)"]],
                    [100 * mm, 25 * mm, 55 * mm]))
    S.append(Paragraph(f"3. Top {top_n} records by model failure probability", H2))
    t = scored.sort_values("failure_probability", ascending=False).head(top_n)
    S.append(_table([["UDI", "Type", "Model p", "Alert", "Recorded failure", "Rules", "TWF advisory"]] +
                    [[str(int(r.udi)), r.type, f"{r.failure_probability:.3f}", "yes" if r.failure_alert else "no", "yes" if r.machine_failure else "no", r.rule_summary.replace("_", " "), r.tool_wear_advisory]
                     for r in t.itertuples()], [18 * mm, 14 * mm, 22 * mm, 16 * mm, 32 * mm, 40 * mm, 38 * mm]))
    S.append(Paragraph("4. Limitations", H2))
    for k in ("threshold_provisional", "tool_wear_advisory", "condition_rules"):
        S.append(_p("• " + notices.ALL[k], SMALL))
    return _build(S)
