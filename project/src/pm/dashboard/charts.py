"""Plotly figure builders. Pure functions: data in, figure out. Colour-blind-safe palette, units on every axis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pm.explain import advisory, rules
from pm.explain.engine import FEATURES, GROUPS, LABELS, UNITS

INK = "#1E293B"
MUTED = "#64748B"
GRID = "#E2E8F0"
OK = "#0F766E"
FAIL = "#C2410C"
BLUE = "#1D4ED8"
AMBER = "#B45309"
POS = "#C2410C"      # pushes the model output up
NEG = "#1D4ED8"      # pushes it down
SHORT = {"air_temp_k": "Air temp", "process_temp_k": "Process temp", "rot_speed_rpm": "Speed", "torque_nm": "Torque", "tool_wear_min": "Tool wear",
         "delta_t": "ΔT", "power_w": "Power", "overstrain": "Wear×torque", "type": "Type"}
GROUP_LABELS = {"thermal": "Thermal (air, process, ΔT)", "speed_torque_power": "Speed / torque / power",
                "wear_overstrain": "Wear / overstrain", "product_type": "Product type"}
STATUS_COLOURS = {"condition_met": FAIL, "approaching": AMBER, "not_met": OK}


def _base(fig: go.Figure, height: int = 340, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, title=dict(text=title, font=dict(size=15, color=INK)) if title else None,
                      margin=dict(l=10, r=10, t=44 if title else 14, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color=INK),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ------------------------------------------------------------------------------------------------ model output
def probability_gauge(p: float, threshold: float) -> go.Figure:
    """Probability on a 0-1 bar with the provisional alert level marked. Colour is NOT the only cue (text + marker)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p, number=dict(valueformat=".3f", font=dict(size=34)),
        title=dict(text="Estimated failure probability<br><span style='font-size:11px;color:#64748B'>uncalibrated RF output</span>"),
        gauge=dict(axis=dict(range=[0, 1], tickvals=[0, 0.25, 0.5, 0.75, 1]), bar=dict(color=FAIL if p >= threshold else BLUE, thickness=0.28),
                   steps=[dict(range=[0, threshold], color="#EEF2FF"), dict(range=[threshold, 1], color="#FFEDD5")],
                   threshold=dict(line=dict(color=INK, width=3), thickness=0.85, value=threshold))))
    fig.add_annotation(x=0.5, y=-0.02, xref="paper", yref="paper", showarrow=False, font=dict(size=11, color=MUTED),
                       text=f"▎provisional alert level = {threshold:.3f}")
    return _base(fig, 250)


def shap_waterfall(expl: dict, top_k: int = 9) -> go.Figure:
    """Waterfall in probability space: average model output + contributions = this row's model output (exact)."""
    a = expl["attribution"]
    feats = sorted(a["features"], key=lambda d: -abs(d["shap"]))
    shown, rest = feats[:top_k], feats[top_k:]
    names = ["Average model output"] + [_feat_name(f) for f in shown]
    vals = [a["base_value"]] + [f["shap"] for f in shown]
    measure = ["absolute"] + ["relative"] * len(shown)
    if rest:
        names.append(f"{len(rest)} other features"); vals.append(sum(f["shap"] for f in rest)); measure.append("relative")
    names.append("This row's model output"); vals.append(expl["model"]["failure_probability"]); measure.append("total")
    fig = go.Figure(go.Waterfall(orientation="h", y=names, x=vals, measure=measure, base=0,
                                 increasing=dict(marker=dict(color=POS)), decreasing=dict(marker=dict(color=NEG)),
                                 totals=dict(marker=dict(color=INK)), connector=dict(line=dict(color=GRID)),
                                 text=[f"{v:+.3f}" if m == "relative" else f"{v:.3f}" for v, m in zip(vals, measure)], textposition="outside", cliponaxis=False,
                                 hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    cum = np.cumsum([v for v, m in zip(vals, measure) if m != "total"])
    hi, lo = max(float(cum.max()), vals[-1], vals[0]), min(float(cum.min()), 0.0)
    pad = 0.18 * (hi - lo or 1.0)
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(title="Model output (probability)", range=[lo - 0.06 * (hi - lo or 1.0), hi + pad])
    return _base(fig, 60 + 34 * len(names), "Local attribution (SHAP)").update_layout(margin=dict(r=52))


def _feat_name(f: dict) -> str:
    v = f["value"]
    vs = v if isinstance(v, str) else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}")
    return f"{SHORT[f['feature']]} = {vs} {f['unit']}".strip()


def group_contrib_bar(groups: dict) -> go.Figure:
    names = [GROUP_LABELS[g] for g in groups]
    vals = list(groups.values())
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=[POS if v >= 0 else NEG for v in vals],
                           text=[f"{v:+.3f}" for v in vals], textposition="outside", hovertemplate="%{y}: %{x:+.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", automargin=True)
    lo, hi = min(min(vals), 0.0), max(max(vals), 0.0)
    span = (hi - lo) or 1.0
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes(title="Contribution (probability points)", range=[lo - 0.3 * span, hi + 0.25 * span])
    return _base(fig, 250, "Four driver groups").update_layout(margin=dict(r=52))


# ------------------------------------------------------------------------------------------------ global SHAP
def global_feature_bar(tbl: pd.DataFrame) -> go.Figure:
    t = tbl.sort_values("mean_abs_shap")
    err = None
    if {"share_ci_lo", "share_ci_hi"} <= set(t.columns):
        err = dict(type="data", symmetric=False, array=(t["share_ci_hi"] - t["share_of_total_abs"]) * 100,
                   arrayminus=(t["share_of_total_abs"] - t["share_ci_lo"]) * 100)
    fig = go.Figure(go.Bar(x=t["share_of_total_abs"] * 100, y=t["label"], orientation="h", marker_color=BLUE, error_x=err,
                           customdata=t["mean_abs_shap"], hovertemplate="%{y}: %{x:.1f}% of total |SHAP| (mean |SHAP| %{customdata:.4f})<extra></extra>"))
    fig.update_xaxes(title="Share of total mean |SHAP| (%); whiskers = 95% bootstrap interval")
    return _base(fig, 380, "Global: nine individual features")


def global_group_bar(tbl: pd.DataFrame) -> go.Figure:
    t = tbl.sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(x=t["share_of_total_abs"] * 100, y=[GROUP_LABELS[g] for g in t["group"]], orientation="h", marker_color="#0369A1",
                           text=[f"{v * 100:.1f}%" for v in t["share_of_total_abs"]], textposition="outside",
                           hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    fig.update_traces(cliponaxis=False)
    fig.update_xaxes(title="Share of total mean |SHAP| (%)", range=[0, float(t["share_of_total_abs"].max()) * 100 * 1.22])
    return _base(fig, 280, "Global: four driver groups")


def shap_beeswarm(values: np.ndarray, features: np.ndarray, display: pd.DataFrame, max_rows: int = 2500, seed: int = 0) -> go.Figure:
    """Compact SHAP summary: one dot per training row; x = SHAP (probability points); colour = feature value rank."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=min(max_rows, len(values)), replace=False)
    order = np.argsort(-np.abs(values).mean(axis=0))
    fig = go.Figure()
    for rank, j in enumerate(order[::-1]):
        f = str(features[j])
        col = display[f].iloc[idx]
        if f == "type":
            c = col.map({"L": 0.0, "M": 0.5, "H": 1.0}).to_numpy(float)
        else:
            c = col.rank(pct=True).to_numpy(float)
        y = rank + rng.uniform(-0.28, 0.28, len(idx))
        fig.add_trace(go.Scattergl(x=values[idx, j], y=y, mode="markers", marker=dict(size=4, color=c, colorscale="RdBu_r", cmin=0, cmax=1, opacity=0.65,
                                   colorbar=dict(title="feature value<br>(low→high)", thickness=10, len=0.5) if rank == 0 else None, showscale=rank == 0),
                                   showlegend=False, hovertemplate=f"{LABELS[f]}<br>SHAP %{{x:+.4f}}<extra></extra>"))
    fig.update_yaxes(tickvals=list(range(len(order))), ticktext=[LABELS[str(features[j])] for j in order[::-1]])
    fig.update_xaxes(title="SHAP value (probability points)")
    return _base(fig, 460, "SHAP summary (direction and spread)")


# ------------------------------------------------------------------------------------------------ rules and TWF
def rule_proximity(expl_rules: list[dict], engineering: dict, record: dict) -> go.Figure:
    """Three bullet-style bars: where the record sits relative to each documented boundary (dashed line = 100% of the boundary).
    PWF has two edges (3500 W and 9000 W) and HDF also needs speed < 1380 rpm; both are marked/stated so a short bar is never misread."""
    lim = rules.OSF_LIMIT[record["type"]]
    dt, pw, ov = (engineering["delta_t_k"]["value"], engineering["mechanical_power_w"]["value"], engineering["wear_torque_min_nm"]["value"])
    rpm = float(record["rot_speed_rpm"])
    rows = [(f"Overstrain (OSF)<br>wear×torque as % of the<br>type-{record['type']} limit", 100 * ov / lim, "osf"),
            ("Power (PWF)<br>P as % of 9000 W<br>(met above 100% or<br>below 3500 W)", 100 * pw / rules.PWF_MAX_W, "pwf"),
            (f"Heat dissipation (HDF)<br>ΔT as % of 8.6 K (needs<br>speed < 1380 rpm; now {rpm:.0f})", 100 * dt / rules.HDF_DT_MAX_K, "hdf")]
    status = {r["code"]: r["status"] for r in expl_rules}
    fig = go.Figure()
    for label, v, c in rows:
        fig.add_trace(go.Bar(x=[v], y=[label], orientation="h", marker_color=STATUS_COLOURS[status[c]], showlegend=False,
                             text=[status[c].replace("_", " ")], textposition="outside", textangle=0, cliponaxis=False,
                             hovertemplate=f"{label}<br>%{{x:.0f}}%<extra></extra>"))
    fig.add_vline(x=100, line=dict(color=INK, width=2, dash="dash"), annotation_text="documented boundary", annotation_position="top")
    low = 100 * rules.PWF_MIN_W / rules.PWF_MAX_W                       # lower PWF edge on the same scale
    fig.add_shape(type="line", x0=low, x1=low, y0=0.6, y1=1.4, xref="x", yref="y", line=dict(color=INK, width=2, dash="dot"))
    fig.add_annotation(x=low, y=0.6, xref="x", yref="y", text="3500 W lower edge", showarrow=False, yshift=8, xanchor="left", font=dict(size=10, color=INK))
    fig.update_yaxes(autorange="reversed", automargin=True)
    fig.update_xaxes(title="% of boundary value", range=[0, max(160, max(r[1] for r in rows) * 1.1)])
    return _base(fig, 330, "Proximity to the documented conditions").update_layout(margin=dict(r=70))


def twf_wear_strip(wear: float) -> go.Figure:
    """Tool-wear axis with the advisory bands and the record marked."""
    xmax = max(260.0, wear * 1.05)
    fig = go.Figure()
    for lo, hi, col, name in ((0, advisory.ADVISORY_MIN, "#DCFCE7", "none (< 200 min)"), (advisory.ADVISORY_MIN, advisory.ELEVATED_MIN, "#FEF3C7", "advisory (200–219)"),
                              (advisory.ELEVATED_MIN, xmax, "#FFEDD5", "elevated (≥ 220)")):
        fig.add_shape(type="rect", x0=lo, x1=hi, y0=0, y1=1, fillcolor=col, line=dict(width=0), layer="below")
        fig.add_annotation(x=(lo + hi) / 2, y=0.18, text=name, showarrow=False, font=dict(size=11, color=INK))
    fig.add_shape(type="line", x0=wear, x1=wear, y0=0, y1=1, line=dict(color=INK, width=4))
    fig.add_annotation(x=wear, y=0.9, text=f"tool wear {wear:g} min", showarrow=False, bgcolor="rgba(255,255,255,0.85)", font=dict(size=12, color=INK))
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_xaxes(title="Tool wear (min)", range=[0, xmax])
    return _base(fig, 170)


def twf_rate_by_wear(tbl: pd.DataFrame) -> go.Figure:
    t = tbl.copy()
    fig = go.Figure(go.Bar(x=t["wear_bin"], y=t["twf_rate"] * 100, marker_color=[AMBER if lo >= advisory.ADVISORY_MIN else MUTED for lo in t["lo"]],
                           customdata=np.column_stack([t["rows"], t["twf"]]), hovertemplate="wear %{x} min<br>TWF rate %{y:.2f}% (%{customdata[1]} of %{customdata[0]} rows)<extra></extra>"))
    fig.update_xaxes(title="Tool wear bin (min)")
    fig.update_yaxes(title="Recorded TWF rate (%)")
    return _base(fig, 330, "Recorded tool-wear failures by wear (training partition, synthetic)")


# ------------------------------------------------------------------------------------------------ analytics
_OUTCOME = {0: "no failure", 1: "failure"}


def distribution(df: pd.DataFrame, col: str, label: str, nbins: int = 40) -> go.Figure:
    fig = go.Figure()
    for k, colr in ((0, "#94A3B8"), (1, FAIL)):
        s = df.loc[df["machine_failure"] == k, col]
        fig.add_trace(go.Histogram(x=s, nbinsx=nbins, name=_OUTCOME[k], marker_color=colr, opacity=0.75, histnorm="percent"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title=label)
    fig.update_yaxes(title="% of group")
    return _base(fig, 300, f"{label} — recorded outcome groups (actual data)")


def scatter_operating_points(df: pd.DataFrame, x: str, y: str, xl: str, yl: str, max_rows: int = 5000, seed: int = 0) -> go.Figure:
    d = df.sample(min(max_rows, len(df)), random_state=seed) if len(df) > max_rows else df
    fig = go.Figure()
    for k, colr, size in ((0, "#94A3B8", 4), (1, FAIL, 7)):
        s = d[d["machine_failure"] == k]
        fig.add_trace(go.Scattergl(x=s[x], y=s[y], mode="markers", name=_OUTCOME[k], marker=dict(color=colr, size=size, opacity=0.55 if k == 0 else 0.9),
                                   customdata=s["udi"], hovertemplate=f"UDI %{{customdata}}<br>{xl} %{{x}}<br>{yl} %{{y}}<extra></extra>"))
    fig.update_xaxes(title=xl)
    fig.update_yaxes(title=yl)
    return _base(fig, 380, f"{yl} vs {xl} (actual data; failures highlighted)")


def failure_mode_counts(df: pd.DataFrame) -> go.Figure:
    names = {"twf": "TWF tool wear", "hdf": "HDF heat dissipation", "pwf": "PWF power", "osf": "OSF overstrain", "rnf": "RNF random"}
    counts = [int(df[m].sum()) for m in names]
    fig = go.Figure(go.Bar(x=list(names.values()), y=counts, marker_color="#334155", text=counts, textposition="outside"))
    fig.update_yaxes(title="Recorded failures (a row can have several modes)")
    return _base(fig, 300, "Recorded failure modes (actual data)")


def failure_rate_by_type(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("type")["machine_failure"].agg(["mean", "sum", "size"])
    g = g.loc[[t for t in ("L", "M", "H") if t in g.index]]            # only the types present in the selection
    fig = go.Figure(go.Bar(x=g.index, y=g["mean"] * 100, marker_color="#334155", text=[f"{m * 100:.2f}%<br>({int(s)} of {int(n)})" for m, s, n in zip(g["mean"], g["sum"], g["size"])],
                           textposition="outside"))
    fig.update_xaxes(title="Product type")
    fig.update_yaxes(title="Recorded failure rate (%)")
    return _base(fig, 300, "Failure rate by product type (actual data)")


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = ["air_temp_k", "process_temp_k", "rot_speed_rpm", "torque_nm", "tool_wear_min", "delta_t_k", "mechanical_power_w", "wear_torque_min_nm", "machine_failure"]
    nice = ["air temp", "process temp", "speed", "torque", "tool wear", "ΔT", "power", "wear×torque", "failure"]
    c = df[cols].corr().to_numpy()
    fig = go.Figure(go.Heatmap(z=c, x=nice, y=nice, zmin=-1, zmax=1, colorscale="RdBu", reversescale=True, text=np.round(c, 2), texttemplate="%{text}",
                               colorbar=dict(title="Pearson r", thickness=10)))
    fig.update_yaxes(autorange="reversed", automargin=True)
    return _base(fig, 440, "Correlation of inputs, engineered quantities and recorded failure (actual data)")


def probability_by_outcome(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for k, colr in ((0, "#94A3B8"), (1, FAIL)):
        fig.add_trace(go.Histogram(x=df.loc[df["machine_failure"] == k, "failure_probability"], xbins=dict(start=0, end=1, size=0.025), name=f"recorded: {_OUTCOME[k]}",
                                   marker_color=colr, opacity=0.8))
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=threshold, line=dict(color=INK, dash="dash"), annotation_text=f"provisional alert {threshold:.3f}", annotation_position="top right")
    fig.update_yaxes(type="log", title="Records (log scale)")
    fig.update_xaxes(title="Model failure probability (MODEL PREDICTION)")
    return _base(fig, 330, "Model output versus recorded outcome")


def wear_failure_rate(df: pd.DataFrame, width: int = 20) -> go.Figure:
    b = (df["tool_wear_min"] // width * width).astype(int)
    g = df.groupby(b)["machine_failure"].agg(["mean", "size"]).reset_index()
    fig = go.Figure(go.Bar(x=g["tool_wear_min"].astype(str) + "–" + (g["tool_wear_min"] + width).astype(str), y=g["mean"] * 100, marker_color="#334155",
                           customdata=g["size"], hovertemplate="wear %{x} min<br>failure rate %{y:.2f}% (n=%{customdata})<extra></extra>"))
    fig.update_xaxes(title="Tool wear bin (min)")
    fig.update_yaxes(title="Recorded failure rate (%)")
    return _base(fig, 300, "Failure rate by tool wear (actual data)")
