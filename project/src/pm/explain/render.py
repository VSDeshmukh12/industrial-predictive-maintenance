"""Plain-text rendering of one explanation. Wording is deliberately non-causal and non-predictive."""
from __future__ import annotations

from pm.explain import notices

_CONSISTENCY = {
    "rule_and_alert": "The documented rule and the model alert agree.",
    "rule_only": "A documented rule is met but the model output is below the alert level (the model does not always match the rules).",
    "alert_only": "The model output is at or above the alert level although no documented rule is met.",
    "neither": "Neither a documented rule nor the model alert is triggered.",
}


def _fmt(f: dict) -> str:
    v = f["value"] if isinstance(f["value"], str) else (f"{f['value']:,.0f}" if abs(f["value"]) >= 1000 else f"{f['value']:.4g}")
    return f"{f['label']} = {v} {f['unit']}".strip()


def render_text(x: dict, top_k: int = 4) -> str:
    m, a, c, adv = x["model"], x["attribution"], x["conditions"], x["tool_wear_advisory"]
    e = x["engineering"]
    L = ["MODEL OUTPUT (classifier)",
         f"  Estimated failure probability {m['failure_probability']:.3f}; provisional alert level {m['threshold']:.3f} -> "
         f"{'ALERT' if m['alert'] else 'no alert'}.",
         "", "ENGINEERING CALCULATIONS",
         f"  ΔT = {e['delta_t_k']['value']:.2f} K;  mechanical power = {e['mechanical_power_w']['value']:.0f} W;  "
         f"wear × torque = {e['wear_torque_min_nm']['value']:.0f} min·Nm",
         "", "MODEL ATTRIBUTION (SHAP; not causation)",
         f"  Average model output {a['base_value']:.3f}. Largest contributions to this row's output:"]
    for f in a["top"][:top_k]:
        L.append(f"    {f['shap']:+.3f}  {_fmt(f)}")
    L += [f"  Driver groups: " + ", ".join(f"{g} {v:+.3f}" for g, v in a["groups"].items()),
          "", "MACHINE-CONDITION INTERPRETATION (documented synthetic-generator rules)", "  " + c["summary_text"]]
    for r in c["results"]:
        L.append(f"    {r['name']}: {r['status'].replace('_', ' ')} ({r['evidence']}; rule: {r['rule']})")
    L += ["  " + _CONSISTENCY[x["model_vs_rules"]],
          "", "TOOL-WEAR ADVISORY (separate; not a classifier prediction)", f"  {adv['level'].upper()}: {adv['message']}",
          "", "NOTICES"]
    L += [f"  - {notices.ALL[k]}" for k in ("synthetic_data", "attribution_not_causation", "threshold_provisional", "scope")]
    return "\n".join(L)
