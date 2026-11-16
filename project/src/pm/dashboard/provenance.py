"""Three-way vocabulary used everywhere in the UI and in reports.

ACTUAL        : a recorded row of the (synthetic) AI4I 2020 dataset, shown exactly as stored.
HYPOTHETICAL  : values typed in by the user; nothing recorded, nothing observed.
PREDICTION    : an output of the frozen classifier (uncalibrated RF probability + provisional alert at 0.365).
Derived layers that are NOT predictions: RULE (documented generator rules), ADVISORY (tool-wear heuristic),
ATTRIBUTION (SHAP: model attribution, not causation), CALC (deterministic engineering arithmetic).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    key: str
    label: str
    colour: str
    blurb: str


ACTUAL = Kind("actual", "ACTUAL DATA", "#0F766E", "A recorded row of the synthetic AI4I 2020 dataset, shown as stored.")
HYPOTHETICAL = Kind("hypothetical", "HYPOTHETICAL INPUT", "#7C3AED", "Values entered by you. Nothing was measured or recorded.")
PREDICTION = Kind("prediction", "MODEL PREDICTION", "#1D4ED8",
                  "Output of the frozen Random Forest classifier (uncalibrated probability; provisional alert level 0.365).")
RULE = Kind("rule", "DOCUMENTED RULE", "#475569",
            "Documented rule of the synthetic data generator. Interpretive aid, not a model output and not a real-world limit.")
ADVISORY = Kind("advisory", "TWF ADVISORY", "#B45309",
                "Separate tool-wear maintenance heuristic. Not a failure prediction and not a probability.")
ATTRIBUTION = Kind("attribution", "MODEL ATTRIBUTION", "#0369A1",
                   "SHAP: how the trained model responds to its inputs. Attribution, not causation.")
CALC = Kind("calc", "ENGINEERING CALCULATION", "#334155", "Deterministic arithmetic on the six inputs.")

ALL = {k.key: k for k in (ACTUAL, HYPOTHETICAL, PREDICTION, RULE, ADVISORY, ATTRIBUTION, CALC)}

SOURCE_ACTUAL = "actual"
SOURCE_MANUAL = "hypothetical"


def input_kind(source: str) -> Kind:
    """Kind that labels the six INPUT values of a record (their origin)."""
    return ACTUAL if source == SOURCE_ACTUAL else HYPOTHETICAL
