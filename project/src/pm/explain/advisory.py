"""Separate tool-wear (TWF) advisory. Never a classifier prediction, never a probability.

Boundaries (approved): wear < 200 min -> none; 200 <= wear < 220 -> advisory; wear >= 220 -> elevated.
The constants are the Phase 4 frozen ones (`pm.final.spec`); the module refuses to import if they drift.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from pm.explain import notices
from pm.final import spec

ADVISORY_MIN = 200.0
ELEVATED_MIN = 220.0
WINDOW_MAX = 240.0           # documented upper end of the generator's tool-wear window (informational)
assert spec.TWF_ADVISORY_WEAR_MIN == ADVISORY_MIN and spec.TWF_ELEVATED_WEAR_MIN == ELEVATED_MIN, "advisory bands drifted from the frozen spec"

_MESSAGES = {
    "none": "Tool wear is below 200 min, i.e. below the documented 200-240 min tool-wear window. No tool-wear advisory.",
    "advisory": "Tool wear is 200-219 min: inside the documented 200-240 min tool-wear window. Consider planning a tool change.",
    "elevated": "Tool wear is 220 min or more: in the upper half of (or beyond) the documented 200-240 min tool-wear window. "
                "Consider replacing the tool at the next opportunity.",
}


@dataclass(frozen=True)
class ToolWearAdvisory:
    level: str                  # "none" | "advisory" | "elevated"
    tool_wear_min: float
    message: str
    kind: str = "tool_wear_advisory"
    is_classifier_prediction: bool = False
    has_probability: bool = False
    notice: str = notices.TWF_ADVISORY

    def to_dict(self) -> dict:
        return asdict(self)


def advisory_level(wear_min: float) -> str:
    w = float(wear_min)
    if not math.isfinite(w) or w < 0:
        raise ValueError(f"tool wear must be a finite, non-negative number of minutes, got {wear_min!r}")
    if w >= ELEVATED_MIN:
        return "elevated"
    if w >= ADVISORY_MIN:
        return "advisory"
    return "none"


def tool_wear_advisory(wear_min: float) -> ToolWearAdvisory:
    lvl = advisory_level(wear_min)
    return ToolWearAdvisory(level=lvl, tool_wear_min=float(wear_min), message=_MESSAGES[lvl])
