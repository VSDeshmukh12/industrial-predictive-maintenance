"""Mandatory wording. Every explanation object carries all of these; tests check them."""

SYNTHETIC = ("AI4I 2020 is a SYNTHETIC dataset (UCI repository; Matzka 2020), not measurements from a real plant. "
             "The model recovers the data generator's rules; it has not been validated on real machines.")
ATTRIBUTION_NOT_CAUSATION = (
    "SHAP values are MODEL ATTRIBUTION: they split the model's output for this row among the model's input "
    "features, relative to the model's average output. They describe how this trained model responds to its "
    "inputs. They are not physical causes: they do not say what would happen to a real machine if a variable "
    "were changed, and correlated inputs (for example delta_t with the two temperatures, or power with speed "
    "and torque) share credit.")
THRESHOLD_PROVISIONAL = ("The alert level 0.365 is a PROVISIONAL threshold (F2 objective, chosen from training-set "
                         "cross-validation in Phase 3 and unchanged since). The alert is a screening flag, not a diagnosis.")
SCOPE = ("This is an offline evaluation of single operating-point records. It is not real-time prediction, it does "
         "not estimate remaining useful life (RUL), and it has no real-world industrial validation.")
TWF_ADVISORY = ("The tool-wear advisory is SEPARATE from the classifier and is NOT a classifier prediction or a "
                "probability. It reports only where tool wear lies relative to the documented 200-240 min "
                "tool-wear window of the synthetic generator. The classifier cannot reliably detect tool-wear (TWF) failures.")
RULES = ("Condition rules are the documented failure rules of the synthetic AI4I 2020 generator (UCI documentation). "
         "They are interpretive aids, not model inputs, and not real-world engineering limits. A rule match is not "
         "a model output; no rule match does not rule out failure (tool-wear and random failures follow no rule).")

ALL = {"synthetic_data": SYNTHETIC, "attribution_not_causation": ATTRIBUTION_NOT_CAUSATION,
       "threshold_provisional": THRESHOLD_PROVISIONAL, "scope": SCOPE, "tool_wear_advisory": TWF_ADVISORY,
       "condition_rules": RULES}
