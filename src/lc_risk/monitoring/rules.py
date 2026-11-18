"""Alert rules for the monitoring layers.

Thresholds follow common credit-risk practice and were not tuned on the monitored period:
PSI 0.10 / 0.25 is the industry convention; an AUC drop of 0.01 / 0.02 is roughly one and two
times the quarter-to-quarter spread of early AUC in the 2015 reference year; actual-to-expected
ratios outside +/-10% / +/-20% are the usual calibration triggers.
See docs/monitoring.md for what each alert means and what to do about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["warning", "critical"]
SEVERITY_ORDER = {"warning": 1, "critical": 2}


@dataclass(frozen=True)
class Rule:
    metric: str
    layer: str
    warning: float
    critical: float
    direction: Literal["above", "below", "outside"]
    description: str

    def check(self, value: float) -> Severity | None:
        if value is None or value != value:  # missing or NaN: not observable yet
            return None
        if self.direction == "above":
            breach = value
        elif self.direction == "below":
            breach = -value
        else:  # distance from 1.0, for actual-to-expected ratios
            breach = abs(value - 1.0)
        if breach >= self.critical:
            return "critical"
        if breach >= self.warning:
            return "warning"
        return None


RULES = (
    # The nullability contract tolerates up to 0.5% missing per required field in
    # training, so a rejection share up to that level is expected.
    Rule("rejected_share", "data quality", 0.005, 0.02, "above",
         "Share of applications the API would reject (types, ranges, categories, required fields)."),
    Rule("max_input_psi", "input drift", 0.10, 0.25, "above",
         "Largest per-field PSI against the reference year."),
    Rule("score_psi", "score drift", 0.10, 0.25, "above",
         "PSI of predicted PD against the reference year."),
    Rule("early_ae", "early performance", 0.10, 0.20, "outside",
         "Defaults within 12 months on book, actual / expected."),
    Rule("early_auc_drop", "early performance", 0.01, 0.02, "above",
         "Reference early AUC minus current early AUC."),
    Rule("final_ae", "final performance", 0.10, 0.20, "outside",
         "Lifetime defaults, actual / expected (calibration)."),
    Rule("final_auc_drop", "final performance", 0.01, 0.02, "above",
         "Reference AUC minus current AUC on lifetime outcomes."),
)  # fmt: skip
RULES_BY_METRIC = {rule.metric: rule for rule in RULES}
