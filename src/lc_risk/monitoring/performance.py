"""Outcome labels as they become observable, and actual-to-expected ratios.

A 36-month loan's final outcome is known only ~40 months after issue (term plus the ~4
months from first missed payment to charge-off). Monitoring cannot wait that long, so it
also tracks an early label: did the loan default with its last payment inside the first
``EARLY_HORIZON_MONTHS`` months on book? That label is observable about a year after issue.

The default month is approximated by the last payment date (``last_pymnt_d``): payments
stop when the borrower defaults. Recoveries after charge-off can occasionally move that date
later; in the 2015-2016 cohort, 0.8% of defaults show a last payment beyond month 36.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lc_risk import config, schema

EARLY_HORIZON_MONTHS = 12
# Months after issue at which each label can be read for a whole vintage.
EARLY_LABEL_AVAILABLE_MONTHS = EARLY_HORIZON_MONTHS + 1  # one month to observe the miss
FINAL_LABEL_AVAILABLE_MONTHS = config.TERM_MONTHS + 4  # term + charge-off lag


def months_on_book_at_last_payment(frame: pd.DataFrame) -> pd.Series:
    issued, last = frame[schema.DATE_COL], frame["last_pymnt_d"]
    months = (last.dt.year - issued.dt.year) * 12 + (last.dt.month - issued.dt.month)
    return months.fillna(0).clip(lower=0)  # no payment ever recorded -> defaulted at once


def early_default(frame: pd.DataFrame, horizon: int = EARLY_HORIZON_MONTHS) -> pd.Series:
    """1 if the loan defaulted with its last payment before ``horizon`` months on book."""
    defaulted = frame[config.TARGET] == 1
    return (defaulted & (months_on_book_at_last_payment(frame) < horizon)).astype("int8")


def actual_to_expected(actual: np.ndarray, expected: np.ndarray) -> float:
    """Sum of observed defaults over the sum the model expected; 1.0 means on target."""
    total = float(np.sum(expected))
    return float(np.sum(actual)) / total if total > 0 else float("nan")
