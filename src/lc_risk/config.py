"""Project-wide paths and the cohort/split definition.

The cohort follows the data-discovery audit in ``scripts/``: the 2020Q3 snapshot leaves
later and 60-month vintages right-censored, so modeling uses 36-month loans issued
2012-2016, split by whole issue years.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "Loan_status_2007-2020Q3.gzip"
INTERIM_DIR = DATA_DIR / "interim"
ARTIFACTS_DIR = ROOT / "artifacts"

TARGET = "default"

TERM_MONTHS = 36
NEGATIVE_STATUSES = frozenset({"Fully Paid"})
POSITIVE_STATUSES = frozenset({"Charged Off", "Default"})

SPLIT_YEARS: dict[str, tuple[int, ...]] = {
    "train": (2012, 2013, 2014),
    "validation": (2015,),
    "test": (2016,),
}
