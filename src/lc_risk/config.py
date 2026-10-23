"""Project-wide paths and the cohort/split definition.

The cohort follows the data-discovery audit in ``scripts/``: the 2020Q3 snapshot leaves
later and 60-month vintages right-censored, so modeling uses 36-month loans issued
2013-2016, split by whole issue years.

2012 is excluded even though its labels are mature. LC rolled out ~30 bureau fields
during 2012, so they are ~52% missing that year and fully populated from 2013 on.
Training on 2012 would teach the model a schema-driven missingness pattern that never
occurs at scoring time.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_FILE = DATA_DIR / "raw" / "Loan_status_2007-2020Q3.gzip"
INTERIM_DIR = DATA_DIR / "interim"
COHORT_FILE = INTERIM_DIR / "cohort.parquet"
COHORT_MANIFEST = INTERIM_DIR / "cohort_manifest.json"
ARTIFACTS_DIR = ROOT / "artifacts"

TARGET = "default"

TERM_MONTHS = 36
NEGATIVE_STATUSES = frozenset({"Fully Paid"})
POSITIVE_STATUSES = frozenset({"Charged Off", "Default"})

SPLIT_YEARS: dict[str, tuple[int, ...]] = {
    "train": (2013, 2014),
    "validation": (2015,),
    "test": (2016,),
}
