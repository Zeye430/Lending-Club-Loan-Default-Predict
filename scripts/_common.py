"""Shared paths and parsing helpers for the data-discovery scripts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

# Parsing and hashing live in the package so discovery and the cohort build share them.
from lc_risk.data.io import read_raw_chunks as _read_raw_chunks
from lc_risk.data.io import sha256_file
from lc_risk.data.parsing import parse_month as parse_issue_date
from lc_risk.data.parsing import parse_percent, parse_term_months

# Re-exported for the discovery scripts.
__all__ = ["parse_issue_date", "parse_percent", "parse_term_months", "sha256_file"]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "Loan_status_2007-2020Q3.gzip"
DICTIONARY_FILE = RAW_DIR / "LCDataDictionary.xlsx"
REPORT_ROOT = ROOT / "reports" / "data_discovery"
AUDIT_DIR = REPORT_ROOT / "audit"
COHORT_DIR = REPORT_ROOT / "cohort"
EDA_DIR = REPORT_ROOT / "eda"

DEFAULT_CHUNK_SIZE = 100_000


def ensure_output_dirs() -> None:
    for path in (RAW_DIR, AUDIT_DIR, COHORT_DIR, EDA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def detect_file_format(path: Path) -> str:
    with path.open("rb") as handle:
        prefix = handle.read(4)
    if prefix.startswith(b"\x1f\x8b"):
        return "gzip"
    if prefix.startswith(b"PK\x03\x04"):
        return "zip"
    return "plain_csv"


def read_raw_chunks(
    path: Path = RAW_FILE,
    *,
    usecols: Iterable[str] | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    return _read_raw_chunks(path, usecols=usecols, chunk_size=chunk_size)


def load_dictionary(path: Path = DICTIONARY_FILE) -> pd.DataFrame:
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name="LoanStats")
    else:
        frame = pd.read_csv(path)

    frame = frame.iloc[:, :2].copy()
    frame.columns = ["dictionary_column", "description"]
    frame = frame.dropna(subset=["dictionary_column"])
    frame["dictionary_column"] = frame["dictionary_column"].astype(str).str.strip()

    # The supplied dictionary uses "verified_status_joint" while the data uses
    # "verification_status_joint". Preserve the source row and add the alias.
    if "verified_status_joint" in set(frame["dictionary_column"]):
        alias = frame.loc[frame["dictionary_column"] == "verified_status_joint"].copy()
        alias["dictionary_column"] = "verification_status_joint"
        frame = pd.concat([frame, alias], ignore_index=True)

    return frame.drop_duplicates("dictionary_column", keep="first")


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp_path, index=index)
    temp_path.replace(path)
