"""Shared paths and parsing helpers for the data-discovery scripts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

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


def sha256_file(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


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
    """Read the misleadingly named .gzip file as plain CSV chunks."""
    return pd.read_csv(
        path,
        usecols=list(usecols) if usecols is not None else None,
        chunksize=chunk_size,
        compression=None,
        low_memory=False,
    )


def parse_issue_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def parse_term_months(series: pd.Series) -> pd.Series:
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def parse_percent(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype("string").str.strip().str.rstrip("%"), errors="coerce")


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
