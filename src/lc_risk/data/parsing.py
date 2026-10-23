"""Parsers for the raw Lending Club export's string encodings."""

from __future__ import annotations

import pandas as pd

EMP_LENGTH_YEARS = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4, "5 years": 5,
    "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9, "10+ years": 10,
}  # fmt: skip


def parse_month(series: pd.Series) -> pd.Series:
    """'Dec-2015' -> Timestamp('2015-12-01'); unparseable -> NaT."""
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def parse_term_months(series: pd.Series) -> pd.Series:
    """' 36 months' -> 36."""
    extracted = series.astype("string").str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def parse_percent(series: pd.Series) -> pd.Series:
    """'13.3%' -> 13.3. Already-numeric input passes through."""
    stripped = series.astype("string").str.strip().str.rstrip("%")
    return pd.to_numeric(stripped, errors="coerce").astype("float64")


def parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' -> 10.0, '< 1 year' -> 0.0; missing or 'n/a' -> NaN."""
    return series.astype("string").str.strip().map(EMP_LENGTH_YEARS).astype("float64")
