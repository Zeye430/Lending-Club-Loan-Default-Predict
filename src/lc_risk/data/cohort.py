"""Build the modeling cohort from the raw Lending Club export.

Streams the 1.7 GB raw CSV, keeps resolved 36-month individual loans in the configured
issue years, parses and cleans the input fields, validates them against the spec,
assigns year-based splits, and writes one parquet file plus a JSON manifest.

    uv run python -m lc_risk.data.cohort
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lc_risk import config, schema
from lc_risk.data.io import read_raw_chunks, sha256_file
from lc_risk.data.parsing import parse_emp_length, parse_month, parse_percent, parse_term_months
from lc_risk.data.validation import validate_inputs
from lc_risk.provenance import git_revision, package_versions

log = logging.getLogger(__name__)

YEAR_TO_SPLIT = {year: split for split, years in config.SPLIT_YEARS.items() for year in years}
LABELS = {
    **dict.fromkeys(config.NEGATIVE_STATUSES, 0),
    **dict.fromkeys(config.POSITIVE_STATUSES, 1),
}

INPUT_COLUMNS = [f.name for f in schema.ALL_INPUT_FIELDS]
RAW_COLUMNS = [
    "id", "issue_d", "term", "application_type",
    *INPUT_COLUMNS, *schema.PRICING_COLS, *schema.OUTCOME_COLS,
]  # fmt: skip
PERCENT_COLUMNS = ("revol_util", "int_rate")
MONTH_COLUMNS = ("earliest_cr_line", "last_pymnt_d")

# Category values merged before validation, with the reason recorded in code.
CATEGORY_MERGES = {
    "home_ownership": {"ANY": "OTHER", "NONE": "OTHER"},  # each <0.02% of loans
    "purpose": {"educational": "other"},  # a single loan in 2013-2016
}

# Values that are impossible or are sentinels in the source system become NaN.
CleaningRule = Callable[[pd.DataFrame], pd.Series]
CLEANING_RULES: dict[str, tuple[str, CleaningRule]] = {
    "dti_negative": ("dti", lambda df: df["dti"] < 0),
    "dti_sentinel_999": ("dti", lambda df: df["dti"] >= 999),
    "annual_inc_non_positive": ("annual_inc", lambda df: df["annual_inc"] <= 0),
    "revol_util_above_200": ("revol_util", lambda df: df["revol_util"] > 200),
}


def _select(chunk: pd.DataFrame, exclusions: Counter[str]) -> pd.DataFrame:
    """Apply cohort filters, counting why rows are dropped (in order of precedence)."""
    issue = parse_month(chunk["issue_d"])
    term = parse_term_months(chunk["term"])
    status = chunk["loan_status"].astype("string")

    in_term = term.eq(config.TERM_MONTHS).fillna(False).to_numpy()
    in_years = issue.dt.year.isin(YEAR_TO_SPLIT).to_numpy()
    individual = chunk["application_type"].eq("Individual").to_numpy()
    resolved = status.isin(LABELS).fillna(False).to_numpy()

    exclusions["term_not_36_months"] += int((~in_term).sum())
    exclusions["issue_year_out_of_scope"] += int((in_term & ~in_years).sum())
    exclusions["joint_application"] += int((in_term & in_years & ~individual).sum())
    exclusions["unresolved_status"] += int((in_term & in_years & individual & ~resolved).sum())

    keep = in_term & in_years & individual & resolved
    selected = chunk.loc[keep].copy()
    selected["issue_d"] = issue[keep]
    return selected


def _parse(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.drop(columns=["term", "application_type"]).rename(columns={"id": schema.ID_COL})
    out[schema.ID_COL] = out[schema.ID_COL].astype("string")
    for column in PERCENT_COLUMNS:
        out[column] = parse_percent(out[column])
    for column in MONTH_COLUMNS:
        out[column] = parse_month(out[column])
    out["emp_length"] = parse_emp_length(out["emp_length"])

    for field in schema.ALL_INPUT_FIELDS:
        if field.kind == "numeric":
            out[field.name] = pd.to_numeric(out[field.name], errors="coerce").astype("float64")
        elif field.kind == "categorical":
            out[field.name] = out[field.name].astype("string").str.strip()
    for column, mapping in CATEGORY_MERGES.items():
        out[column] = out[column].replace(mapping)
    return out


def _clean(frame: pd.DataFrame) -> dict[str, int]:
    """Null out impossible/sentinel values in place; return how many each rule touched."""
    counts = {}
    for rule, (column, condition) in CLEANING_RULES.items():
        mask = condition(frame).fillna(False)
        counts[rule] = int(mask.sum())
        frame.loc[mask, column] = np.nan
    return counts


def build_cohort(
    raw_path: Path = config.RAW_FILE, chunk_size: int = 200_000
) -> tuple[pd.DataFrame, dict]:
    """Return the cohort frame and a summary of how it was derived."""
    exclusions: Counter[str] = Counter()
    rows_scanned = 0
    parts = []
    for chunk in read_raw_chunks(raw_path, usecols=RAW_COLUMNS, chunk_size=chunk_size):
        rows_scanned += len(chunk)
        parts.append(_parse(_select(chunk, exclusions)))
        log.info("scanned %s rows, kept %s", f"{rows_scanned:,}", f"{sum(map(len, parts)):,}")

    cohort = pd.concat(parts, ignore_index=True)
    cleaning = _clean(cohort)

    violations = validate_inputs(cohort, schema.ALL_INPUT_FIELDS)
    if violations:
        raise ValueError("cohort violates the input spec:\n" + "\n".join(map(str, violations)))
    if cohort[schema.ID_COL].duplicated().any():
        raise ValueError("duplicate loan ids in cohort")

    cohort[config.TARGET] = cohort["loan_status"].map(LABELS).astype("int8")
    cohort["split"] = cohort[schema.DATE_COL].dt.year.map(YEAR_TO_SPLIT).astype("string")
    cohort = cohort.sort_values([schema.DATE_COL, schema.ID_COL], ignore_index=True)

    first = [schema.ID_COL, schema.DATE_COL, "split", config.TARGET]
    cohort = cohort[first + [c for c in cohort.columns if c not in first]]

    summary = {
        "rows_scanned": rows_scanned,
        "rows_kept": len(cohort),
        "exclusions": dict(exclusions),
        "cleaning_rules_applied": cleaning,
        "category_merges": CATEGORY_MERGES,
        "splits": _split_summary(cohort),
    }
    return cohort, summary


def _split_summary(cohort: pd.DataFrame) -> dict[str, dict]:
    summary = {}
    for split, years in config.SPLIT_YEARS.items():
        part = cohort.loc[cohort["split"] == split, config.TARGET]
        summary[split] = {
            "issue_years": list(years),
            "rows": len(part),
            "defaults": int(part.sum()),
            "default_rate": round(float(part.mean()), 5),
        }
    return summary


def write_cohort(cohort: pd.DataFrame, summary: dict, raw_path: Path, out_file: Path) -> dict:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(out_file, index=False)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": git_revision(),
        "libraries": package_versions("pandas", "pyarrow", "numpy"),
        "definition": {
            "term_months": config.TERM_MONTHS,
            "application_type": "Individual",
            "label_negative": sorted(config.NEGATIVE_STATUSES),
            "label_positive": sorted(config.POSITIVE_STATUSES),
            "split_years": config.SPLIT_YEARS,
        },
        "raw_file": {
            "name": raw_path.name,
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "cohort_file": {
            "name": out_file.name,
            "rows": len(cohort),
            "columns": len(cohort.columns),
            "sha256": sha256_file(out_file),
        },
        **summary,
    }
    manifest_path = out_file.with_name(out_file.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_cohort(split: str | None = None, path: Path = config.COHORT_FILE) -> pd.DataFrame:
    filters = [("split", "==", split)] if split else None
    return pd.read_parquet(path, filters=filters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw", type=Path, default=config.RAW_FILE)
    parser.add_argument("--out", type=Path, default=config.COHORT_FILE)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cohort, summary = build_cohort(args.raw, args.chunk_size)
    manifest = write_cohort(cohort, summary, args.raw, args.out)
    log.info("wrote %s (%s rows)", args.out, f"{len(cohort):,}")
    for split, stats in manifest["splits"].items():
        log.info(
            "  %-10s years=%s rows=%s default_rate=%.2f%%",
            split, stats["issue_years"], f"{stats['rows']:,}", 100 * stats["default_rate"],
        )  # fmt: skip


if __name__ == "__main__":
    main()
