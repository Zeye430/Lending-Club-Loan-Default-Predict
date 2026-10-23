"""Profile the full raw dataset without making cleaning decisions."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import pandas as pd

from _common import (
    AUDIT_DIR,
    DEFAULT_CHUNK_SIZE,
    DICTIONARY_FILE,
    RAW_FILE,
    detect_file_format,
    ensure_output_dirs,
    load_dictionary,
    parse_issue_date,
    parse_percent,
    read_raw_chunks,
    write_csv,
    write_json,
)

UNIQUE_CAP = 201


def _count_true(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum())


def profile_raw_data(chunk_size: int) -> None:
    ensure_output_dirs()
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Missing raw dataset: {RAW_FILE}")
    if detect_file_format(RAW_FILE) != "plain_csv":
        raise ValueError(f"Expected Kaggle's plain CSV payload, got {detect_file_format(RAW_FILE)}")

    header = pd.read_csv(RAW_FILE, nrows=0, compression=None)
    columns = list(header.columns)
    dictionary = load_dictionary(DICTIONARY_FILE)
    description_map = dictionary.set_index("dictionary_column")["description"].to_dict()

    row_count = 0
    missing = pd.Series(0, index=columns, dtype="int64")
    dtype_counts = {column: Counter() for column in columns}
    unique_values: dict[str, set[str]] = {column: set() for column in columns}
    numeric_min: dict[str, float | None] = {column: None for column in columns}
    numeric_max: dict[str, float | None] = {column: None for column in columns}
    year_nonnull = pd.DataFrame(columns=columns, dtype="int64")
    year_rows: Counter[int] = Counter()
    status_counts: Counter[str] = Counter()
    id_hash_parts: list[np.ndarray] = []

    quality_counts: Counter[str] = Counter()
    business_columns = [column for column in columns if column != "Unnamed: 0"]

    for chunk_number, chunk in enumerate(read_raw_chunks(chunk_size=chunk_size), start=1):
        row_count += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")

        issue_date = parse_issue_date(chunk["issue_d"])
        years = issue_date.dt.year
        quality_counts["unparsed_issue_d"] += int(years.isna().sum())
        quality_counts["blank_business_rows"] += int(
            chunk[business_columns].isna().all(axis=1).sum()
        )

        for year, count in years.value_counts().items():
            year_rows[int(year)] += int(count)

        valid_years = years.dropna().astype(int)
        if not valid_years.empty:
            presence = chunk.loc[valid_years.index].notna().groupby(valid_years).sum()
            year_nonnull = year_nonnull.add(presence, fill_value=0)

        for status, count in chunk["loan_status"].value_counts(dropna=False).items():
            status_counts[str(status)] += int(count)

        if "id" in chunk:
            ids = chunk["id"].dropna().astype("string")
            if not ids.empty:
                id_hash_parts.append(
                    pd.util.hash_pandas_object(ids, index=False).to_numpy(dtype="uint64")
                )

        loan_amnt = pd.to_numeric(chunk.get("loan_amnt"), errors="coerce")
        funded_amnt = pd.to_numeric(chunk.get("funded_amnt"), errors="coerce")
        fico_low = pd.to_numeric(chunk.get("fico_range_low"), errors="coerce")
        fico_high = pd.to_numeric(chunk.get("fico_range_high"), errors="coerce")
        dti = pd.to_numeric(chunk.get("dti"), errors="coerce")
        annual_inc = pd.to_numeric(chunk.get("annual_inc"), errors="coerce")
        open_acc = pd.to_numeric(chunk.get("open_acc"), errors="coerce")
        revol_util = parse_percent(chunk.get("revol_util"))

        quality_counts["funded_amnt_gt_loan_amnt"] += _count_true(funded_amnt > loan_amnt)
        quality_counts["fico_low_gt_high"] += _count_true(fico_low > fico_high)
        quality_counts["negative_dti"] += _count_true(dti < 0)
        quality_counts["negative_annual_inc"] += _count_true(annual_inc < 0)
        quality_counts["negative_open_acc"] += _count_true(open_acc < 0)
        quality_counts["revol_util_gt_200"] += _count_true(revol_util > 200)

        for column in columns:
            series = chunk[column]
            dtype_counts[column][str(series.dtype)] += len(chunk)

            if len(unique_values[column]) < UNIQUE_CAP:
                remaining = UNIQUE_CAP - len(unique_values[column])
                values = series.dropna().astype(str).unique()[:remaining]
                unique_values[column].update(values.tolist())

            if pd.api.types.is_numeric_dtype(series):
                current_min = series.min(skipna=True)
                current_max = series.max(skipna=True)
                if pd.notna(current_min):
                    old_min = numeric_min[column]
                    old_max = numeric_max[column]
                    numeric_min[column] = (
                        float(current_min) if old_min is None else min(old_min, float(current_min))
                    )
                    numeric_max[column] = (
                        float(current_max) if old_max is None else max(old_max, float(current_max))
                    )

        if chunk_number % 5 == 0:
            print(f"Profiled {row_count:,} rows", flush=True)

    duplicate_id_hashes = 0
    unique_id_count = 0
    if id_hash_parts:
        id_hashes = np.concatenate(id_hash_parts)
        id_hashes.sort()
        duplicate_id_hashes = int(np.count_nonzero(id_hashes[1:] == id_hashes[:-1]))
        unique_id_count = int(len(id_hashes) - duplicate_id_hashes)

    profile_rows = []
    for position, column in enumerate(columns, start=1):
        observed = unique_values[column]
        capped = len(observed) >= UNIQUE_CAP
        year_counts = year_nonnull.get(column, pd.Series(dtype="float64"))
        active_years = sorted(int(year) for year, count in year_counts.items() if count > 0)
        primary_dtype = (
            dtype_counts[column].most_common(1)[0][0] if dtype_counts[column] else "unknown"
        )
        profile_rows.append(
            {
                "position": position,
                "column": column,
                "description": description_map.get(column),
                "dictionary_status": (
                    "documented" if column in description_map else "undocumented"
                ),
                "primary_dtype": primary_dtype,
                "observed_dtypes": "; ".join(sorted(dtype_counts[column])),
                "missing_count": int(missing[column]),
                "missing_rate": float(missing[column] / row_count),
                "unique_count": f">={UNIQUE_CAP}" if capped else str(len(observed)),
                "sample_values": " | ".join(sorted(observed)[:5]),
                "numeric_min": numeric_min[column],
                "numeric_max": numeric_max[column],
                "first_year_nonnull": active_years[0] if active_years else None,
                "last_year_nonnull": active_years[-1] if active_years else None,
                "is_all_missing": len(observed) == 0,
                "is_constant_or_all_missing": len(observed) <= 1,
            }
        )

    profile = pd.DataFrame(profile_rows)
    write_csv(profile, AUDIT_DIR / "raw_column_profile.csv")

    year_nonnull = year_nonnull.sort_index().fillna(0).astype("int64")
    year_nonnull_rate = year_nonnull.copy().astype("float64")
    for year in year_nonnull_rate.index:
        year_nonnull_rate.loc[year] = year_nonnull.loc[year] / year_rows[int(year)]
    year_nonnull_rate.insert(
        0, "row_count", [year_rows[int(year)] for year in year_nonnull_rate.index]
    )
    year_nonnull_rate.index.name = "issue_year"
    write_csv(
        year_nonnull_rate.reset_index(),
        AUDIT_DIR / "raw_nonnull_rate_by_year.csv",
    )

    rows_by_year = pd.DataFrame(sorted(year_rows.items()), columns=["issue_year", "row_count"])
    write_csv(rows_by_year, AUDIT_DIR / "raw_rows_by_year.csv")

    loan_status = pd.DataFrame(status_counts.most_common(), columns=["loan_status", "row_count"])
    loan_status["row_share"] = loan_status["row_count"] / row_count
    write_csv(loan_status, AUDIT_DIR / "raw_loan_status_counts.csv")

    raw_set = set(columns)
    dictionary_set = set(dictionary["dictionary_column"])
    mismatch_rows = [
        {"mismatch_type": "raw_not_dictionary", "column": column}
        for column in sorted(raw_set - dictionary_set)
    ] + [
        {"mismatch_type": "dictionary_not_raw", "column": column}
        for column in sorted(dictionary_set - raw_set)
    ]
    write_csv(
        pd.DataFrame(mismatch_rows, columns=["mismatch_type", "column"]),
        AUDIT_DIR / "dictionary_mismatches.csv",
    )

    checks = [
        {
            "check": "row_count",
            "value": row_count,
            "severity": "info",
            "interpretation": "Total rows parsed from the raw CSV.",
        },
        {
            "check": "column_count",
            "value": len(columns),
            "severity": "info",
            "interpretation": "Columns parsed from the raw CSV header.",
        },
        {
            "check": "unique_id_count",
            "value": unique_id_count,
            "severity": "info",
            "interpretation": "Unique non-null loan IDs, measured with 64-bit hashes.",
        },
        {
            "check": "duplicate_id_hash_count",
            "value": duplicate_id_hashes,
            "severity": "warning" if duplicate_id_hashes else "pass",
            "interpretation": "Duplicate loan-ID hashes; hash collision risk is negligible.",
        },
    ]
    for check, value in quality_counts.items():
        severity = "pass" if value == 0 else "warning"
        if check in {"unparsed_issue_d", "blank_business_rows"} and value <= 1:
            severity = "review"
        checks.append(
            {
                "check": check,
                "value": int(value),
                "severity": severity,
                "interpretation": "Observed count; review before defining cleaning rules.",
            }
        )
    write_csv(pd.DataFrame(checks), AUDIT_DIR / "data_quality_checks.csv")

    summary = {
        "rows": row_count,
        "columns": len(columns),
        "first_issue_year": min(year_rows),
        "last_issue_year": max(year_rows),
        "columns_with_missing_values": int((missing > 0).sum()),
        "columns_over_30pct_missing": int(((missing / row_count) > 0.30).sum()),
        "constant_or_all_missing_columns": profile.loc[
            profile["is_constant_or_all_missing"], "column"
        ].tolist(),
        "raw_columns_not_in_dictionary": sorted(raw_set - dictionary_set),
        "dictionary_columns_not_in_raw": sorted(dictionary_set - raw_set),
        "duplicate_id_hash_count": duplicate_id_hashes,
        "quality_counts": dict(quality_counts),
    }
    write_json(summary, AUDIT_DIR / "raw_audit_summary.json")

    print("Raw-data audit complete:")
    print(f"  shape       : {row_count:,} rows x {len(columns)} columns")
    print(f"  issue years : {min(year_rows)}-{max(year_rows)}")
    print(f"  duplicate IDs: {duplicate_id_hashes:,}")
    print(f"  output      : {AUDIT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    profile_raw_data(chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
