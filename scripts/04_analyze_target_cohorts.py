"""Audit loan-status maturity and propose an observation-safe modeling cohort."""

from __future__ import annotations

import argparse
from collections import Counter

import pandas as pd

from _common import (
    COHORT_DIR,
    DEFAULT_CHUNK_SIZE,
    ensure_output_dirs,
    parse_issue_date,
    parse_term_months,
    read_raw_chunks,
    write_csv,
    write_json,
)

NEGATIVE_STATUSES = {"Fully Paid"}
POSITIVE_STATUSES = {"Charged Off", "Default"}
STRICT_FINAL_STATUSES = {"Fully Paid", "Charged Off"}
POLICY_FAILURE_STATUSES = {
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off",
}

CANDIDATE_TERM = 36
CANDIDATE_START_YEAR = 2012
CANDIDATE_END_YEAR = 2016


def analyze_cohorts(chunk_size: int) -> None:
    ensure_output_dirs()
    yearly_status: Counter[tuple[int, int, str]] = Counter()
    monthly_status: Counter[tuple[str, int, str]] = Counter()
    candidate_status: Counter[str] = Counter()
    candidate_by_year: Counter[tuple[int, str]] = Counter()

    columns = ["issue_d", "term", "loan_status"]
    rows_seen = 0

    for chunk_number, chunk in enumerate(
        read_raw_chunks(usecols=columns, chunk_size=chunk_size), start=1
    ):
        rows_seen += len(chunk)
        issue_date = parse_issue_date(chunk["issue_d"])
        term = parse_term_months(chunk["term"])
        status = chunk["loan_status"].astype("string")

        frame = pd.DataFrame(
            {
                "issue_date": issue_date,
                "issue_year": issue_date.dt.year.astype("Int64"),
                "issue_month": issue_date.dt.to_period("M").astype("string"),
                "term_months": term,
                "loan_status": status,
            }
        ).dropna(subset=["issue_year", "term_months", "loan_status"])

        for key, count in frame.value_counts(
            ["issue_year", "term_months", "loan_status"], dropna=False
        ).items():
            yearly_status[(int(key[0]), int(key[1]), str(key[2]))] += int(count)

        for key, count in frame.value_counts(
            ["issue_month", "term_months", "loan_status"], dropna=False
        ).items():
            monthly_status[(str(key[0]), int(key[1]), str(key[2]))] += int(count)

        candidate = frame[
            (frame["term_months"] == CANDIDATE_TERM)
            & frame["issue_year"].between(CANDIDATE_START_YEAR, CANDIDATE_END_YEAR)
        ]
        for value, count in candidate["loan_status"].value_counts().items():
            candidate_status[str(value)] += int(count)
        for key, count in candidate.value_counts(["issue_year", "loan_status"]).items():
            candidate_by_year[(int(key[0]), str(key[1]))] += int(count)

        if chunk_number % 10 == 0:
            print(f"Scanned {rows_seen:,} rows", flush=True)

    status_by_year_term = pd.DataFrame(
        [
            {
                "issue_year": year,
                "term_months": term,
                "loan_status": status,
                "row_count": count,
            }
            for (year, term, status), count in sorted(yearly_status.items())
        ]
    )
    write_csv(
        status_by_year_term,
        COHORT_DIR / "loan_status_by_year_and_term.csv",
    )

    status_by_month_term = pd.DataFrame(
        [
            {
                "issue_month": month,
                "term_months": term,
                "loan_status": status,
                "row_count": count,
            }
            for (month, term, status), count in sorted(monthly_status.items())
        ]
    )
    write_csv(
        status_by_month_term,
        COHORT_DIR / "loan_status_by_month_and_term.csv",
    )

    grouped = status_by_year_term.groupby(["issue_year", "term_months"])
    maturity_rows = []
    for (year, term), group in grouped:
        counts = group.set_index("loan_status")["row_count"]
        all_rows = int(counts.sum())
        strict_final = int(counts.reindex(sorted(STRICT_FINAL_STATUSES), fill_value=0).sum())
        policy_failure = int(counts.reindex(sorted(POLICY_FAILURE_STATUSES), fill_value=0).sum())
        current = int(counts.get("Current", 0))
        charged_off = int(counts.get("Charged Off", 0))
        default = int(counts.get("Default", 0))
        fully_paid = int(counts.get("Fully Paid", 0))
        maturity_rows.append(
            {
                "issue_year": int(year),
                "term_months": int(term),
                "all_rows": all_rows,
                "strict_final_rows": strict_final,
                "strict_final_share": strict_final / all_rows,
                "policy_failure_rows": policy_failure,
                "current_rows": current,
                "fully_paid_rows": fully_paid,
                "charged_off_rows": charged_off,
                "default_rows": default,
                "strict_charge_off_rate": (charged_off / strict_final if strict_final else None),
            }
        )
    maturity = pd.DataFrame(maturity_rows).sort_values(["issue_year", "term_months"])
    write_csv(maturity, COHORT_DIR / "cohort_maturity_by_year_and_term.csv")

    candidate_status_df = pd.DataFrame(
        candidate_status.most_common(), columns=["loan_status", "row_count"]
    )
    candidate_total = int(candidate_status_df["row_count"].sum())
    candidate_status_df["row_share"] = candidate_status_df["row_count"] / candidate_total
    write_csv(candidate_status_df, COHORT_DIR / "candidate_cohort_status.csv")

    candidate_year_df = pd.DataFrame(
        [
            {"issue_year": year, "loan_status": status, "row_count": count}
            for (year, status), count in sorted(candidate_by_year.items())
        ]
    )
    write_csv(candidate_year_df, COHORT_DIR / "candidate_status_by_year.csv")

    modelable = sum(
        count
        for status, count in candidate_status.items()
        if status in NEGATIVE_STATUSES | POSITIVE_STATUSES
    )
    positives = sum(
        count for status, count in candidate_status.items() if status in POSITIVE_STATUSES
    )
    strict_modelable = sum(
        count for status, count in candidate_status.items() if status in STRICT_FINAL_STATUSES
    )

    split_rows = []
    for year in range(CANDIDATE_START_YEAR, CANDIDATE_END_YEAR + 1):
        split_name = "train" if year <= 2014 else "validation" if year == 2015 else "test"
        year_counts = {
            status: candidate_by_year.get((year, status), 0)
            for status in NEGATIVE_STATUSES | POSITIVE_STATUSES
        }
        year_total = int(sum(year_counts.values()))
        year_positive = int(sum(year_counts.get(status, 0) for status in POSITIVE_STATUSES))
        split_rows.append(
            {
                "split": split_name,
                "issue_year": year,
                "modelable_rows": year_total,
                "positive_rows": year_positive,
                "default_rate": year_positive / year_total if year_total else None,
            }
        )
    write_csv(pd.DataFrame(split_rows), COHORT_DIR / "proposed_time_split.csv")

    summary = {
        "rows_scanned": rows_seen,
        "candidate_definition": {
            "term_months": CANDIDATE_TERM,
            "issue_year_start": CANDIDATE_START_YEAR,
            "issue_year_end": CANDIDATE_END_YEAR,
            "negative_statuses": sorted(NEGATIVE_STATUSES),
            "positive_statuses": sorted(POSITIVE_STATUSES),
        },
        "candidate_rows_all_statuses": candidate_total,
        "candidate_modelable_rows": modelable,
        "candidate_positive_rows": positives,
        "candidate_default_rate": positives / modelable if modelable else None,
        "candidate_strict_fp_co_rows": strict_modelable,
        "proposed_split": {
            "train": "2012-2014",
            "validation": "2015",
            "test": "2016",
        },
        "rationale": [
            "The dataset snapshot is 2020Q3, so later vintages are right-censored.",
            "The 36-month 2012-2016 cohort is nearly fully observed.",
            "Sixty-month loans need an earlier cutoff and should be a later extension.",
            "Current, late, grace-period, and issued statuses are not final labels.",
        ],
    }
    write_json(summary, COHORT_DIR / "candidate_cohort_summary.json")

    print("Target/cohort audit complete:")
    print(f"  candidate all statuses : {candidate_total:,}")
    print(f"  modelable rows         : {modelable:,}")
    print(f"  candidate default rate : {summary['candidate_default_rate']:.2%}")
    print(f"  output                 : {COHORT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()
    analyze_cohorts(chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
