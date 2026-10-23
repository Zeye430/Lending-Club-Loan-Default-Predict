"""Create a data-backed comparison with the original src/01-05 workflow."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from _common import AUDIT_DIR, COHORT_DIR, EDA_DIR, REPORT_ROOT, ensure_output_dirs


def compare() -> None:
    ensure_output_dirs()
    required = [
        AUDIT_DIR / "raw_audit_summary.json",
        AUDIT_DIR / "raw_column_profile.csv",
        AUDIT_DIR / "column_catalog.csv",
        COHORT_DIR / "cohort_maturity_by_year_and_term.csv",
        COHORT_DIR / "candidate_cohort_summary.json",
        EDA_DIR / "single_feature_auc_sample.csv",
        EDA_DIR / "candidate_feature_availability.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing prerequisite reports: {missing}")

    with required[0].open(encoding="utf-8") as handle:
        audit = json.load(handle)
    profile = pd.read_csv(required[1])
    catalog = pd.read_csv(required[2])
    maturity = pd.read_csv(required[3])
    with required[4].open(encoding="utf-8") as handle:
        cohort = json.load(handle)
    auc = pd.read_csv(required[5])
    availability = pd.read_csv(required[6])

    old_window = maturity[maturity["issue_year"].between(2014, 2017)]
    old_total = int(old_window["all_rows"].sum())
    old_final = int(old_window["strict_final_rows"].sum())
    old_final_share = old_final / old_total

    year_2017 = maturity[maturity["issue_year"] == 2017]
    y2017_total = int(year_2017["all_rows"].sum())
    y2017_final = int(year_2017["strict_final_rows"].sum())

    high_missing = profile.loc[profile["missing_rate"] > 0.30, "column"].tolist()
    bureau_dropped_old = [
        "tax_liens",
        "acc_now_delinq",
        "num_sats",
        "delinq_amnt",
        "num_tl_30dpd",
        "chargeoff_within_12_mths",
        "collections_12_mths_ex_med",
        "num_tl_120dpd_2m",
    ]
    bureau_dropped_old = [
        column for column in bureau_dropped_old if column in set(catalog["column"])
    ]

    model_a_count = int(
        catalog["model_a_role"].isin(["candidate_feature", "candidate_with_review"]).sum()
    )
    model_b_count = int(
        catalog["model_b_role"].isin(["candidate_feature", "candidate_with_review"]).sum()
    )
    unavailable = set(
        availability.loc[availability["unavailable_in_development_cohort"], "feature"]
    )
    model_a_usable_count = int(
        (
            catalog["model_a_role"].isin(["candidate_feature", "candidate_with_review"])
            & ~catalog["column"].isin(unavailable)
        ).sum()
    )
    model_b_usable_count = int(
        (
            catalog["model_b_role"].isin(["candidate_feature", "candidate_with_review"])
            & ~catalog["column"].isin(unavailable)
        ).sum()
    )

    auc_lines = "\n".join(
        f"- `{row.feature}`: `{row.single_feature_auc:.4f}`" for row in auc.itertuples()
    )
    high_missing_examples = ", ".join(f"`{column}`" for column in high_missing[:12])
    bureau_examples = ", ".join(f"`{column}`" for column in bureau_dropped_old)

    content = f"""# Original workflow versus data-discovery rebuild

## Dataset facts

- Raw shape: {audit["rows"]:,} rows by {audit["columns"]} columns.
- Issue-year coverage: {audit["first_issue_year"]}-{audit["last_issue_year"]}.
- Candidate cohort: {cohort["candidate_modelable_rows"]:,} modelable 36-month loans,
  issued from 2012 through 2016.
- Candidate default rate: {cohort["candidate_default_rate"]:.2%}.

## Cohort and target

The original loader selected 2014-2017 before quantifying label maturity. Across that
window, only {old_final_share:.1%} of records have strict final `Fully Paid` or
`Charged Off` labels in the 2020Q3 snapshot. In 2017 alone, the strict final share is
{y2017_final / y2017_total:.1%}.

Filtering an immature vintage to completed statuses does not solve censoring. It selects
loans partly according to how quickly they paid off or defaulted. The rebuild first audits
maturity by term and vintage, then proposes a development population.

## Missing values

The original cleaning script dropped every field above 30% missingness. The raw dataset
has {len(high_missing)} such fields, including {high_missing_examples}.

That rule combines several different mechanisms:

- post-origination fields that must be excluded for leakage;
- joint-application fields that are conditionally applicable;
- event-history fields where missing can mean no prior event;
- fields introduced in later schema vintages;
- ordinary unknown values.

The rebuild records missingness by year and by outcome before choosing a treatment.

## Feature availability

The original `drop_uninformative_cols` list removed bureau risk fields including
{bureau_examples}. Low prevalence does not make a delinquency or public-record field
uninformative. These fields remain candidates until a training-only ablation evaluates
their incremental and stability value.

The new catalog initially identifies {model_a_count} applicant/bureau candidates for
Model A and {model_b_count} for Model B. Cohort-specific availability checks remove
{len(unavailable)} fields that are 100% missing in the 2012-2015 development data, leaving
{model_a_usable_count} usable candidates for Model A and {model_b_usable_count} for
Model B. The primary difference is `sub_grade`.
`grade` is redundant, while `int_rate` and `installment` are reserved for a separate
pricing sensitivity.

## Internal-rating signal

Single-feature diagnostics on the development sample:

{auc_lines}

These diagnostics show why the original model leaned heavily on LC's rating, but they do
not replace the controlled Model A versus Model B experiment.

## EDA and splitting

The original workflow performed an 80/20 row cut inside chronologically sorted
2014-2016 data. Multiple loans from the same issue month can therefore fall on both sides
of the boundary. The rebuild proposes full-year boundaries: train 2012-2014, validate
2015, and test 2016, subject to the final cohort decision.

Feature-target EDA is restricted to the 2012-2015 development cohort. The 2016 test
period does not determine transformations, outlier caps, missing-value rules, category
groupings, or feature selection; it appears only in the pre-freeze maturity and temporal
drift audit.

## Expected modeling benefits

1. More reliable labels reduce noise caused by right-censoring.
2. Restored bureau fields may recover signal previously discarded before validation.
3. Missingness-aware treatment can preserve event-history and joint-application signal.
4. Stable vintage-aware features should generalize better across time.
5. A controlled `sub_grade` ablation will measure the value of LC's internal rating
   instead of mixing it with pricing information.

These changes improve the experimental design and the potential for out-of-time
performance. They do not guarantee a higher headline AUC, and a lower but unbiased metric
is preferable to an optimistic result produced by immature labels or leakage.
"""
    path = REPORT_ROOT / "original_vs_rebuild.md"
    temp_path = path.with_suffix(".md.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)
    print(f"Comparison report written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    compare()


if __name__ == "__main__":
    main()
