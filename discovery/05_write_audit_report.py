"""Publish the raw-data audit as a readable report: discovery/reports/data_audit.md.

Reads the machine outputs of stages 01-04 (under reports/data_discovery/, not committed)
and writes a markdown report plus figures that are committed with the code.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from _common import AUDIT_DIR, COHORT_DIR, ROOT
from lc_risk import config, schema

PUBLISHED_DIR = ROOT / "discovery" / "reports"
FIGURES_DIR = PUBLISHED_DIR / "figures"
REPORT_FILE = PUBLISHED_DIR / "data_audit.md"

CANDIDATE_ROLES = {"candidate_feature", "candidate_with_review"}
CATEGORY_NOTES = {
    "credit_bureau": "Bureau attributes pulled at application",
    "applicant": "Self-reported by the applicant",
    "loan_request": "Amount, term, purpose",
    "verification_process": "LC income verification",
    "lc_internal_rating": "LC grade / sub-grade",
    "pricing_output": "Interest rate, installment",
    "funding_or_listing": "Funding amounts, listing status, issue date",
    "post_origination": "Payments, recoveries, hardship, settlement",
    "identifier": "IDs and URLs",
    "administrative": "Platform policy code",
    "target_source": "Loan status",
}


def md_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def load_inputs() -> dict:
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    paths = {
        "source": AUDIT_DIR / "source_manifest.json",
        "audit": AUDIT_DIR / "raw_audit_summary.json",
        "quality": AUDIT_DIR / "data_quality_checks.csv",
        "catalog": AUDIT_DIR / "column_catalog.csv",
        "nonnull": AUDIT_DIR / "raw_nonnull_rate_by_year.csv",
        "maturity": COHORT_DIR / "cohort_maturity_by_year_and_term.csv",
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run discovery stages 01-04 first. Missing: {missing}")
    return {
        name: read_json(path) if path.suffix == ".json" else pd.read_csv(path)
        for name, path in paths.items()
    }


def plot_label_maturity(maturity: pd.DataFrame) -> str:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    window = maturity[maturity["issue_year"].between(2010, 2017)]
    fig, ax = plt.subplots(figsize=(8, 4))
    for term, style in ((36, "o-"), (60, "s--")):
        rows = window[window["term_months"] == term]
        ax.plot(rows["issue_year"], 100 * rows["strict_final_share"], style, label=f"{term}-month")
    years = config.SPLIT_YEARS
    ax.axvspan(min(years["train"]) - 0.4, max(years["test"]) + 0.4, alpha=0.12, color="tab:green")
    ax.text(min(years["train"]), 52, "modeling cohort", color="tab:green")
    ax.set_ylim(50, 102)
    ax.set_xlabel("Issue year")
    ax.set_ylabel("Loans with a final status (%)")
    ax.set_title("Label maturity in the 2020Q3 snapshot")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "label_maturity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.relative_to(PUBLISHED_DIR).as_posix()


def schema_rollout_groups(catalog: pd.DataFrame, nonnull: pd.DataFrame) -> pd.DataFrame:
    """Group candidate fields by when LC populated them, from non-null rates per year."""
    candidates = catalog.loc[catalog["model_b_role"].isin(CANDIDATE_ROLES), "column"]
    rates = nonnull.set_index("issue_year").loc[2012:2016, list(candidates)].T

    def group(row: pd.Series) -> str:
        if row.max() < 0.05:
            return "Joint / secondary applicant"
        if row[2014] < 0.05 and row[2016] > 0.85:
            return "Added late 2015"
        if row[2012] < 0.9 and row[2013] > 0.95:
            return "Rolled out during 2012"
        return "Stable 2012-2016"

    rates["group"] = rates.apply(group, axis=1)
    summary = rates.groupby("group").agg(
        fields=(2012, "size"),
        **{str(y): (y, "median") for y in range(2012, 2017)},
    )
    examples = rates.groupby("group").apply(lambda g: ", ".join(f"`{c}`" for c in g.index[:3]))
    summary = summary.assign(examples=examples).reset_index()
    for year in map(str, range(2012, 2017)):
        summary[year] = summary[year].map(pct)
    order = ["Stable 2012-2016", "Rolled out during 2012", "Added late 2015",
             "Joint / secondary applicant"]  # fmt: skip
    return summary.set_index("group").loc[[g for g in order if g in summary["group"].values]]


def build_report(inputs: dict) -> str:
    source, audit = inputs["source"], inputs["audit"]
    maturity, catalog = inputs["maturity"], inputs["catalog"]
    figure = plot_label_maturity(maturity)

    quality = inputs["quality"]
    quality = quality[~quality["check"].isin(["row_count", "column_count"])].copy()
    quality["value"] = quality["value"].map("{:,}".format)
    quality_table = md_table(quality[["check", "value", "severity"]])

    recent = maturity[maturity["issue_year"].between(2012, 2017)].copy()
    maturity_table = md_table(
        pd.DataFrame(
            {
                "issue year": recent["issue_year"],
                "term": recent["term_months"].map("{}m".format),
                "loans": recent["all_rows"].map("{:,}".format),
                "final status": recent["strict_final_share"].map(pct),
                "still current": recent["current_rows"].map("{:,}".format),
                "charge-off rate (resolved)": recent["strict_charge_off_rate"].map(pct),
            }
        )
    )

    categories = catalog.groupby("source_category").agg(
        columns=("column", "size"),
        model_a=("model_a_role", lambda r: int(r.isin(CANDIDATE_ROLES).sum())),
    )
    categories["what it is"] = categories.index.map(CATEGORY_NOTES)
    categories = categories.sort_values("columns", ascending=False).reset_index()
    categories.columns = ["category", "columns", "input candidates", "what it is"]

    rollout = schema_rollout_groups(catalog, inputs["nonnull"]).reset_index()

    window = maturity[maturity["issue_year"].between(2014, 2017)]
    v1_final_share = window["strict_final_rows"].sum() / window["all_rows"].sum()
    y2017 = maturity[maturity["issue_year"] == 2017]
    y2017_share = y2017["strict_final_rows"].sum() / y2017["all_rows"].sum()
    over_30 = audit["columns_over_30pct_missing"]

    n_inputs = len(schema.APPLICATION_FIELDS)
    splits = {
        name: f"{min(years)}–{max(years)}" if len(years) > 1 else str(years[0])
        for name, years in config.SPLIT_YEARS.items()
    }
    generated = datetime.now(UTC).strftime("%Y-%m-%d")

    return f"""# Raw data audit

> Generated by `discovery/05_write_audit_report.py` on {generated} from the outputs of
> stages 01–04. Do not edit by hand; rerun `python discovery/run_data_discovery.py`.

This report answers one question: **which loans and which columns can a default model
trust?** Each section ends with the decision it led to; section 6 shows where each decision
lives in code.

## 1. Source

| | |
|---|---|
| Dataset | Kaggle `{source["dataset"]}` |
| Raw file | `{source["raw_file"]}` ({source["raw_file_size_bytes"] / 1e9:.2f} GB) |
| SHA-256 | `{source["raw_file_sha256"][:16]}…` |
| Shape | {audit["rows"]:,} loans × {audit["columns"]} columns |
| Issue years | {audit["first_issue_year"]}–{audit["last_issue_year"]} (snapshot taken 2020Q3) |

## 2. Data quality

The export is clean at the row level: loan IDs are unique, and only a handful of values are
impossible.

{quality_table}

**Decision:** impossible values (negative DTI, revolving utilization above 200%, non-positive
income, the DTI sentinel 999) become null, and every rule's hit count is recorded in the
cohort manifest.

## 3. Label maturity: which loans have a trustworthy outcome?

A loan's label (paid off or charged off) is only known once the loan is resolved. The
snapshot was taken in 2020Q3, so recent and long-term loans are often still running.

![Label maturity]({figure})

{maturity_table}

Keeping only the resolved loans from an immature vintage does not fix this. Loans resolve
early because they either default quickly or are prepaid, so the kept sample is biased
toward both extremes. 36-month loans issued through 2016 are almost fully resolved, while
60-month loans lag by about two years.

**Decision:** model 36-month loans issued {min(config.SPLIT_YEARS["train"])}–{max(config.SPLIT_YEARS["test"])}.
The 60-month book would need its own, earlier cohort.

## 4. Column taxonomy: what is known at application time?

Every column was classified by when its value exists relative to the lending decision.

{md_table(categories)}

Post-origination fields (payments, recoveries, settlements) leak the outcome. LC's grade
and interest rate are outputs of LC's own risk model, so using them as inputs would
mostly reproduce LC's model.

**Decision:** inputs are application and bureau fields only. `sub_grade` is added in
Model B alone, to measure what LC's rating contributes on top. `int_rate` is kept for
evaluation, never as an input. Section 5 narrows the candidates further.

## 5. Schema availability: when did LC start populating each field?

Median non-null rate by issue year for the input candidates:

{md_table(rollout)}

Missing values in these groups come from LC's export schema changing over time; the
borrowers did have these attributes. A model trained on 2012 would learn a missingness
pattern that never appears in new applications, and the fields added in late 2015 are
empty in every training year.

**Decision:** start the cohort in 2013, and exclude the late-2015 and joint-applicant
fields. Event-history fields such as `mths_since_last_delinq` stay in: there, a missing
value means "no such event" and carries signal.

## 6. From audit to code

| Decision | Value | Lives in |
|---|---|---|
| Term | {config.TERM_MONTHS} months | `src/lc_risk/config.py` |
| Splits by issue year | train {splits["train"]}, validation {splits["validation"]}, test {splits["test"]} | `src/lc_risk/config.py` |
| Labels | {", ".join(sorted(config.POSITIVE_STATUSES))} = 1; {", ".join(sorted(config.NEGATIVE_STATUSES))} = 0 | `src/lc_risk/config.py` |
| Inputs | {n_inputs} application/bureau fields (+ `sub_grade` in Model B) | `src/lc_risk/schema.py` |
| Cleaning rules | null out impossible and sentinel values | `src/lc_risk/data/cohort.py` |

## 7. What this changes from the v1 course project

- **Cohort.** v1 used 2014–2017 loans of both terms. Only {pct(v1_final_share)} of those had
  a final status, and only {pct(y2017_share)} of the 2017 test vintage did.
- **Missing values.** v1 dropped all {over_30} columns above 30% missing, which removed
  event-history fields whose missingness is informative.
- **Bureau fields.** v1 dropped low-prevalence bureau risk fields (`tax_liens`,
  `acc_now_delinq`, `chargeoff_within_12_mths`, …) before any validation.
- **Split.** v1 cut chronologically sorted rows at 80/20, so the same issue month landed in
  both train and validation. v2 splits by whole years.
- **LC rating.** v1 used `sub_grade` and `int_rate` together, so the model's value beyond LC's
  own rating was never measured.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    report = build_report(load_inputs())
    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
