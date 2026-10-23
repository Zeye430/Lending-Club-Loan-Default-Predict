"""Combine the data dictionary, empirical profile, and prediction-time policy."""

from __future__ import annotations

import argparse

import pandas as pd

from _common import AUDIT_DIR, ensure_output_dirs, write_csv, write_json

IDENTIFIERS = {"Unnamed: 0", "id", "url"}

LOAN_REQUEST = {"loan_amnt", "term", "purpose"}

APPLICANT_PROVIDED = {
    "emp_title",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "title",
    "zip_code",
    "addr_state",
    "application_type",
    "annual_inc_joint",
    "dti_joint",
}

VERIFICATION_PROCESS = {"verification_status", "verification_status_joint"}

LC_RATING = {"grade", "sub_grade"}
PRICING_OUTPUT = {"int_rate", "installment"}

FUNDING_OR_LISTING = {
    "funded_amnt",
    "funded_amnt_inv",
    "issue_d",
    "initial_list_status",
}

ADMINISTRATIVE = {"policy_code"}

POST_ORIGINATION = {
    "pymnt_plan",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "last_credit_pull_d",
    "last_fico_range_high",
    "last_fico_range_low",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "deferral_term",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "payment_plan_start_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
}

TARGET_SOURCE = {"loan_status"}

HIGH_CARDINALITY_TEXT = {"emp_title", "title"}
FAIRNESS_REVIEW = {"zip_code", "addr_state"}
REDUNDANT_OR_DERIVED = {
    "grade",
    "fico_range_high",
    "funded_amnt",
    "funded_amnt_inv",
    "installment",
}


def source_category(column: str) -> str:
    if column in IDENTIFIERS:
        return "identifier"
    if column in LOAN_REQUEST:
        return "loan_request"
    if column in APPLICANT_PROVIDED:
        return "applicant"
    if column in VERIFICATION_PROCESS:
        return "verification_process"
    if column in LC_RATING:
        return "lc_internal_rating"
    if column in PRICING_OUTPUT:
        return "pricing_output"
    if column in FUNDING_OR_LISTING:
        return "funding_or_listing"
    if column in ADMINISTRATIVE:
        return "administrative"
    if column in POST_ORIGINATION:
        return "post_origination"
    if column in TARGET_SOURCE:
        return "target_source"
    return "credit_bureau"


def timing(category: str) -> str:
    return {
        "identifier": "administrative",
        "loan_request": "application_or_final_offer",
        "applicant": "application",
        "verification_process": "pre_origination_verification",
        "credit_bureau": "application_or_origination_bureau_pull",
        "lc_internal_rating": "after_lc_underwriting",
        "pricing_output": "after_lc_underwriting",
        "funding_or_listing": "listing_or_funding",
        "administrative": "platform_policy",
        "target_source": "observed_after_origination",
        "post_origination": "after_first_payment_or_distress_event",
    }[category]


def feature_policy(column: str, category: str) -> tuple[str, str, str]:
    """Return Model A role, Model B role, and rationale."""
    if category == "target_source":
        return "target_only", "target_only", "Source used to construct the outcome label."
    if category == "post_origination":
        return (
            "exclude_leakage",
            "exclude_leakage",
            "Created after origination and can reveal repayment or distress outcomes.",
        )
    if category == "identifier":
        return "exclude", "exclude", "Identifier or exported row index, not borrower risk."
    if column == "issue_d":
        return (
            "split_only",
            "split_only",
            "Use for vintage splits and as-of macro joins, not as a borrower attribute.",
        )
    if category in {"funding_or_listing", "administrative"}:
        return (
            "exclude_initially",
            "exclude_initially",
            "Platform workflow or funding information can encode selection policy.",
        )
    if category == "pricing_output":
        return (
            "exclude",
            "exclude_primary_comparison",
            "Pricing contains LC underwriting information; reserve for a sensitivity test.",
        )
    if category == "lc_internal_rating":
        if column == "sub_grade":
            return (
                "exclude",
                "candidate_feature",
                "The single added field in the primary LC-rating ablation.",
            )
        return (
            "exclude",
            "exclude_redundant",
            "Grade is fully contained in the first character of sub_grade.",
        )
    if column in HIGH_CARDINALITY_TEXT:
        return (
            "exclude_initially",
            "exclude_initially",
            "Free text has high cardinality, privacy, stability, and explanation risk.",
        )
    if column == "zip_code":
        return (
            "fairness_review",
            "fairness_review",
            "Geography may proxy protected characteristics; exclude until reviewed.",
        )
    if column == "fico_range_high":
        return (
            "exclude_redundant",
            "exclude_redundant",
            "FICO range high is nearly deterministic from fico_range_low.",
        )
    if column in FAIRNESS_REVIEW:
        return (
            "candidate_with_review",
            "candidate_with_review",
            "Potential geographic signal requires stability and fairness review.",
        )
    if category == "verification_process":
        return (
            "candidate_with_review",
            "candidate_with_review",
            "Known by origination but can encode LC verification policy and selection.",
        )
    return (
        "candidate_feature",
        "candidate_feature",
        "Available by origination and directly related to application or credit risk.",
    )


def build_catalog() -> None:
    ensure_output_dirs()
    profile_path = AUDIT_DIR / "raw_column_profile.csv"
    if not profile_path.exists():
        raise FileNotFoundError(
            f"Missing {profile_path}. Run scripts/02_validate_raw_data.py first."
        )

    profile = pd.read_csv(profile_path)
    rows = []
    for record in profile.to_dict(orient="records"):
        column = record["column"]
        category = source_category(column)
        model_a_role, model_b_role, reason = feature_policy(column, category)

        confidence = "high"
        if category in {
            "loan_request",
            "verification_process",
            "funding_or_listing",
        }:
            confidence = "medium"

        rows.append(
            {
                **record,
                "source_category": category,
                "available_at": timing(category),
                "model_a_role": model_a_role,
                "model_b_role": model_b_role,
                "decision_reason": reason,
                "timing_confidence": confidence,
                "requires_fairness_review": column in FAIRNESS_REVIEW,
            }
        )

    catalog = pd.DataFrame(rows)
    write_csv(catalog, AUDIT_DIR / "column_catalog.csv")

    category_summary = (
        catalog.groupby("source_category", dropna=False)
        .agg(
            column_count=("column", "count"),
            median_missing_rate=("missing_rate", "median"),
            first_year=("first_year_nonnull", "min"),
            last_year=("last_year_nonnull", "max"),
        )
        .reset_index()
        .sort_values("source_category")
    )
    write_csv(category_summary, AUDIT_DIR / "column_category_summary.csv")

    role_summary = (
        catalog.groupby(["model_a_role", "model_b_role"])
        .size()
        .rename("column_count")
        .reset_index()
    )
    write_csv(role_summary, AUDIT_DIR / "feature_policy_summary.csv")

    model_a = catalog.loc[
        catalog["model_a_role"].isin(["candidate_feature", "candidate_with_review"]),
        "column",
    ].tolist()
    model_b = catalog.loc[
        catalog["model_b_role"].isin(["candidate_feature", "candidate_with_review"]),
        "column",
    ].tolist()

    policy = {
        "prediction_time": "approved loan at origination, before any repayment",
        "model_a_name": "Independent Risk",
        "model_b_name": "LC-Assisted Risk",
        "model_a_candidate_feature_count": len(model_a),
        "model_b_candidate_feature_count": len(model_b),
        "primary_ablation_difference": sorted(set(model_b) - set(model_a)),
        "model_a_candidates": model_a,
        "model_b_candidates": model_b,
        "notes": [
            "Candidate does not mean final keep; stability and training-only EDA remain required.",
            "int_rate and installment are excluded from the primary comparison.",
            "A later pricing sensitivity may add int_rate to Model B.",
        ],
    }
    write_json(policy, AUDIT_DIR / "feature_set_policy.json")

    print("Column catalog complete:")
    print(f"  columns classified : {len(catalog)}")
    print(f"  Model A candidates : {len(model_a)}")
    print(f"  Model B candidates : {len(model_b)}")
    print(f"  primary difference : {policy['primary_ablation_difference']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    build_catalog()


if __name__ == "__main__":
    main()
