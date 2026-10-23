"""Single source of truth for model inputs.

Every consumer reads its column lists from here: the cohort builder (parsing and
validation), the training pipeline, the API request model, and drift monitoring.

Inputs are the fields known when a 36-month loan application is scored, before LC
assigns a grade or price. Fields are excluded on purpose when they are:

- created after origination (payments, recoveries, settlements): outcome leakage;
- LC pricing (``int_rate``, ``installment``): kept only for evaluation;
- joint-application or secondary-applicant fields: absent in the training years;
- bureau fields LC added in late 2015 (``open_acc_6m``, ``all_util``, ...): 100%
  missing in the training years, so a model cannot learn from them;
- redundant (``fico_range_high``, ``grade``) or high-cardinality text (``emp_title``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FieldKind = Literal["numeric", "categorical", "date"]


@dataclass(frozen=True)
class Field:
    name: str
    kind: FieldKind
    description: str
    ge: float | None = None
    le: float | None = None
    categories: tuple[str, ...] = ()


def _num(name: str, description: str, ge: float | None = 0, le: float | None = None) -> Field:
    return Field(name, "numeric", description, ge=ge, le=le)


def _cat(name: str, description: str, categories: tuple[str, ...]) -> Field:
    return Field(name, "categorical", description, categories=categories)


HOME_OWNERSHIP = ("MORTGAGE", "RENT", "OWN", "OTHER")
VERIFICATION_STATUS = ("Not Verified", "Source Verified", "Verified")
PURPOSES = (
    "debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
    "medical", "car", "small_business", "vacation", "moving", "house",
    "renewable_energy", "wedding",
)  # fmt: skip
US_STATES = (
    "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "IA", "ID",
    "IL", "IN", "KS", "KY", "LA", "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC",
    "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC", "SD",
    "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY",
)  # fmt: skip
SUB_GRADES = tuple(f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6))

APPLICATION_FIELDS: tuple[Field, ...] = (
    # loan request and applicant
    _num("loan_amnt", "Requested loan amount (USD).", ge=500, le=40_000),
    _cat("purpose", "Borrower-stated loan purpose.", PURPOSES),
    _num("emp_length", "Employment length in years; 0 = <1 year, 10 = 10+ years.", le=10),
    _cat("home_ownership", "Home ownership status.", HOME_OWNERSHIP),
    _num("annual_inc", "Self-reported annual income (USD).", ge=1),
    _cat("verification_status", "Whether income or its source was verified.", VERIFICATION_STATUS),
    _cat("addr_state", "Borrower state.", US_STATES),
    _num("dti", "Monthly debt payments / monthly income, excl. mortgage and this loan (%)."),
    # credit bureau: profile
    _num("fico_range_low", "Lower bound of FICO range at application.", ge=300, le=850),
    Field("earliest_cr_line", "date", "Month the earliest reported credit line was opened."),
    _num("open_acc", "Open credit lines."),
    _num("total_acc", "Total credit lines in file."),
    _num("revol_bal", "Total revolving balance (USD)."),
    _num("revol_util", "Revolving utilization (%).", le=200),
    _num("tot_cur_bal", "Total current balance of all accounts (USD)."),
    _num("avg_cur_bal", "Average current balance of all accounts (USD)."),
    _num("total_rev_hi_lim", "Total revolving credit limit (USD)."),
    _num("tot_hi_cred_lim", "Total high credit / credit limit (USD)."),
    _num("total_bal_ex_mort", "Total balance excluding mortgage (USD)."),
    _num("total_bc_limit", "Total bankcard credit limit (USD)."),
    _num("total_il_high_credit_limit", "Total installment credit limit (USD)."),
    _num("bc_open_to_buy", "Total open-to-buy on revolving bankcards (USD)."),
    _num("bc_util", "Bankcard balance / limit (%)."),
    _num("percent_bc_gt_75", "Share of bankcards above 75% of limit (%).", le=100),
    _num("mort_acc", "Mortgage accounts."),
    _num("num_actv_bc_tl", "Active bankcard accounts."),
    _num("num_actv_rev_tl", "Active revolving trades."),
    _num("num_bc_sats", "Satisfactory bankcard accounts."),
    _num("num_bc_tl", "Bankcard accounts."),
    _num("num_il_tl", "Installment accounts."),
    _num("num_op_rev_tl", "Open revolving accounts."),
    _num("num_rev_accts", "Revolving accounts."),
    _num("num_rev_tl_bal_gt_0", "Revolving trades with balance > 0."),
    _num("num_sats", "Satisfactory accounts."),
    # credit bureau: recent activity
    _num("inq_last_6mths", "Inquiries in past 6 months, excl. auto and mortgage."),
    _num("acc_open_past_24mths", "Trades opened in past 24 months."),
    _num("num_tl_op_past_12m", "Accounts opened in past 12 months."),
    _num("mo_sin_old_il_acct", "Months since oldest installment account opened."),
    _num("mo_sin_old_rev_tl_op", "Months since oldest revolving account opened."),
    _num("mo_sin_rcnt_rev_tl_op", "Months since most recent revolving account opened."),
    _num("mo_sin_rcnt_tl", "Months since most recent account opened."),
    _num("mths_since_recent_bc", "Months since most recent bankcard opened."),
    _num("mths_since_recent_inq", "Months since most recent inquiry; null = none on file."),
    # credit bureau: delinquency and derogatory history.
    # For the mths_since_* fields, null means no such event on file, not an unknown value.
    _num("delinq_2yrs", "30+ day delinquencies in past 2 years."),
    _num("mths_since_last_delinq", "Months since last delinquency; null = none."),
    _num("mths_since_last_major_derog", "Months since last 90+ day rating; null = none."),
    _num("mths_since_recent_bc_dlq", "Months since last bankcard delinquency; null = none."),
    _num("mths_since_recent_revol_delinq", "Months since last revolving delinquency; null = none."),
    _num("acc_now_delinq", "Accounts currently delinquent."),
    _num("delinq_amnt", "Past-due amount on currently delinquent accounts (USD)."),
    _num("num_tl_30dpd", "Accounts currently 30 days past due."),
    _num("num_tl_120dpd_2m", "Accounts currently 120 days past due."),
    _num("num_tl_90g_dpd_24m", "Accounts 90+ days past due in past 24 months."),
    _num("num_accts_ever_120_pd", "Accounts ever 120+ days past due."),
    _num("pct_tl_nvr_dlq", "Trades never delinquent (%).", le=100),
    _num("chargeoff_within_12_mths", "Charge-offs in past 12 months."),
    _num("collections_12_mths_ex_med", "Collections in past 12 months, excl. medical."),
    _num("tot_coll_amt", "Total collection amounts ever owed (USD)."),
    _num("pub_rec", "Derogatory public records."),
    _num("mths_since_last_record", "Months since last public record; null = none."),
    _num("pub_rec_bankruptcies", "Public record bankruptcies."),
    _num("tax_liens", "Tax liens."),
)

# LC's own risk grade. Used only by Model B, to measure its incremental value.
LC_RATING_FIELD = _cat("sub_grade", "LC-assigned sub-grade, A1 (best) to G5.", SUB_GRADES)

# Non-feature columns carried in the cohort for splitting, evaluation, and audit.
ID_COL = "loan_id"
DATE_COL = "issue_d"
PRICING_COLS = ("int_rate", "installment")
OUTCOME_COLS = ("loan_status", "funded_amnt", "total_pymnt", "recoveries", "last_pymnt_d")


def model_inputs(include_lc_rating: bool) -> tuple[Field, ...]:
    if include_lc_rating:
        return (*APPLICATION_FIELDS, LC_RATING_FIELD)
    return APPLICATION_FIELDS


ALL_INPUT_FIELDS = model_inputs(include_lc_rating=True)
FIELDS_BY_NAME = {f.name: f for f in ALL_INPUT_FIELDS}
