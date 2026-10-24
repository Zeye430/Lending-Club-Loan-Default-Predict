"""Request and response models for the scoring API, generated from ``lc_risk.schema``.

Field names, types, allowed ranges, categories, and descriptions all come from the shared
schema, so the API cannot drift from what the model was trained on. Unknown fields are
rejected rather than silently ignored.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from lc_risk import schema

# Every field that was always present in training is required; only fields where a null
# is meaningful (schema.Field.nullable) may be omitted.
REQUIRED_FIELDS = tuple(f.name for f in schema.APPLICATION_FIELDS if not f.nullable)
MAX_BATCH = 1000

# A typical applicant: training-set medians/modes, with no delinquency or public record.
EXAMPLE_APPLICATION = {
    "loan_amnt": 10000, "purpose": "debt_consolidation", "emp_length": 6,
    "home_ownership": "MORTGAGE", "annual_inc": 60000, "verification_status": "Not Verified",
    "addr_state": "CA", "dti": 16.8, "fico_range_low": 685, "earliest_cr_line": "1999-07-01",
    "open_acc": 10, "total_acc": 23, "revol_bal": 10810, "revol_util": 56.1,
    "tot_cur_bal": 65124, "avg_cur_bal": 6374, "total_rev_hi_lim": 21200,
    "tot_hi_cred_lim": 92594, "total_bal_ex_mort": 32546, "total_bc_limit": 13100,
    "total_il_high_credit_limit": 26507, "bc_open_to_buy": 3531, "bc_util": 68.5,
    "percent_bc_gt_75": 50.0, "mort_acc": 1, "num_actv_bc_tl": 3, "num_actv_rev_tl": 5,
    "num_bc_sats": 4, "num_bc_tl": 8, "num_il_tl": 6, "num_op_rev_tl": 7, "num_rev_accts": 14,
    "num_rev_tl_bal_gt_0": 5, "num_sats": 10, "inq_last_6mths": 0, "acc_open_past_24mths": 4,
    "num_tl_op_past_12m": 2, "mo_sin_old_il_acct": 128, "mo_sin_old_rev_tl_op": 163,
    "mo_sin_rcnt_rev_tl_op": 8, "mo_sin_rcnt_tl": 6, "mths_since_recent_bc": 14,
    "mths_since_recent_inq": 6, "delinq_2yrs": 0, "mths_since_last_delinq": None,
    "mths_since_last_major_derog": None, "mths_since_recent_bc_dlq": None,
    "mths_since_recent_revol_delinq": None, "acc_now_delinq": 0, "delinq_amnt": 0,
    "num_tl_30dpd": 0, "num_tl_120dpd_2m": 0, "num_tl_90g_dpd_24m": 0,
    "num_accts_ever_120_pd": 0, "pct_tl_nvr_dlq": 100, "chargeoff_within_12_mths": 0,
    "collections_12_mths_ex_med": 0, "tot_coll_amt": 0, "pub_rec": 0,
    "mths_since_last_record": None, "pub_rec_bankruptcies": 0, "tax_liens": 0,
}  # fmt: skip


def _field_definition(field: schema.Field):
    if field.kind == "numeric":
        annotation, constraints = float, {"ge": field.ge, "le": field.le}
    elif field.kind == "categorical":
        annotation, constraints = Literal[field.categories], {}
    else:
        annotation, constraints = date, {}
    if field.name in schema.EXCLUDED_FROM_MODEL:
        description = f"{field.description} Recorded for monitoring; not used by the model."
    else:
        description = field.description
    if field.name in REQUIRED_FIELDS:
        return annotation, Field(..., description=description, **constraints)
    return annotation | None, Field(None, description=description, **constraints)


Application = create_model(
    "Application",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_APPLICATION}),
    application_date=(
        date | None,
        Field(None, description="Date the application is scored; defaults to today."),
    ),
    **{f.name: _field_definition(f) for f in schema.APPLICATION_FIELDS},
)


class Reason(BaseModel):
    feature: str
    description: str
    value: float | str | None
    contribution: float = Field(description="Increase in calibrated log-odds of default.")


class Score(BaseModel):
    probability_of_default: float
    reasons: list[Reason] = Field(description="Top factors raising this applicant's risk.")
    warnings: list[str] = Field(description="Inputs outside the population the model saw.")


class ModelInfo(BaseModel):
    name: str
    version: str


class ScoreResponse(Score):
    model: ModelInfo


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    applications: list[Application] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    model: ModelInfo
    results: list[Score]
