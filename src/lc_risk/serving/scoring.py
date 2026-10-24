"""Turn validated API requests into model inputs, and model outputs into responses."""

from __future__ import annotations

from datetime import date

import pandas as pd
from pydantic import BaseModel

from lc_risk import schema
from lc_risk.bundle import Bundle
from lc_risk.features import DERIVED_DESCRIPTIONS
from lc_risk.serving.schemas import Reason, Score

# Edges of LC's historical approval box (discovery/eda.ipynb, section 6). Applicants past
# them are scored, but the model never saw anyone like them.
POLICY_CHECKS = (
    ("fico_range_low", lambda r: r["fico_range_low"] < 660, "FICO below 660"),
    ("dti", lambda r: r["dti"] >= 40, "DTI of 40% or more"),
    ("loan_amnt", lambda r: r["loan_amnt"] > 35_000, "36-month loan above $35,000"),
    ("annual_inc", lambda r: r["loan_amnt"] / r["annual_inc"] > 0.5, "loan above 50% of income"),
)
WARNING_SUFFIX = ": outside the approved population the model was trained on"


def to_frame(applications: list[BaseModel], today: date | None = None) -> pd.DataFrame:
    """Build the raw-input frame the model pipeline expects (same columns as the cohort)."""
    today = today or date.today()
    rows = [a.model_dump() for a in applications]
    frame = pd.DataFrame(rows, columns=[f.name for f in schema.APPLICATION_FIELDS])
    for field in schema.APPLICATION_FIELDS:
        if field.kind == "numeric":
            frame[field.name] = pd.to_numeric(frame[field.name]).astype("float64")
        elif field.kind == "date":
            frame[field.name] = pd.to_datetime(frame[field.name])
    frame[schema.DATE_COL] = pd.to_datetime([r["application_date"] or today for r in rows])
    return frame


def policy_warnings(row: pd.Series) -> list[str]:
    return [message + WARNING_SUFFIX for field, check, message in POLICY_CHECKS
            if pd.notna(row[field]) and check(row)]  # fmt: skip


def _describe(feature: str) -> str:
    field = schema.FIELDS_BY_NAME.get(feature)
    return field.description if field else DERIVED_DESCRIPTIONS.get(feature, feature)


def score(bundle: Bundle, applications: list[BaseModel], top_k: int = 3) -> list[Score]:
    frame = to_frame(applications)
    probabilities = bundle.model.predict_pd(frame)
    explanations = bundle.model.explain(frame, top_k=top_k)
    return [
        Score(
            probability_of_default=round(float(p), 6),
            reasons=[
                Reason(
                    feature=r.feature,
                    description=_describe(r.feature),
                    value=r.value,
                    contribution=r.contribution,
                )
                for r in reasons
            ],
            warnings=policy_warnings(row),
        )
        for p, reasons, (_, row) in zip(probabilities, explanations, frame.iterrows(), strict=True)
    ]
