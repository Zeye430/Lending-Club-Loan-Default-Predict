"""Raw application fields -> model features.

``FeatureBuilder`` is stateless: it learns nothing from data, so the training pipeline and
the scoring service produce identical features from identical inputs. Everything it needs
(categories, field lists) comes from ``lc_risk.schema``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from lc_risk import schema

DERIVED_FEATURES = ("credit_history_months", "loan_to_income")
SUB_GRADE_RANK = "sub_grade_rank"
_SUB_GRADE_TO_RANK = {grade: rank for rank, grade in enumerate(schema.SUB_GRADES)}

# Expected direction of risk (+1: higher value -> higher risk), taken from the EDA
# (discovery/eda.ipynb, section 3). XGBoost enforces these when constraints are enabled.
MONOTONE_DIRECTIONS = {
    "fico_range_low": -1,
    "annual_inc": -1,
    "mort_acc": -1,
    "credit_history_months": -1,
    "dti": +1,
    "loan_to_income": +1,
    "revol_util": +1,
    "inq_last_6mths": +1,
    "acc_open_past_24mths": +1,
    "num_tl_op_past_12m": +1,
    "delinq_2yrs": +1,
    "pub_rec_bankruptcies": +1,
    SUB_GRADE_RANK: +1,
}


class FeatureBuilder(TransformerMixin, BaseEstimator):
    """Turn a frame of raw inputs (``schema`` fields + ``issue_d``) into model features.

    - numeric fields pass through as float; nulls stay null (they carry signal);
    - categorical fields become pandas categoricals over the schema's fixed categories,
      so an unseen value at scoring time becomes null instead of a new column;
    - ``sub_grade`` (Model B only) becomes an ordinal rank, A1 = 0 ... G5 = 34;
    - ``earliest_cr_line`` and ``issue_d`` become ``credit_history_months``.
    """

    def __init__(self, include_lc_rating: bool = False):
        self.include_lc_rating = include_lc_rating

    @property
    def input_fields(self) -> tuple[schema.Field, ...]:
        return schema.model_inputs(self.include_lc_rating)

    def fit(self, X: pd.DataFrame, y=None) -> FeatureBuilder:
        self.feature_names_out_ = np.array(self._output_columns(), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        required = [f.name for f in self.input_fields] + [schema.DATE_COL]
        if missing := [c for c in required if c not in X.columns]:
            raise KeyError(f"missing input columns: {missing}")

        out = {}
        for field in self.input_fields:
            values = X[field.name]
            if field.name == schema.LC_RATING_FIELD.name:
                out[SUB_GRADE_RANK] = values.map(_SUB_GRADE_TO_RANK).astype("float64")
            elif field.kind == "numeric":
                out[field.name] = pd.to_numeric(values, errors="coerce").astype("float64")
            elif field.kind == "categorical":
                known = values.where(values.isin(field.categories))
                out[field.name] = pd.Categorical(known, categories=field.categories)

        issued = pd.to_datetime(X[schema.DATE_COL])
        opened = pd.to_datetime(X["earliest_cr_line"])
        months = (issued.dt.year - opened.dt.year) * 12 + (issued.dt.month - opened.dt.month)
        out["credit_history_months"] = months.astype("float64").clip(lower=0)
        annual_inc = out["annual_inc"].where(out["annual_inc"] > 0)
        out["loan_to_income"] = out["loan_amnt"] / annual_inc

        return pd.DataFrame(out, index=X.index)[self._output_columns()]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self._output_columns(), dtype=object)

    def _output_columns(self) -> list[str]:
        columns = []
        for field in self.input_fields:
            if field.name == schema.LC_RATING_FIELD.name:
                columns.append(SUB_GRADE_RANK)
            elif field.kind != "date":
                columns.append(field.name)
        return [*columns, *DERIVED_FEATURES]


def monotone_constraints(feature_names: list[str]) -> dict[str, int]:
    return {
        name: MONOTONE_DIRECTIONS[name] for name in feature_names if name in MONOTONE_DIRECTIONS
    }
