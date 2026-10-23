import numpy as np
import pandas as pd
import pytest

from lc_risk import schema
from lc_risk.features import (
    DERIVED_FEATURES,
    MONOTONE_DIRECTIONS,
    SUB_GRADE_RANK,
    FeatureBuilder,
    monotone_constraints,
)


def test_model_a_and_b_outputs(applications):
    frame, _ = applications
    a = FeatureBuilder(include_lc_rating=False).fit_transform(frame)
    b = FeatureBuilder(include_lc_rating=True).fit_transform(frame)

    assert set(b.columns) - set(a.columns) == {SUB_GRADE_RANK}
    assert set(DERIVED_FEATURES) <= set(a.columns)
    assert not {"sub_grade", "earliest_cr_line", schema.DATE_COL} & set(b.columns)
    assert list(b.columns) == list(
        FeatureBuilder(include_lc_rating=True).fit(frame).get_feature_names_out()
    )


def test_categoricals_use_fixed_categories_and_null_unknowns(applications):
    frame, _ = applications
    frame = frame.head(3).assign(purpose=["car", "not_a_purpose", None])
    out = FeatureBuilder().fit_transform(frame)

    assert list(out["purpose"].cat.categories) == list(schema.PURPOSES)
    assert out["purpose"].tolist()[0] == "car"
    assert out["purpose"].isna().tolist()[1:] == [True, True]


def test_sub_grade_rank_and_derived_values(applications):
    frame, _ = applications
    frame = frame.head(2).assign(
        sub_grade=["A1", "G5"],
        issue_d=pd.to_datetime(["2015-06-01", "2015-06-01"]),
        earliest_cr_line=pd.to_datetime(["2005-01-01", None]),
        loan_amnt=[10_000.0, 10_000.0],
        annual_inc=[50_000.0, 0.0],
    )
    out = FeatureBuilder(include_lc_rating=True).fit_transform(frame)

    assert out[SUB_GRADE_RANK].tolist() == [0.0, 34.0]
    assert out["credit_history_months"].iloc[0] == 125
    assert np.isnan(out["credit_history_months"].iloc[1])
    assert out["loan_to_income"].iloc[0] == pytest.approx(0.2)
    assert np.isnan(out["loan_to_income"].iloc[1])


def test_same_input_gives_same_features_row_by_row(applications):
    """Scoring one row must equal scoring it inside a batch (no batch-level state)."""
    frame, _ = applications
    builder = FeatureBuilder(include_lc_rating=True).fit(frame)
    batch = builder.transform(frame.head(50))
    single = builder.transform(frame.iloc[[7]])
    pd.testing.assert_frame_equal(single, batch.iloc[[7]])


def test_missing_column_is_reported(applications):
    frame, _ = applications
    with pytest.raises(KeyError, match="fico_range_low"):
        FeatureBuilder().fit_transform(frame.drop(columns="fico_range_low"))


def test_monotone_directions_reference_real_features(applications):
    frame, _ = applications
    names = list(FeatureBuilder(include_lc_rating=True).fit(frame).get_feature_names_out())
    assert set(MONOTONE_DIRECTIONS) <= set(names)
    assert monotone_constraints(names) == MONOTONE_DIRECTIONS
