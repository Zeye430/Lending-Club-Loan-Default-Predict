import numpy as np
import pandas as pd

from lc_risk.data.validation import check_nullability, validate_inputs
from lc_risk.schema import Field

FIELDS = (
    Field("amount", "numeric", "", ge=0, le=100),
    Field("color", "categorical", "", categories=("red", "blue")),
    Field("opened", "date", ""),
)


def frame(**overrides):
    data = {
        "amount": [10.0, np.nan],
        "color": ["red", None],
        "opened": pd.to_datetime(["2015-01-01", None]),
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_conforming_frame_with_nulls_passes():
    assert validate_inputs(frame(), FIELDS) == []


def test_detects_range_violations():
    problems = validate_inputs(frame(amount=[-1.0, 101.0]), FIELDS)
    assert [p.problem for p in problems] == ["below minimum 0", "above maximum 100"]


def test_detects_unknown_category_and_missing_column():
    problems = validate_inputs(frame(color=["red", "green"]).drop(columns="opened"), FIELDS)
    assert {p.field for p in problems} == {"color", "opened"}
    assert "green" in next(p.problem for p in problems if p.field == "color")


def test_detects_wrong_dtypes():
    problems = validate_inputs(frame(amount=["10", "x"], opened=["2015-01", "x"]), FIELDS)
    assert {p.field for p in problems} == {"amount", "opened"}


def test_nullability_contract():
    fields = (Field("always", "numeric", ""), Field("sometimes", "numeric", "", nullable=True))
    frame = pd.DataFrame({"always": [1.0] * 99 + [np.nan], "sometimes": [np.nan] * 100})
    assert check_nullability(frame, fields, max_missing_rate=0.02) == []
    [problem] = check_nullability(frame, fields, max_missing_rate=0.005)
    assert problem.field == "always" and problem.count == 1
