import numpy as np
import pandas as pd

from lc_risk.data.validation import validate_inputs
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
