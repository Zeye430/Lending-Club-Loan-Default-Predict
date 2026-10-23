import numpy as np
import pandas as pd

from lc_risk.data.parsing import parse_emp_length, parse_month, parse_percent, parse_term_months


def test_parse_month():
    parsed = parse_month(pd.Series(["Dec-2015", "Jan-1999", "garbage", None]))
    assert parsed.iloc[0] == pd.Timestamp("2015-12-01")
    assert parsed.iloc[1] == pd.Timestamp("1999-01-01")
    assert parsed.iloc[2:].isna().all()


def test_parse_term_months():
    parsed = parse_term_months(pd.Series([" 36 months", "60 months", None]))
    assert parsed.tolist()[:2] == [36, 60]
    assert pd.isna(parsed.iloc[2])


def test_parse_percent_handles_strings_and_numbers():
    parsed = parse_percent(pd.Series(["13.3%", " 0.6% ", 24.0, None]))
    assert parsed.dtype == "float64"
    np.testing.assert_array_equal(parsed.to_numpy()[:3], [13.3, 0.6, 24.0])
    assert np.isnan(parsed.iloc[3])


def test_parse_emp_length():
    parsed = parse_emp_length(pd.Series(["< 1 year", "1 year", "10+ years", "n/a", None]))
    np.testing.assert_array_equal(parsed.to_numpy()[:3], [0.0, 1.0, 10.0])
    assert parsed.iloc[3:].isna().all()
