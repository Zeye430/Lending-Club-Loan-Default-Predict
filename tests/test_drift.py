import numpy as np
import pandas as pd
import pytest

from lc_risk.monitoring.drift import psi, psi_table
from lc_risk.schema import Field

rng = np.random.default_rng(0)


def test_identical_distributions_have_near_zero_psi():
    sample = pd.Series(rng.normal(size=20_000))
    assert psi(sample, sample.sample(frac=1, random_state=1)) == pytest.approx(0, abs=1e-9)


def test_psi_grows_with_the_size_of_the_shift():
    reference = pd.Series(rng.normal(size=20_000))
    small = psi(reference, pd.Series(rng.normal(0.1, size=20_000)))
    large = psi(reference, pd.Series(rng.normal(1.0, size=20_000)))
    assert small < 0.1 < 0.25 < large


def test_change_in_missingness_is_drift():
    reference = pd.Series(rng.normal(size=10_000))
    current = reference.copy()
    current.iloc[:3_000] = np.nan
    assert psi(reference, current) > 0.25


def test_categorical_psi_handles_unseen_categories():
    reference = pd.Series(["a"] * 500 + ["b"] * 500)
    current = pd.Series(["a"] * 500 + ["c"] * 500)
    assert psi(reference, current) > 0.25
    assert psi(reference, reference) == pytest.approx(0)


def test_psi_table_sorts_and_skips_dates():
    reference = pd.DataFrame({"x": rng.normal(size=5_000), "y": rng.normal(size=5_000)})
    current = reference.assign(y=reference["y"] + 1)
    reference["d"] = current["d"] = pd.Timestamp("2015-01-01")
    fields = [Field("x", "numeric", ""), Field("y", "numeric", ""), Field("d", "date", "")]

    table = psi_table(reference, current, fields)
    assert table["field"].tolist() == ["y", "x"]
