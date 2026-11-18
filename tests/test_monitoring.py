import numpy as np
import pandas as pd
import pytest

from lc_risk.monitoring.performance import (
    actual_to_expected,
    early_default,
    months_on_book_at_last_payment,
)
from lc_risk.monitoring.rules import RULES_BY_METRIC


def loans(issued, last_payment, default):
    return pd.DataFrame(
        {
            "issue_d": pd.to_datetime(issued),
            "last_pymnt_d": pd.to_datetime(last_payment),
            "default": default,
        }
    )


def test_months_on_book_and_early_default():
    frame = loans(
        ["2016-01-01", "2016-01-01", "2016-01-01", "2016-01-01"],
        ["2016-06-01", "2017-03-01", None, "2016-04-01"],
        [1, 1, 1, 0],
    )
    assert months_on_book_at_last_payment(frame).tolist() == [5, 14, 0, 3]
    # paid-off or still-paying loans are never early defaults, however early they stop
    assert early_default(frame).tolist() == [1, 0, 1, 0]
    assert early_default(frame, horizon=24).tolist() == [1, 1, 1, 0]


def test_actual_to_expected():
    assert actual_to_expected(np.array([1, 0, 1]), np.array([0.5, 0.5, 1.0])) == 1.0
    assert np.isnan(actual_to_expected(np.array([1]), np.array([0.0])))  # nothing expected


@pytest.mark.parametrize(
    ("metric", "value", "expected"),
    [
        ("max_input_psi", 0.05, None),
        ("max_input_psi", 0.12, "warning"),
        ("max_input_psi", 0.30, "critical"),
        ("early_ae", 1.05, None),
        ("early_ae", 0.88, "warning"),
        ("early_ae", 1.25, "critical"),
        ("early_auc_drop", -0.01, None),  # improvement is not an alert
        ("early_auc_drop", 0.015, "warning"),
        ("rejected_share", 0.001, None),  # within the contract's 0.5% tolerance
        ("rejected_share", 0.008, "warning"),
        ("rejected_share", 0.03, "critical"),
        ("final_ae", float("nan"), None),  # not observable yet
    ],
)
def test_rules(metric, value, expected):
    assert RULES_BY_METRIC[metric].check(value) == expected


def synthetic_cohort(seed=0, per_month=1_000):
    """Two years of synthetic loans; the second year's last six months drift on FICO.

    Batches must be large: PSI's sampling noise grows as batches shrink, and the maximum
    over ~60 fields would cross 0.1 with no real drift at a few hundred loans per month.
    """
    from tests.conftest import make_applications

    parts = []
    for split, year in (("validation", 2015), ("test", 2016)):
        frame, y = make_applications(12 * per_month, seed=seed + year)
        frame["default"] = y
        frame["split"] = split
        months = np.repeat(np.arange(1, 13), per_month)
        frame["issue_d"] = pd.to_datetime({"year": year, "month": months, "day": 1})
        rng = np.random.default_rng(year)
        stopped = rng.integers(0, 30, len(frame))
        frame["last_pymnt_d"] = frame["issue_d"] + pd.to_timedelta(stopped * 30, unit="D")
        if split == "test":
            late = frame["issue_d"].dt.month > 6
            frame.loc[late, "fico_range_low"] = frame.loc[late, "fico_range_low"] - 60
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def test_replay_detects_injected_drift_and_writes_report(tmp_path):
    from lc_risk.bundle import load_bundle
    from lc_risk.monitoring.simulate import run
    from lc_risk.serving.app import DEFAULT_MODEL_DIR

    result = run(load_bundle(DEFAULT_MODEL_DIR), synthetic_cohort(), tmp_path)
    metrics, first = result["metrics"], result["first"].set_index("metric")

    assert list(metrics["vintage"]) == [f"2016-{m:02d}" for m in range(1, 13)]
    # FICO shifts in July: input drift must alert from that month, not before
    assert first.loc["max_input_psi", "first_warning"] == "2016-07"
    assert metrics.loc[metrics["vintage"] == "2016-07", "max_psi_field"].item() == "fico_range_low"
    # 5% of synthetic values are null, including required fields: the API would reject them
    assert first.loc["rejected_share", "first_critical"] == "2016-01"
    # performance alerts can only become visible after their labels mature
    for metric, lag in (("early_auc_drop", 13), ("final_auc_drop", 40)):
        when = result["alerts"][result["alerts"]["metric"] == metric]
        lags = (pd.PeriodIndex(when["observable_at"], freq="M")
                - pd.PeriodIndex(when["vintage"], freq="M"))  # fmt: skip
        assert all(lag_.n == lag for lag_ in lags)

    report = (tmp_path / "report.md").read_text()
    assert "## Headline" in report and "## Data quality" in report
    assert (tmp_path / "figures" / "alert_timeline.png").exists()
    assert (tmp_path / "monthly_metrics.csv").exists()


def test_quiet_layers_are_reported_when_nothing_fires(tmp_path):
    """Regression: a never-firing metric must count as quiet even though pandas stores NaN."""
    from lc_risk.bundle import load_bundle
    from lc_risk.monitoring.simulate import run
    from lc_risk.serving.app import DEFAULT_MODEL_DIR

    # production = the reference year replayed a year later, so no input can drift
    baseline = synthetic_cohort().query("split == 'validation'")
    replay = baseline.assign(split="test", issue_d=baseline["issue_d"] + pd.DateOffset(years=1))
    run(load_bundle(DEFAULT_MODEL_DIR), pd.concat([baseline, replay]), tmp_path)

    report = (tmp_path / "report.md").read_text()
    assert "No input drift, score drift alert fired" in report
