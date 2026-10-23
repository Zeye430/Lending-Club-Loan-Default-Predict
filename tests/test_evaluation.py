import numpy as np
import pytest

from lc_risk.evaluation import (
    approval_curve,
    bootstrap_auc_difference,
    calibration,
    discrimination,
    realized_return,
)

rng = np.random.default_rng(0)
P = rng.uniform(0.01, 0.6, 20_000)
Y = (rng.random(P.size) < P).astype(int)


def test_discrimination_relations():
    m = discrimination(Y, P)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert 0.5 < m["auc"] < 1
    assert 0 < m["ks"] < 1
    assert discrimination(Y, -P)["auc"] == pytest.approx(1 - m["auc"])


def test_calibrated_scores_have_small_ece_and_shifted_scores_do_not():
    assert calibration(Y, P)["ece"] < 0.01
    shifted = calibration(Y, P * 0.5)
    assert shifted["ece"] > 0.05
    assert shifted["mean_predicted"] < shifted["observed_rate"]


def test_approval_curve_bad_rate_rises_with_approval_rate():
    curve = approval_curve(Y, P)
    assert curve["bad_rate"].is_monotonic_increasing
    assert curve["bad_rate"].iloc[-1] == pytest.approx(Y.mean())


def test_realized_return_prefers_safe_loans():
    funded = np.full(4, 100.0)
    repaid = np.array([120.0, 115.0, 40.0, 30.0])
    good_first = np.array([0.1, 0.2, 0.8, 0.9])
    assert realized_return(good_first, funded, repaid, 0.5) == pytest.approx(0.175)
    assert realized_return(-good_first, funded, repaid, 0.5) == pytest.approx(-0.65)
    assert realized_return(good_first, funded, repaid, 1.0) == pytest.approx(-0.2375)


def test_bootstrap_detects_a_real_improvement():
    noisy = P + rng.normal(0, 0.3, P.size)
    result = bootstrap_auc_difference(Y, noisy, P, n_boot=50)
    assert result["ci_low"] > 0
    assert result["ci_low"] <= result["difference"] <= result["ci_high"]


def test_realized_return_rejects_missing_amounts():
    with pytest.raises(ValueError, match="finite"):
        realized_return(
            np.array([0.1, 0.2]), np.array([100.0, np.nan]), np.array([110.0, 90.0]), 1.0
        )
