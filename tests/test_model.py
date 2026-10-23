import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from lc_risk.model import PlattCalibrator, RiskModel, logit, sigmoid
from lc_risk.training.estimators import logistic_pipeline, xgboost_pipeline
from tests.conftest import make_applications

PARAMS = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.2}


@pytest.fixture(scope="module")
def fitted():
    X, y = make_applications(4_000, seed=1)
    pipeline = xgboost_pipeline(include_lc_rating=True, params=PARAMS, n_jobs=1).fit(X, y)
    X_cal, y_cal = make_applications(2_000, seed=2)
    return RiskModel(pipeline).with_calibration(X_cal, y_cal), X_cal, y_cal


def test_platt_recovers_known_shift():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=50_000)
    y = (rng.random(raw.size) < sigmoid(0.5 * raw - 1.0)).astype(int)
    cal = PlattCalibrator.fit(raw, y)
    assert cal.slope == pytest.approx(0.5, abs=0.05)
    assert cal.intercept == pytest.approx(-1.0, abs=0.05)


def test_calibration_keeps_ranking_and_matches_base_rate(fitted):
    model, X, y = fitted
    raw = sigmoid(model.raw_log_odds(X))
    pd_ = model.predict_pd(X)
    assert roc_auc_score(y, pd_) == pytest.approx(roc_auc_score(y, raw))
    assert pd_.mean() == pytest.approx(y.mean(), abs=0.01)


def test_model_learns_the_synthetic_signal(fitted):
    model, X, y = fitted
    assert roc_auc_score(y, model.predict_pd(X)) > 0.7


def test_reason_codes_are_positive_and_sum_consistently(fitted):
    model, X, _ = fitted
    reasons = model.explain(X.head(20), top_k=3)
    assert len(reasons) == 20
    for row in reasons:
        assert len(row) <= 3
        assert all(r.contribution > 0 for r in row)
        assert [r.contribution for r in row] == sorted((r.contribution for r in row), reverse=True)
    names = {r.feature for row in reasons for r in row}
    assert names <= set(model.feature_names)


def test_monotone_constraint_holds_for_fico(fitted):
    model, X, _ = fitted
    row = X.head(1)
    grid = row.loc[row.index.repeat(30)].assign(fico_range_low=np.linspace(600, 850, 30))
    pd_ = model.predict_pd(grid)
    assert np.all(np.diff(pd_) <= 1e-9)


def test_logistic_baseline_runs_with_nulls_and_unknown_categories():
    X, y = make_applications(2_000, seed=3)
    pipeline = logistic_pipeline(include_lc_rating=False).fit(X, y)
    scored = X.head(5).assign(purpose="unknown_purpose")
    assert np.isfinite(logit(pipeline.predict_proba(scored)[:, 1])).all()
    with pytest.raises(NotImplementedError):
        RiskModel(pipeline).explain(scored)
