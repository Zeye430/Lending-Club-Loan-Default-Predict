"""End-to-end check of the one-shot test evaluation on a synthetic cohort and run."""

import json
import re

import joblib
import numpy as np
import pandas as pd
import pytest

from lc_risk import config
from lc_risk.bundle import load_bundle
from lc_risk.model import RiskModel
from lc_risk.training.estimators import logistic_pipeline, xgboost_pipeline
from lc_risk.training.evaluate import evaluate
from lc_risk.training.release import release
from tests.conftest import make_applications

PARAMS = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.3}
SPLIT_DATES = {"train": "2014-03-01", "validation": "2015-06-01", "test": "2016-01-01"}


def synthetic_split(split, n, seed):
    frame, y = make_applications(n, seed=seed)
    frame[config.TARGET] = y
    frame["split"] = split
    frame["issue_d"] = pd.to_datetime(SPLIT_DATES[split])
    if split == "test":  # spread over two quarters so the by-quarter table has rows
        frame.loc[frame.index[n // 2 :], "issue_d"] = pd.Timestamp("2016-07-01")
    frame["loan_amnt"] = frame["loan_amnt"].fillna(10_000.0)  # never null in the real cohort
    frame["funded_amnt"] = frame["loan_amnt"]
    frame["total_pymnt"] = np.where(y == 1, 0.4, 1.15) * frame["loan_amnt"]
    return frame


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    splits = {s: synthetic_split(s, 1_500, seed) for seed, s in enumerate(SPLIT_DATES)}
    pd.concat(splits.values(), ignore_index=True).to_parquet(root / "cohort.parquet")

    run = root / "run"
    (run / "models").mkdir(parents=True)
    (run / "figures").mkdir()
    train, valid = splits["train"], splits["validation"]
    candidates = {
        "logistic_A": logistic_pipeline(False),
        "xgboost_A": xgboost_pipeline(False, PARAMS, n_jobs=1),
        "xgboost_A_unconstrained": xgboost_pipeline(False, PARAMS, monotone=False, n_jobs=1),
        "xgboost_B": xgboost_pipeline(True, PARAMS, n_jobs=1),
    }
    validation = {}
    for name, pipeline in candidates.items():
        pipeline.fit(train, train[config.TARGET])
        model = RiskModel(pipeline).with_calibration(valid, valid[config.TARGET])
        joblib.dump(model, run / "models" / f"{name}.joblib")
        validation[name] = {
            "params": PARAMS,
            "platt": {"slope": model.calibrator.slope, "intercept": model.calibrator.intercept},
            "auc": 0.8,
        }
    (run / "run.json").write_text(json.dumps({"run_id": "synthetic", "validation": validation}))
    return run, root / "cohort.parquet"


def test_evaluate_writes_results_and_report(run_dir):
    run, cohort = run_dir
    result = evaluate(run, cohort_path=cohort)

    assert set(result["metrics"]) >= {"xgboost_A", "xgboost_B", "logistic_A", "lc_sub_grade"}
    assert result["metrics"]["xgboost_A"]["auc"] > 0.6
    assert len(result["policy"]) == 4 * 4  # four rankings x four approval rates
    assert all(np.isfinite(p["realized_return"]) for p in result["policy"])
    assert [q["quarter"] for q in result["stability"]["by_quarter"]] == ["2016Q1", "2016Q3"]
    assert "B_vs_A_xgboost (value of LC sub_grade)" in result["comparisons"]

    report = (run / "test_report.md").read_text()
    for heading in ("Discrimination", "bootstrap", "Lending policy", "Calibration", "drives"):
        assert heading in report
    assert (run / "figures" / "calibration_test.png").exists()
    assert "test" in json.loads((run / "run.json").read_text())


def test_second_evaluation_is_refused(run_dir):
    run, cohort = run_dir
    with pytest.raises(SystemExit, match="already evaluated"):
        evaluate(run, cohort_path=cohort)


def test_release_produces_a_loadable_bundle_and_card(run_dir, tmp_path):
    run, cohort = run_dir
    out = release(run, "xgboost_A", "synthetic-model", "0.1.0", tmp_path)

    bundle = load_bundle(out)
    sample = pd.read_parquet(cohort).head(5)
    assert bundle.model.predict_pd(sample).shape == (5,)
    assert bundle.metadata["monotone_constraints"]["fico_range_low"] == -1

    card = (out / "model_card.md").read_text()
    for section in (
        "What it does",
        "out-of-time test",
        "Lending impact",
        "Limitations",
        "Provenance",
    ):
        assert section in card
    assert not re.search(r"\bnan\b", card, re.IGNORECASE), "unformatted missing value in card"
    assert (out / "figures" / "approval_test.png").exists()
    assert (out / "test_report.md").exists()


def test_card_includes_ceiling_diagnostics_when_present(run_dir, tmp_path):
    run, _ = run_dir
    diagnostics = {
        "learning_curve": [
            {"fraction": 0.5, "rows": 750, "auc": 0.80, "trees": 10},
            {"fraction": 1.0, "rows": 1500, "auc": 0.81, "trees": 20},
        ],
        "auc_gain_per_doubling": 0.01,
        "feature_engineering": {"gain": 0.001, "candidates": ["a", "b"]},
        "tuning": {"trials": 30, "best_trial": 4, "top5_spread": 0.0009},
        "lc_rating_gain_validation": 0.0105,
    }
    (run / "diagnostics.json").write_text(json.dumps(diagnostics))
    try:
        out = release(run, "xgboost_A", "synthetic-model", "0.1.0", tmp_path)
    finally:
        (run / "diagnostics.json").unlink()

    card = (out / "model_card.md").read_text()
    assert "performance ceiling" in card
    assert "+0.0105" in card
    assert load_bundle(out).metadata["diagnostics"]["tuning"]["best_trial"] == 4
