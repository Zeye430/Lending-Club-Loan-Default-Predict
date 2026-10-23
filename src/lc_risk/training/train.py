"""Tune, fit, and calibrate the candidate models using the train and validation splits only.

For each input variant (A: application + bureau; B: A + LC sub_grade):
- logistic regression baseline, C chosen on validation AUC;
- XGBoost with monotone constraints, tuned by Optuna on validation AUC (early stopping and
  median pruning), then refit at the best iteration;
- the same XGBoost parameters without constraints, to price the constraints;
- Platt calibration of each model on the validation year.

The test split is never loaded here. Outputs go to artifacts/runs/<run_id>/.

    uv run python -m lc_risk.training.train --trials 30
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import optuna
import pandas as pd
import xgboost as xgb

from lc_risk import config, schema
from lc_risk.data.cohort import load_cohort
from lc_risk.evaluation import approval_curve, calibration, discrimination
from lc_risk.features import FeatureBuilder
from lc_risk.model import RiskModel, sigmoid
from lc_risk.provenance import git_revision, package_versions, source_digest
from lc_risk.training.estimators import SEED, logistic_pipeline, xgboost_pipeline
from lc_risk.training.reporting import md_table, plot_approval_curves, plot_calibration

log = logging.getLogger(__name__)

VARIANTS = {"A": False, "B": True}  # variant -> include LC sub_grade
LOGISTIC_C_GRID = (0.01, 0.1, 1.0)
MAX_TREES = 3000
EARLY_STOPPING = 100
PRUNING_PERIOD = 50


class _PruneOnValidation(xgb.callback.TrainingCallback):
    """Report validation AUC to Optuna every few rounds and stop hopeless trials early."""

    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def after_iteration(self, model, epoch, evals_log) -> bool:
        if epoch % PRUNING_PERIOD == 0:
            self.trial.report(evals_log["validation_0"]["auc"][-1], epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return False


def search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
    }


def tune_xgboost(include_lc_rating, train, valid, n_trials) -> tuple[dict, pd.DataFrame]:
    """Return the best params (with n_estimators fixed at the best iteration) and the trial log."""
    builder = FeatureBuilder(include_lc_rating=include_lc_rating).fit(train)
    X_train, X_valid = builder.transform(train), builder.transform(valid)
    y_train, y_valid = train[config.TARGET], valid[config.TARGET]

    def objective(trial: optuna.Trial) -> float:
        params = search_space(trial)
        model = xgboost_pipeline(include_lc_rating, {**params, "n_estimators": MAX_TREES})
        estimator = model.named_steps["model"]
        estimator.set_params(
            early_stopping_rounds=EARLY_STOPPING, callbacks=[_PruneOnValidation(trial)]
        )
        started = time.time()
        estimator.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        trial.set_user_attr("best_iteration", int(estimator.best_iteration))
        log.info(
            "  trial %2d  auc=%.4f  trees=%4d  %.0fs",
            trial.number, estimator.best_score, estimator.best_iteration + 1, time.time() - started,
        )  # fmt: skip
        return float(estimator.best_score)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=8),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )
    study.optimize(objective, n_trials=n_trials)
    best = {**study.best_params, "n_estimators": study.best_trial.user_attrs["best_iteration"] + 1}
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    return best, trials


def fit_logistic(include_lc_rating, train, valid) -> tuple[RiskModel, float]:
    best = None
    for C in LOGISTIC_C_GRID:
        pipeline = logistic_pipeline(include_lc_rating, C=C).fit(train, train[config.TARGET])
        auc = discrimination(valid[config.TARGET], pipeline.predict_proba(valid)[:, 1])["auc"]
        log.info("  logistic C=%-5s auc=%.4f", C, auc)
        if best is None or auc > best[2]:
            best = (pipeline, C, auc)
    pipeline, C, _ = best
    return RiskModel(pipeline).with_calibration(valid, valid[config.TARGET]), C


def validation_metrics(model: RiskModel, valid: pd.DataFrame) -> dict:
    y = valid[config.TARGET].to_numpy()
    raw = sigmoid(model.raw_log_odds(valid))
    return {
        **discrimination(y, raw),
        "raw_calibration": calibration(y, raw),
        "platt": {"slope": model.calibrator.slope, "intercept": model.calibrator.intercept},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Captured before training starts, so later edits to the tree cannot leak into the record.
    revision = {**git_revision(), "source_sha256": source_digest()}
    if revision["dirty"]:
        log.warning("working tree has uncommitted changes; this run is not exactly reproducible")

    run_dir = config.ARTIFACTS_DIR / "runs" / args.run_id
    (run_dir / "models").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)

    train, valid = load_cohort("train"), load_cohort("validation")
    log.info("train %s rows, validation %s rows", f"{len(train):,}", f"{len(valid):,}")

    results: dict[str, dict] = {}
    for variant in args.variants:
        include = VARIANTS[variant]
        log.info("variant %s (sub_grade %s)", variant, "included" if include else "excluded")

        logistic, C = fit_logistic(include, train, valid)
        joblib.dump(logistic, run_dir / "models" / f"logistic_{variant}.joblib")
        results[f"logistic_{variant}"] = {"params": {"C": C}, **validation_metrics(logistic, valid)}

        params, trials = tune_xgboost(include, train, valid, args.trials)
        trials.to_csv(run_dir / f"optuna_trials_{variant}.csv", index=False)

        for monotone in (True, False):
            pipeline = xgboost_pipeline(include, params, monotone=monotone)
            pipeline.fit(train, train[config.TARGET])
            model = RiskModel(pipeline).with_calibration(valid, valid[config.TARGET])
            name = f"xgboost_{variant}" + ("" if monotone else "_unconstrained")
            joblib.dump(model, run_dir / "models" / f"{name}.joblib")
            results[name] = {"params": params, **validation_metrics(model, valid)}
            log.info("  %s  validation auc=%.4f", name, results[name]["auc"])

    run = {
        "run_id": args.run_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": revision,
        "libraries": package_versions("scikit-learn", "xgboost", "optuna", "pandas", "numpy"),
        "cohort_manifest": json.loads(config.COHORT_MANIFEST.read_text())["cohort_file"],
        "optuna_trials_per_variant": args.trials,
        "validation": results,
    }
    (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    write_validation_report(run_dir, run, valid)
    log.info("run written to %s", run_dir)


def _years(split: str) -> str:
    years = config.SPLIT_YEARS[split]
    return f"{years[0]}–{years[-1]}" if len(years) > 1 else str(years[0])


def write_validation_report(run_dir: Path, run: dict, valid: pd.DataFrame) -> None:
    results = run["validation"]
    y = valid[config.TARGET].to_numpy()
    rows = [
        {
            "model": name,
            "AUC": r["auc"],
            "Gini": r["gini"],
            "KS": r["ks"],
            "mean PD (raw)": r["raw_calibration"]["mean_predicted"],
            "ECE (raw)": r["raw_calibration"]["ece"],
        }
        for name, r in results.items()
    ]
    grade_rank = valid["sub_grade"].map({g: i for i, g in enumerate(schema.SUB_GRADES)})

    scores, raw_curves = {"LC sub_grade": grade_rank.to_numpy(dtype=float)}, {}
    for name in ("xgboost_A", "xgboost_B"):
        if name in results:
            model = joblib.load(run_dir / "models" / f"{name}.joblib")
            scores[name] = model.predict_pd(valid)
            raw_curves[name + " (raw)"] = sigmoid(model.raw_log_odds(valid))
    plot_approval_curves(y, scores, run_dir / "figures" / "approval_validation.png",
                         "Validation 2015: bad rate vs approval rate")  # fmt: skip
    plot_calibration(y, raw_curves, run_dir / "figures" / "calibration_raw_validation.png",
                     "Validation 2015, before calibration")  # fmt: skip

    at_80 = {
        label: approval_curve(y, s, approval_rates=[0.8])["bad_rate"].iloc[0]
        for label, s in scores.items()
    }
    lines = [
        f"# Validation report — run `{run['run_id']}`",
        "",
        f"Train {_years('train')}, validation {_years('validation')}. "
        "The test year was not used. Calibration is fit on the validation year, so calibration "
        "metrics here are shown **before** calibration; calibrated metrics are only meaningful "
        "on the test year.",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        f"Validation default rate: {y.mean():.4f}. Optuna trials per variant: "
        f"{run['optuna_trials_per_variant']}.",
        "",
        "## Bad rate at 80% approval",
        "",
        md_table(pd.DataFrame({"ranking": list(at_80), "bad rate": list(at_80.values())})),
        "",
        "![approval](figures/approval_validation.png)",
        "![calibration](figures/calibration_raw_validation.png)",
    ]
    (run_dir / "validation_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
