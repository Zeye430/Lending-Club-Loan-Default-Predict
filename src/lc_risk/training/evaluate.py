"""One-shot out-of-time evaluation of a training run on the test year.

This is the only module that loads the test split. It refuses to run twice on the same run
(pass --force to override), so that test results cannot quietly feed back into model
choices. All selection rules were fixed before this ran; see docs/model_selection.md.

    uv run python -m lc_risk.training.evaluate --run v2-main
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lc_risk import config, schema
from lc_risk.data.cohort import load_cohort
from lc_risk.evaluation import (
    approval_curve,
    bootstrap_auc_difference,
    calibration,
    discrimination,
    realized_return,
)
from lc_risk.model import RiskModel, sigmoid
from lc_risk.monitoring.drift import psi, psi_table
from lc_risk.training.reporting import md_table, plot_approval_curves, plot_calibration

log = logging.getLogger(__name__)

PRIMARY = "xgboost_A"
POLICY_RATES = (0.7, 0.8, 0.9, 1.0)


def grade_rank(frame: pd.DataFrame) -> np.ndarray:
    return frame["sub_grade"].map({g: i for i, g in enumerate(schema.SUB_GRADES)}).to_numpy(float)


def mean_abs_contributions(model: RiskModel, frame: pd.DataFrame, n: int = 20_000) -> pd.Series:
    import xgboost as xgb

    sample = frame.sample(min(n, len(frame)), random_state=0)
    features = model.features.transform(sample)
    contrib = model.estimator.get_booster().predict(
        xgb.DMatrix(features, enable_categorical=True), pred_contribs=True
    )[:, :-1]
    return pd.Series(np.abs(contrib).mean(axis=0), index=features.columns).sort_values(
        ascending=False
    )


def evaluate(run_dir: Path, force: bool = False, cohort_path: Path = config.COHORT_FILE) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    if "test" in run and not force:
        raise SystemExit(f"run {run['run_id']} was already evaluated on test; use --force")

    test = load_cohort("test", cohort_path)
    valid = load_cohort("validation", cohort_path)
    y = test[config.TARGET].to_numpy()
    log.info("test %s rows, default rate %.4f", f"{len(test):,}", y.mean())

    models = {p.stem: joblib.load(p) for p in sorted((run_dir / "models").glob("*.joblib"))}
    pd_ = {name: m.predict_pd(test) for name, m in models.items()}
    raw = {name: sigmoid(m.raw_log_odds(test)) for name, m in models.items()}

    metrics = {
        name: {
            **discrimination(y, p),
            "calibration": calibration(y, p),
            "raw_calibration": calibration(y, raw[name]),
        }
        for name, p in pd_.items()
    }
    lc_grade = grade_rank(test)
    metrics["lc_sub_grade"] = discrimination(y, lc_grade)

    comparisons = {}
    pairs = {
        "B_vs_A_xgboost (value of LC sub_grade)": ("xgboost_A", "xgboost_B"),
        "xgboost_vs_logistic_A": ("logistic_A", "xgboost_A"),
        "unconstrained_vs_monotone_A": ("xgboost_A", "xgboost_A_unconstrained"),
    }
    for label, (a, b) in pairs.items():
        if a in pd_ and b in pd_:
            comparisons[label] = bootstrap_auc_difference(y, pd_[a], pd_[b])
            log.info("%s: %+.4f", label, comparisons[label]["difference"])

    funded, repaid = test["funded_amnt"].to_numpy(), test["total_pymnt"].to_numpy()
    rankings = {
        "LC sub_grade": lc_grade,
        **{k: pd_[k] for k in ("logistic_A", PRIMARY, "xgboost_B") if k in pd_},
    }
    policy = []
    for label, score in rankings.items():
        bad = approval_curve(y, score, approval_rates=POLICY_RATES)["bad_rate"]
        for rate, bad_rate in zip(POLICY_RATES, bad, strict=True):
            policy.append({
                "ranking": label, "approval_rate": rate, "bad_rate": float(bad_rate),
                "realized_return": realized_return(score, funded, repaid, rate),
            })  # fmt: skip

    primary = models[PRIMARY]
    quarters = test["issue_d"].dt.to_period("Q").astype(str)
    by_quarter = [
        {"quarter": q, "loans": int(m.sum()), "auc": discrimination(y[m], pd_[PRIMARY][m])["auc"],
         "mean_pd": float(pd_[PRIMARY][m].mean()), "default_rate": float(y[m].mean())}
        for q in sorted(quarters.unique()) if (m := (quarters == q).to_numpy()).any()
    ]  # fmt: skip
    stability = {
        "score_psi_validation_to_test": psi(
            pd.Series(primary.predict_pd(valid)), pd.Series(pd_[PRIMARY])
        ),
        "input_psi_top": psi_table(valid, test, schema.APPLICATION_FIELDS)
        .head(8)
        .to_dict("records"),
        "by_quarter": by_quarter,
    }
    importance = mean_abs_contributions(primary, test).head(15)

    result = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "test_rows": len(test),
        "test_default_rate": float(y.mean()),
        "metrics": metrics,
        "comparisons": comparisons,
        "policy": policy,
        "stability": stability,
        "importance": importance.round(5).to_dict(),
    }
    run["test"] = result
    (run_dir / "run.json").write_text(json.dumps(run, indent=2, default=float) + "\n")

    figures = run_dir / "figures"
    plot_calibration(y, {f"{PRIMARY} raw": raw[PRIMARY], f"{PRIMARY} calibrated": pd_[PRIMARY]},
                     figures / "calibration_test.png", "Test 2016: calibration")  # fmt: skip
    plot_approval_curves(y, rankings, figures / "approval_test.png",
                         "Test 2016: bad rate vs approval rate")  # fmt: skip
    write_test_report(run_dir, run)
    return result


def write_test_report(run_dir: Path, run: dict) -> None:
    t = run["test"]
    rows = [
        {"model": name, "AUC": m["auc"], "Gini": m["gini"], "KS": m["ks"],
         "Brier": m.get("calibration", {}).get("brier", float("nan")),
         "ECE": m.get("calibration", {}).get("ece", float("nan")),
         "mean PD": m.get("calibration", {}).get("mean_predicted", float("nan"))}
        for name, m in t["metrics"].items()
    ]  # fmt: skip
    comparisons = pd.DataFrame(
        [{"comparison": k, "AUC difference": v["difference"], "95% CI low": v["ci_low"],
          "95% CI high": v["ci_high"]} for k, v in t["comparisons"].items()]
    )  # fmt: skip
    policy_rows = [
        {"ranking": p["ranking"], "approval rate": f"{p['approval_rate']:.0%}",
         "bad rate": f"{p['bad_rate']:.2%}", "realized return": f"{p['realized_return']:+.2%}"}
        for p in t["policy"]
    ]  # fmt: skip
    quarters = pd.DataFrame(t["stability"]["by_quarter"])
    importance = pd.DataFrame(
        {"feature": list(t["importance"]), "mean abs contribution": list(t["importance"].values())}
    )
    lines = [
        f"# Test report — run `{run['run_id']}`",
        "",
        f"Out-of-time test: {t['test_rows']:,} loans issued in {config.SPLIT_YEARS['test'][0]}, "
        f"default rate {t['test_default_rate']:.2%}. Models were tuned on train/validation "
        f"only and calibrated on the validation year. Evaluated once at {t['evaluated_at']}.",
        "",
        "## Discrimination and calibration",
        "",
        md_table(pd.DataFrame(rows)),
        "",
        "`lc_sub_grade` ranks loans by LC's own grade; it has no probability, so no calibration.",
        "",
        "## Paired bootstrap comparisons (200 resamples)",
        "",
        md_table(comparisons),
        "",
        "## Lending policy: approve the lowest-risk share",
        "",
        "Realized return = (total repaid incl. recoveries − funded) / funded, from actual cash "
        "flows over the life of the loans (not annualized).",
        "",
        md_table(pd.DataFrame(policy_rows)),
        "",
        "![approval](figures/approval_test.png)",
        "",
        "## Calibration and stability of the primary model",
        "",
        "![calibration](figures/calibration_test.png)",
        "",
        f"Score PSI, validation → test: {t['stability']['score_psi_validation_to_test']:.4f}",
        "",
        md_table(quarters),
        "",
        "## What drives the primary model (mean |SHAP| in log-odds)",
        "",
        md_table(importance),
    ]
    (run_dir / "test_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="run id under artifacts/runs/")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    evaluate(config.ARTIFACTS_DIR / "runs" / args.run, force=args.force)
    log.info("test report written")


if __name__ == "__main__":
    main()
