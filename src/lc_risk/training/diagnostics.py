"""Where is the performance ceiling? Diagnostics on train/validation only.

1. Learning curve: validation AUC when training on 10%-100% of the training rows.
2. Feature engineering: validation AUC after adding ten hand-built ratio features.
3. Tuning surface: spread of validation AUC across the Optuna trials.
4. Information: what LC's sub_grade adds (Model B vs A), read from the run.

If more rows, more features, and more tuning each add little while new information
(LC's grade) adds more, the ceiling is the information in the inputs, not the model.

    uv run python -m lc_risk.training.diagnostics --run v2-release
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from lc_risk import config
from lc_risk.data.cohort import load_cohort
from lc_risk.features import FeatureBuilder, monotone_constraints
from lc_risk.training.estimators import SEED

log = logging.getLogger(__name__)

LEARNING_CURVE_FRACTIONS = (0.1, 0.25, 0.5, 1.0)


def candidate_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Ratio features commonly built by hand for credit models (not used in production)."""
    income = raw["annual_inc"].where(raw["annual_inc"] > 0)
    accounts = raw["total_acc"].where(raw["total_acc"] > 0)
    # payment on the requested amount at a fixed 12% APR benchmark, independent of LC's price
    payment = raw["loan_amnt"] * (0.01 / (1 - 1.01**-36))
    derogatory = ["delinq_2yrs", "pub_rec", "collections_12_mths_ex_med",
                  "chargeoff_within_12_mths", "num_accts_ever_120_pd", "tax_liens"]  # fmt: skip
    return pd.DataFrame(
        {
            "revol_bal_to_inc": raw["revol_bal"] / income,
            "bureau_debt_to_inc": raw["total_bal_ex_mort"] / income,
            "loan_to_rev_limit": raw["loan_amnt"]
            / raw["total_rev_hi_lim"].where(raw["total_rev_hi_lim"] > 0),
            "open_to_buy_to_inc": raw["bc_open_to_buy"] / income,
            "dti_after_loan": raw["dti"] + 100 * payment / (income / 12),
            "open_share": raw["open_acc"] / accounts,
            "recent_accts_share": raw["acc_open_past_24mths"] / accounts,
            "inq_per_open": raw["inq_last_6mths"] / (raw["open_acc"] + 1),
            "derog_count": raw[derogatory].sum(axis=1, min_count=1),
            "emp_length_missing": raw["emp_length"].isna().astype(float),
        },
        index=raw.index,
    )


def _validation_auc(params, X_train, y_train, X_valid, y_valid) -> tuple[float, int]:
    model = xgb.XGBClassifier(
        **{k: v for k, v in params.items() if k != "n_estimators"},
        n_estimators=3000,
        early_stopping_rounds=100,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        monotone_constraints=monotone_constraints(list(X_train.columns)),
        eval_metric="auc",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    return float(model.best_score), int(model.best_iteration + 1)


def run_diagnostics(run_dir) -> dict:
    run = json.loads((run_dir / "run.json").read_text())
    params = run["validation"]["xgboost_A"]["params"]
    train, valid = load_cohort("train"), load_cohort("validation")
    builder = FeatureBuilder(include_lc_rating=False).fit(train)
    X_train, X_valid = builder.transform(train), builder.transform(valid)
    y_train, y_valid = train[config.TARGET].to_numpy(), valid[config.TARGET].to_numpy()

    curve = []
    order = np.random.default_rng(SEED).permutation(len(X_train))
    for fraction in LEARNING_CURVE_FRACTIONS:
        idx = order[: int(fraction * len(order))]
        auc, trees = _validation_auc(params, X_train.iloc[idx], y_train[idx], X_valid, y_valid)
        curve.append({"fraction": fraction, "rows": len(idx), "auc": auc, "trees": trees})
        log.info("learning curve %4.0f%%: auc=%.4f", 100 * fraction, auc)

    engineered_auc, _ = _validation_auc(
        params,
        pd.concat([X_train, candidate_features(train)], axis=1),
        y_train,
        pd.concat([X_valid, candidate_features(valid)], axis=1),
        y_valid,
    )
    base_auc = curve[-1]["auc"]
    log.info("engineered features: %.4f vs %.4f", engineered_auc, base_auc)

    trials = pd.read_csv(run_dir / "optuna_trials_A.csv")
    complete = trials.loc[trials["state"] == "COMPLETE", "value"].sort_values(ascending=False)
    best_trial = int(trials.loc[trials["value"].idxmax(), "number"])

    result = {
        "learning_curve": curve,
        "auc_gain_per_doubling": float(
            np.polyfit(np.log2([c["rows"] for c in curve]), [c["auc"] for c in curve], 1)[0]
        ),
        "feature_engineering": {
            "base_auc": base_auc,
            "with_candidates_auc": engineered_auc,
            "gain": engineered_auc - base_auc,
            "candidates": list(candidate_features(valid.head(1)).columns),
        },
        "tuning": {
            "trials": len(trials),
            "completed": int(len(complete)),
            "best_trial": best_trial,
            "top5_spread": float(complete.iloc[0] - complete.iloc[min(4, len(complete) - 1)]),
        },
        "lc_rating_gain_validation": run["validation"]["xgboost_B"]["auc"]
        - run["validation"]["xgboost_A"]["auc"],
    }
    (run_dir / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_diagnostics(config.ARTIFACTS_DIR / "runs" / args.run)


if __name__ == "__main__":
    main()
