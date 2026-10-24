"""Release one model from an evaluated run into models/<name>/ (committed to the repo).

Writes the bundle (model.joblib + metadata.json), a generated model card, and copies of the
run's validation and test reports with their figures.

    uv run python -m lc_risk.training.release --run v2-main --model xgboost_A
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import joblib

from lc_risk import config, schema
from lc_risk.bundle import save_bundle
from lc_risk.features import MONOTONE_DIRECTIONS

DEFAULT_NAME = "lc-default-risk"
MODELS_DIR = config.ROOT / "models"
REPORTS = ("validation_report.md", "test_report.md")


def build_metadata(
    run: dict,
    model_key: str,
    name: str,
    version: str,
    model,
    diagnostics: dict | None = None,
    changes: str | None = None,
) -> dict:
    test = run["test"]
    policy = [p for p in test["policy"] if p["ranking"] in (model_key, "LC sub_grade")]
    return {
        "name": name,
        "version": version,
        "released_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run_id": run["run_id"],
        "model_key": model_key,
        "target": f"P({config.TARGET}) for a {config.TERM_MONTHS}-month individual LC loan",
        "input_fields": [f.name for f in model.features.input_fields],
        "feature_names": model.feature_names,
        "excluded_fields": {
            k: v
            for k, v in schema.EXCLUDED_FROM_MODEL.items()
            if k not in {f.name for f in model.features.input_fields}
        },
        "changes": changes,
        "monotone_constraints": {
            k: v for k, v in MONOTONE_DIRECTIONS.items() if k in model.feature_names
        },
        "params": run["validation"][model_key]["params"],
        "calibration": run["validation"][model_key]["platt"],
        "validation": {k: v for k, v in run["validation"][model_key].items() if k != "params"},
        "test": {
            "rows": test["test_rows"],
            "default_rate": test["test_default_rate"],
            **test["metrics"][model_key],
        },
        "test_benchmarks": {
            k: {"auc": v["auc"], "gini": v["gini"]}
            for k, v in test["metrics"].items()
            if k != model_key
        },
        "test_comparisons": test["comparisons"],
        "test_policy": policy,
        "test_stability": test["stability"],
        "top_drivers": test["importance"],
        "code": run.get("code"),
        "libraries": run.get("libraries", {}),
        "cohort": run.get("cohort_manifest"),
        "diagnostics": diagnostics,
    }


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _ceiling_section(diag: dict | None) -> list[str]:
    if not diag:
        return []
    curve = " → ".join(f"{c['auc']:.4f} ({c['fraction']:.0%})" for c in diag["learning_curve"])
    fe, tuning = diag["feature_engineering"], diag["tuning"]
    return [
        "## Where the performance ceiling is",
        "",
        "Validation-year diagnostics (`lc_risk.training.diagnostics`):",
        "",
        "| lever | effect on validation AUC |",
        "|---|---|",
        f"| more training rows | {curve}; about {diag['auc_gain_per_doubling']:+.4f} per doubling |",
        f"| {len(fe['candidates'])} hand-built ratio features | {fe['gain']:+.4f} |",
        f"| hyperparameter tuning | best of {tuning['trials']} trials was #{tuning['best_trial']}"
        f" (random warm-up); top-5 spread {tuning['top5_spread']:.4f} |",
        f"| new information: LC's sub_grade | {diag['lc_rating_gain_validation']:+.4f} |",
        "",
        "More rows, more features, and more tuning each add little; new information adds more. "
        "The ceiling is the information in application-time bureau data for an already-approved "
        "population, not the model. Gains would come from new sources (cash-flow data, trended "
        "bureau history, the bureau fields LC added in late 2015) rather than from modeling.",
        "",
    ]


def model_card(meta: dict) -> str:
    t, val = meta["test"], meta["validation"]
    bench = meta["test_benchmarks"]
    comparisons = meta["test_comparisons"]
    b_vs_a = next((v for k, v in comparisons.items() if k.startswith("B_vs_A")), None)
    xgb_vs_lr = comparisons.get("xgboost_vs_logistic_A")

    policy = {(p["ranking"], p["approval_rate"]): p for p in meta["test_policy"]}
    ours, lc = policy.get((meta["model_key"], 0.8)), policy.get(("LC sub_grade", 0.8))
    everyone = policy.get((meta["model_key"], 1.0))

    drivers = "\n".join(
        f"| {name} | {value:.3f} |" for name, value in list(meta["top_drivers"].items())[:10]
    )
    constraints = ", ".join(
        f"`{k}` (higher → {'more' if v > 0 else 'less'} risk)"
        for k, v in meta["monotone_constraints"].items()
    )
    quarters = "\n".join(
        f"| {q['quarter']} | {q['loans']:,} | {q['auc']:.3f} | {_pct(q['mean_pd'])} "
        f"| {_pct(q['default_rate'])} |"
        for q in meta["test_stability"]["by_quarter"]
    )
    first_q, last_q = (
        meta["test_stability"]["by_quarter"][0],
        meta["test_stability"]["by_quarter"][-1],
    )
    cal = t["calibration"]
    excluded = meta.get("excluded_fields") or {}
    code = meta.get("code") or {}
    cohort = meta.get("cohort") or {}

    lines = [
        f"# Model card — {meta['name']} v{meta['version']}",
        "",
        "## What it does",
        "",
        f"Predicts the probability that a {config.TERM_MONTHS}-month individual Lending Club "
        "loan is charged off, from information available when the application is scored: "
        f"{len(meta['input_fields'])} application and credit-bureau fields. It does **not** "
        "use LC's grade or interest rate. Output: a calibrated probability of default (PD) "
        "plus the top risk-increasing factors for that applicant (reason codes).",
        "",
        *[
            f"Accepted by the service but deliberately **not** used by the model: `{k}`. {v}"
            for k, v in excluded.items()
        ],
        "" if excluded else None,
        *(["## Changes in this version", "", meta["changes"], ""] if meta.get("changes") else []),
        f"Algorithm: XGBoost ({meta['params'].get('n_estimators')} trees, max depth "
        f"{meta['params'].get('max_depth')}), with monotone constraints on "
        f"{len(meta['monotone_constraints'])} features, followed by Platt calibration "
        f"(slope {meta['calibration']['slope']:.3f}, intercept "
        f"{meta['calibration']['intercept']:+.3f}).",
        "",
        "## Intended use",
        "",
        "- Ranking and pricing support for 36-month unsecured personal loans similar to LC's "
        "2013–2016 book; portfolio-level expected-loss estimates.",
        "- **Not** for: 60-month loans, joint applications, applicants outside LC's historical "
        "approval range (e.g. FICO < 660, DTI ≥ 40%), or as the sole basis of an adverse "
        "decision without human review.",
        "",
        "## Data",
        "",
        f"- Train: loans issued {config.SPLIT_YEARS['train'][0]}–{config.SPLIT_YEARS['train'][-1]}; "
        f"calibration/validation: {config.SPLIT_YEARS['validation'][0]}; "
        f"out-of-time test: {config.SPLIT_YEARS['test'][0]} ({t['rows']:,} loans, "
        f"{_pct(t['default_rate'])} default rate).",
        "- Cohort definition and evidence: `discovery/reports/data_audit.md` and "
        "`discovery/eda.ipynb`. Selection rules, fixed before testing: `docs/model_selection.md`.",
        "",
        "## Performance on the out-of-time test year",
        "",
        "| | this model | logistic baseline | LC sub-grade ranking |",
        "|---|---|---|---|",
        f"| AUC | {t['auc']:.4f} | {bench.get('logistic_A', {}).get('auc', float('nan')):.4f} "
        f"| {bench.get('lc_sub_grade', {}).get('auc', float('nan')):.4f} |",
        f"| Gini | {t['gini']:.4f} | {bench.get('logistic_A', {}).get('gini', float('nan')):.4f} "
        f"| {bench.get('lc_sub_grade', {}).get('gini', float('nan')):.4f} |",
        f"| KS | {t['ks']:.4f} | | |",
        "",
        f"Validation AUC was {val['auc']:.4f}.",
    ]
    if xgb_vs_lr:
        lines.append(
            f"XGBoost vs the logistic baseline: {xgb_vs_lr['difference']:+.4f} AUC "
            f"(95% CI {xgb_vs_lr['ci_low']:+.4f} to {xgb_vs_lr['ci_high']:+.4f})."
        )
    if b_vs_a:
        lines += [
            "",
            "**Value of LC's grade (ablation).** Adding `sub_grade` to the same model changes "
            f"test AUC by {b_vs_a['difference']:+.4f} (95% CI {b_vs_a['ci_low']:+.4f} to "
            f"{b_vs_a['ci_high']:+.4f}). This model deliberately does not use it; see "
            "`docs/model_selection.md`.",
        ]
    lines += [
        "",
        "## Calibration",
        "",
        f"Test year: mean predicted PD {_pct(cal['mean_predicted'], 2)} vs observed default rate "
        f"{_pct(cal['observed_rate'], 2)}; expected calibration error {cal['ece']:.4f}; "
        f"Brier score {cal['brier']:.4f}.",
        "",
        "![calibration](figures/calibration_test.png)",
        "",
        "By issue quarter of the test year:",
        "",
        "| quarter | loans | AUC | mean PD | default rate |",
        "|---|---|---|---|---|",
        quarters,
        "",
    ]
    if ours and lc and everyone:
        lines += [
            "## Lending impact",
            "",
            "Approving the 80% of test-year applicants with the lowest predicted risk:",
            "",
            "| ranking | default rate among approved | realized return on approved book |",
            "|---|---|---|",
            f"| this model | {_pct(ours['bad_rate'], 2)} | {ours['realized_return']:+.2%} |",
            f"| LC sub-grade | {_pct(lc['bad_rate'], 2)} | {lc['realized_return']:+.2%} |",
            f"| approve everyone | {_pct(everyone['bad_rate'], 2)} | {everyone['realized_return']:+.2%} |",
            "",
            "Realized return uses each loan's actual repayments and recoveries over its life, "
            "not annualized.",
            "",
            "![approval](figures/approval_test.png)",
            "",
        ]
    lines += [
        "## Explainability",
        "",
        f"Monotone constraints: {constraints}.",
        "",
        "Reason codes are exact TreeSHAP contributions computed by XGBoost, scaled by the "
        "calibration slope, so they are in calibrated log-odds.",
        "",
        "| top drivers (mean abs contribution, test year) | |",
        "|---|---|",
        drivers,
        "",
        *_ceiling_section(meta.get("diagnostics")),
        "## Limitations and risks",
        "",
        "- **Approved loans only.** Rejected applicants are absent from the data, so the model "
        "has not learned risk outside LC's approval range (no reject inference).",
        "- **Drift.** Default rates rose within every LC grade from 2013 to 2016, and the "
        "calibration was fit on 2015. Expect under-prediction if the trend continues; "
        "recalibrate on the latest matured vintage.",
        "- **Selection-encoded inputs.** `verification_status` reflects LC choosing to verify "
        "riskier applicants; it is predictive but not causal.",
        "- **Fairness.** The data holds no protected attributes, so disparate impact cannot be "
        "measured directly. "
        + (
            "`addr_state` is excluded from the model; geography can still leak through "
            "correlated bureau fields, so monitor approval and default rates by state."
            if "addr_state" in excluded
            else "`addr_state` can proxy for them; review its contribution before any real use."
        ),
        "- **Serialization.** The model is a pickle; it loads only with the pinned "
        "scikit-learn and xgboost versions (checked at load time).",
        "",
        "## Monitoring",
        "",
        f"Within the test year, AUC fell from {first_q['auc']:.3f} ({first_q['quarter']}) to "
        f"{last_q['auc']:.3f} ({last_q['quarter']}), while the score PSI between validation and "
        f"test was only {meta['test_stability']['score_psi_validation_to_test']:.4f}. **Input and "
        "score drift monitoring alone would not have caught this.** In production, track input "
        "PSI per field and score PSI per batch (alert above 0.25), and compare predicted PD with "
        "the realized default rate and AUC as each vintage's labels mature.",
        "",
        "## Follow-up work",
        "",
        "- Refit on 2013–2015 and recalibrate on the newest matured vintage, with a fresh holdout.",
        "- A separate model for 60-month loans on an earlier cohort.",
        "",
        "## Provenance",
        "",
        f"- Training run `{meta['run_id']}`; code commit `{code.get('commit')}` "
        f"(dirty: {code.get('dirty')}), package source SHA-256 `{code.get('source_sha256')}`",
        f"- Cohort `{cohort.get('name')}` SHA-256 `{cohort.get('sha256')}`",
        f"- Libraries: {', '.join(f'{k} {v}' for k, v in meta['libraries'].items())}",
        "- Full reports: [validation](validation_report.md), [test](test_report.md)",
    ]
    return "\n".join(line for line in lines if line is not None) + "\n"


def release(
    run_dir: Path,
    model_key: str,
    name: str,
    version: str,
    out_root: Path,
    changes: str | None = None,
) -> Path:
    run = json.loads((run_dir / "run.json").read_text())
    if "test" not in run:
        raise SystemExit("evaluate the run on the test year before releasing")

    model = joblib.load(run_dir / "models" / f"{model_key}.joblib")
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    diagnostics_file = run_dir / "diagnostics.json"
    diagnostics = json.loads(diagnostics_file.read_text()) if diagnostics_file.exists() else None
    metadata = save_bundle(
        model,
        build_metadata(run, model_key, name, version, model, diagnostics, changes),
        out_dir,
    )

    shutil.copytree(run_dir / "figures", out_dir / "figures")
    for report in REPORTS:
        if (run_dir / report).exists():
            shutil.copy2(run_dir / report, out_dir / report)
    (out_dir / "model_card.md").write_text(model_card(metadata))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--model", default="xgboost_A")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--version", required=True)
    parser.add_argument("--changes", help="markdown describing what changed in this version")
    args = parser.parse_args()
    out = release(config.ARTIFACTS_DIR / "runs" / args.run, args.model, args.name, args.version,
                  MODELS_DIR, args.changes)  # fmt: skip
    print(f"released {args.model} from run {args.run} -> {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
