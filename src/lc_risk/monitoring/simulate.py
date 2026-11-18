"""Replay the test year as production traffic and report when each monitoring layer alerts.

Each 2016 issue month is one batch of scored applications. For every batch it computes
data-quality, input-drift and score-drift metrics (observable the month the loans are
scored), early performance (observable 13 months later) and final performance (observable
40 months later), pooling the trailing three vintages for performance metrics. Alert rules
come from ``lc_risk.monitoring.rules``; the reference period is the model's calibration year.

    uv run python -m lc_risk.monitoring.simulate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lc_risk import config, schema
from lc_risk.bundle import Bundle, load_bundle
from lc_risk.data.cohort import load_cohort
from lc_risk.data.validation import validate_inputs
from lc_risk.monitoring.drift import psi, psi_table
from lc_risk.monitoring.performance import (
    EARLY_LABEL_AVAILABLE_MONTHS,
    FINAL_LABEL_AVAILABLE_MONTHS,
    actual_to_expected,
    early_default,
)
from lc_risk.monitoring.rules import RULES, SEVERITY_ORDER
from lc_risk.serving.scoring import POLICY_CHECKS

log = logging.getLogger(__name__)

REFERENCE_SPLIT = "validation"  # the calibration year
PRODUCTION_SPLIT = "test"
WINDOW = 3  # vintages pooled for performance metrics
DEFAULT_OUT = config.ROOT / "docs" / "monitoring"
LAYERS = ("data quality", "input drift", "score drift", "early performance", "final performance")
OBSERVABLE_AFTER = {  # months after the vintage month when a layer's metrics can be read
    "data quality": 0,
    "input drift": 0,
    "score drift": 0,
    "early performance": EARLY_LABEL_AVAILABLE_MONTHS,
    "final performance": FINAL_LABEL_AVAILABLE_MONTHS,
}


def score(bundle: Bundle, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pd"] = bundle.model.predict_pd(frame)
    out["early"] = early_default(frame)
    out["vintage"] = frame[schema.DATE_COL].dt.to_period("M")
    return out


def _auc(y: pd.Series, p: pd.Series) -> float:
    return float(roc_auc_score(y, p)) if y.nunique() == 2 else float("nan")


def reference_stats(reference: pd.DataFrame) -> dict:
    return {
        "loans": len(reference),
        "early_auc": _auc(reference["early"], reference["pd"]),
        "final_auc": _auc(reference[config.TARGET], reference["pd"]),
        # early defaults per unit of predicted lifetime PD in the reference year, so that
        # expected early defaults = k * sum(PD) and the reference early A/E is 1 by design
        "early_per_pd": float(reference["early"].sum() / reference["pd"].sum()),
        "final_ae": actual_to_expected(reference[config.TARGET], reference["pd"]),
    }


def batch_quality(batch: pd.DataFrame) -> dict:
    violations = sum(v.count for v in validate_inputs(batch, schema.APPLICATION_FIELDS))
    required = [f.name for f in schema.APPLICATION_FIELDS if not f.nullable]
    missing_required = int(batch[required].isna().any(axis=1).sum())
    out_of_policy = np.zeros(len(batch), dtype=bool)
    for field, check, _ in POLICY_CHECKS:
        out_of_policy |= (batch[field].notna() & check(batch)).to_numpy()
    rejected = violations + missing_required
    return {
        "rejected": rejected,
        "rejected_share": rejected / max(len(batch), 1),
        "out_of_policy_share": float(out_of_policy.mean()),
    }


def monthly_metrics(production: pd.DataFrame, reference: pd.DataFrame, ref: dict) -> pd.DataFrame:
    vintages = sorted(production["vintage"].unique())
    rows = []
    for i, vintage in enumerate(vintages):
        batch = production[production["vintage"] == vintage]
        drift = psi_table(reference, batch, schema.APPLICATION_FIELDS)
        pooled = production[production["vintage"].isin(vintages[max(0, i - WINDOW + 1) : i + 1])]
        early_auc = _auc(pooled["early"], pooled["pd"])
        final_auc = _auc(pooled[config.TARGET], pooled["pd"])
        rows.append(
            {
                "vintage": str(vintage),
                "loans": len(batch),
                **batch_quality(batch),
                "max_input_psi": float(drift["psi"].iloc[0]),
                "max_psi_field": drift["field"].iloc[0],
                "score_psi": psi(reference["pd"], batch["pd"]),
                "mean_pd": float(batch["pd"].mean()),
                "early_rate": float(pooled["early"].mean()),
                "early_ae": actual_to_expected(pooled["early"], ref["early_per_pd"] * pooled["pd"]),
                "early_auc": early_auc,
                "early_auc_drop": ref["early_auc"] - early_auc,
                "final_rate": float(pooled[config.TARGET].mean()),
                "final_ae": actual_to_expected(pooled[config.TARGET], pooled["pd"]),
                "final_auc": final_auc,
                "final_auc_drop": ref["final_auc"] - final_auc,
            }
        )
    return pd.DataFrame(rows)


def alerts(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in metrics.iterrows():
        vintage = pd.Period(row["vintage"], freq="M")
        for rule in RULES:
            severity = rule.check(row[rule.metric])
            if severity:
                rows.append(
                    {
                        "vintage": row["vintage"],
                        "observable_at": str(vintage + OBSERVABLE_AFTER[rule.layer]),
                        "layer": rule.layer,
                        "metric": rule.metric,
                        "value": float(row[rule.metric]),
                        "severity": severity,
                    }
                )
    columns = ["vintage", "observable_at", "layer", "metric", "value", "severity"]
    return pd.DataFrame(rows, columns=columns)


def first_alerts(alert_table: pd.DataFrame) -> pd.DataFrame:
    """Earliest calendar month at which each rule fires, per severity."""
    rows = []
    for rule in RULES:
        row = {"metric": rule.metric, "layer": rule.layer}
        for severity in SEVERITY_ORDER:
            hits = alert_table[
                (alert_table["metric"] == rule.metric)
                & (alert_table["severity"].map(SEVERITY_ORDER) >= SEVERITY_ORDER[severity])
            ]
            first = hits.sort_values(["observable_at", "vintage"]).head(1)
            row[f"first_{severity}"] = first["observable_at"].iloc[0] if len(first) else None
            row[f"{severity}_vintage"] = first["vintage"].iloc[0] if len(first) else None
        rows.append(row)
    return pd.DataFrame(rows)


def seen(value) -> str | None:
    """The month an alert became visible, or None. Pandas stores a missing one as NaN."""
    return value if isinstance(value, str) else None


# Early and lifetime versions of the same measurement, compared like for like.
PAIRED_METRICS = {"AUC drop": ("early_auc_drop", "final_auc_drop"),
                  "actual / expected": ("early_ae", "final_ae")}  # fmt: skip


def lead_times(first: pd.DataFrame, severity: str = "warning") -> dict[str, int | None]:
    """Months by which the early metric's first alert precedes the lifetime metric's."""
    when = first.set_index("metric")[f"first_{severity}"]
    leads = {}
    for name, (early, final) in PAIRED_METRICS.items():
        if seen(when.get(early)) and seen(when.get(final)):
            leads[name] = (pd.Period(when[final], freq="M") - pd.Period(when[early], freq="M")).n
        else:
            leads[name] = None
    return leads


def plot_performance(metrics: pd.DataFrame, ref: dict, path: Path) -> None:
    x = pd.PeriodIndex(metrics["vintage"], freq="M").to_timestamp()
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    left.plot(x, metrics["early_auc"], "o-", label="early (12 months on book)")
    left.plot(x, metrics["final_auc"], "s-", label="final (lifetime)")
    for value, color in ((ref["early_auc"], "C0"), (ref["final_auc"], "C1")):
        left.axhline(value, color=color, ls="--", lw=0.9)  # reference
        left.axhline(value - 0.01, color=color, ls=":", lw=0.9)  # warning
        left.axhline(value - 0.02, color=color, ls=":", lw=1.6)  # critical
    left.set_ylim(min(metrics[["early_auc", "final_auc"]].min()) - 0.01,
                  max(ref["early_auc"], ref["final_auc"]) + 0.01)  # fmt: skip
    left.set_title(
        "AUC by vintage, trailing 3 months\n(dashed: 2015 reference; dotted: warning, critical)"
    )
    left.legend(fontsize=8)
    right.plot(x, metrics["early_ae"], "o-", label="early defaults")
    right.plot(x, metrics["final_ae"], "s-", label="lifetime defaults")
    right.axhline(1.0, color="black", lw=0.8)
    right.axhspan(1.10, 1.20, color="orange", alpha=0.12)
    right.axhspan(0.80, 0.90, color="orange", alpha=0.12)
    right.axhspan(1.20, 2.0, color="red", alpha=0.08)
    right.set_ylim(0.85, max(1.3, metrics[["early_ae", "final_ae"]].max().max() + 0.05))
    right.set_title("Actual / expected defaults (bands: warning, critical)")
    right.legend(fontsize=8)
    for ax in (left, right):
        ax.tick_params(axis="x", rotation=45)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_timeline(alert_table: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    start = pd.Period(metrics["vintage"].iloc[0], freq="M")
    end = pd.Period(metrics["vintage"].iloc[-1], freq="M") + FINAL_LABEL_AVAILABLE_MONTHS
    fig, ax = plt.subplots(figsize=(12, 3.2))
    for y, layer in enumerate(LAYERS):
        hits = alert_table[alert_table["layer"] == layer]
        for severity, color in (("warning", "orange"), ("critical", "red")):
            months = hits.loc[hits["severity"] == severity, "observable_at"]
            xs = [(pd.Period(m, freq="M") - start).n for m in months]
            ax.scatter(xs, [y] * len(xs), color=color, s=30, label=severity if y == 3 else None)
    ticks = range(0, (end - start).n + 1, 6)
    ax.set_xticks(list(ticks), [str(start + t) for t in ticks], rotation=45)
    ax.set_yticks(range(len(LAYERS)), LAYERS)
    ax.set_xlim(-1, (end - start).n + 1)
    ax.set_title("When each alert becomes visible (calendar month)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _md(frame: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "—" if np.isnan(v) else f"{v:.3f}"
        return "—" if v is None else str(v)

    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += [
        "| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)
    ]
    return "\n".join(lines)


def missing_required_fields(production: pd.DataFrame) -> dict[str, int]:
    required = [f.name for f in schema.APPLICATION_FIELDS if not f.nullable]
    counts = production[required].isna().sum()
    return {name: int(n) for name, n in counts[counts > 0].items()}


def write_report(out: Path, bundle: Bundle, ref: dict, metrics: pd.DataFrame,
                 alert_table: pd.DataFrame, first: pd.DataFrame,
                 missing_fields: dict[str, int]) -> None:  # fmt: skip
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_performance(metrics, ref, figures / "performance_by_vintage.png")
    plot_timeline(alert_table, metrics, figures / "alert_timeline.png")
    metrics.to_csv(out / "monthly_metrics.csv", index=False)
    (out / "reference.json").write_text(json.dumps(ref, indent=2) + "\n")

    when = first.set_index("metric")
    leads = lead_times(first)
    quiet = [rule.layer for rule in RULES
             if rule.layer in LAYERS[:3] and not seen(when.loc[rule.metric, "first_warning"])]  # fmt: skip

    headline = []
    early, final = PAIRED_METRICS["AUC drop"]
    if leads["AUC drop"] is not None:
        headline.append(
            f"- **Ranking quality (AUC).** The early-label AUC fell past the warning line in data "
            f"visible in **{when.loc[early, 'first_warning']}**. The same drop measured on lifetime "
            f"outcomes became visible only in **{when.loc[final, 'first_warning']}**, "
            f"**{leads['AUC drop']} months later**."
        )
    early, final = PAIRED_METRICS["actual / expected"]
    if seen(when.loc[early, "first_warning"]):
        first_vintage = metrics.iloc[0]
        line = (
            f"- **Default level (actual / expected).** Early defaults ran above expectation from "
            f"the first vintage (A/E {first_vintage['early_ae']:.2f}, visible "
            f"{when.loc[early, 'first_warning']}), peaking at {metrics['early_ae'].max():.2f}. "
            f"Lifetime A/E peaked at only {metrics['final_ae'].max():.2f}"
        )
        if leads["actual / expected"] is not None:
            line += f", first warning visible {when.loc[final, 'first_warning']}"
        line += (
            ". The early ratio points the right way but overstates the size, because 2016 "
            "defaults also came earlier in the loan's life than in 2015."
        )
        headline.append(line)
    if quiet:
        headline.append(
            f"- **No {', '.join(sorted(set(quiet)))} alert fired in any month.** The applications "
            "looked like the reference year while their outcomes did not, so monitoring the "
            "inputs alone would have missed this degradation."
        )

    rejected_total = int(metrics["rejected"].sum())
    quality = (
        f"{rejected_total:,} of {int(metrics['loans'].sum()):,} applications "
        f"({rejected_total / metrics['loans'].sum():.3%}) would be rejected by the API, all for "
        f"missing a required field ({', '.join(f'`{k}`: {v}' for k, v in missing_fields.items())}). "
        "That is within the 0.5% the input contract tolerates, so no alert fired; whether those "
        "fields should become nullable is a product decision."
        if rejected_total
        else "No application would have been rejected by the API."
    )

    columns = {"first_warning": "first warning visible", "warning_vintage": "from vintage",
               "first_critical": "first critical visible", "critical_vintage": "from vintage "}  # fmt: skip
    table = metrics[["vintage", "loans", "rejected", "max_input_psi", "score_psi", "mean_pd",
                     "early_ae", "early_auc", "final_ae", "final_auc"]]  # fmt: skip
    lines = [
        "# Monitoring replay — 2016 as production traffic",
        "",
        "> Generated by `python -m lc_risk.monitoring.simulate`. Do not edit by hand. The rules and",
        "> the response playbook are in [README.md](README.md).",
        "",
        f"Model `{bundle.name}` v{bundle.version}. Reference: the {ref['loans']:,} loans of the "
        f"calibration year ({config.SPLIT_YEARS[REFERENCE_SPLIT][0]}). Production: "
        f"{int(metrics['loans'].sum()):,} loans issued in {config.SPLIT_YEARS[PRODUCTION_SPLIT][0]}, "
        f"replayed one issue month at a time. Performance metrics pool the trailing {WINDOW} "
        "vintages. Early labels are readable 13 months after issue; lifetime labels after 40.",
        "",
        "## Headline",
        "",
        *headline,
        "",
        "## First alert per metric",
        "",
        _md(first.rename(columns=columns)),
        "",
        "![alert timeline](figures/alert_timeline.png)",
        "",
        "## Performance by vintage",
        "",
        f"Reference: early AUC {ref['early_auc']:.3f}, lifetime AUC {ref['final_auc']:.3f}.",
        "",
        "![performance](figures/performance_by_vintage.png)",
        "",
        "## Data quality",
        "",
        quality,
        "",
        "## Monthly metrics",
        "",
        _md(table),
        "",
        "## Caveats",
        "",
        "- The replay reads outcomes from the 2020Q3 snapshot, so it shows what *would* have been "
        "visible at each date; nothing here was observed live.",
        "- The default month is approximated by the last payment date (see "
        "`lc_risk/monitoring/performance.py`).",
        "- The reference year also served for calibration and early stopping, so its AUCs are "
        "slightly optimistic, which makes the AUC-drop alerts slightly more sensitive.",
        "- The alert thresholds are conventional values fixed in `lc_risk/monitoring/rules.py`. "
        "One rule was corrected after the first replay: data quality originally fired on any "
        "rejected application, which contradicted the 0.5% tolerance built into the input "
        "contract.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run(bundle: Bundle, cohort: pd.DataFrame, out: Path) -> dict:
    reference = score(bundle, cohort[cohort["split"] == REFERENCE_SPLIT])
    production = score(bundle, cohort[cohort["split"] == PRODUCTION_SPLIT])
    ref = reference_stats(reference)
    metrics = monthly_metrics(production, reference, ref)
    alert_table = alerts(metrics)
    first = first_alerts(alert_table)
    write_report(out, bundle, ref, metrics, alert_table, first, missing_required_fields(production))
    return {"reference": ref, "metrics": metrics, "alerts": alert_table, "first": first}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir", type=Path, default=config.ROOT / "models" / "lc-default-risk"
    )
    parser.add_argument("--cohort", type=Path, default=config.COHORT_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    cohort = load_cohort(path=args.cohort)
    result = run(load_bundle(args.model_dir), cohort, args.out)
    log.info("first alerts:\n%s", result["first"].to_string(index=False))
    log.info("report written to %s", args.out / "report.md")


if __name__ == "__main__":
    main()
