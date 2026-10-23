"""Run leakage-aware EDA on the proposed mature 36-month loan cohort."""

from __future__ import annotations

import argparse
import os
from collections import Counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lending-club-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/lending-club-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _common import (
    AUDIT_DIR,
    COHORT_DIR,
    DEFAULT_CHUNK_SIZE,
    EDA_DIR,
    RAW_FILE,
    ensure_output_dirs,
    parse_issue_date,
    parse_percent,
    parse_term_months,
    read_raw_chunks,
    write_csv,
    write_json,
)

NEGATIVE_STATUSES = {"Fully Paid"}
POSITIVE_STATUSES = {"Charged Off", "Default"}
CANDIDATE_TERM = 36
CANDIDATE_START_YEAR = 2012
CANDIDATE_END_YEAR = 2016
DEVELOPMENT_END_YEAR = 2015

CATEGORICAL_FEATURES = [
    "sub_grade",
    "grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "addr_state",
]

NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "fico_range_low",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "acc_open_past_24mths",
    "mort_acc",
    "num_accts_ever_120_pd",
    "pub_rec_bankruptcies",
    "tax_liens",
    "tot_cur_bal",
    "bc_open_to_buy",
    "bc_util",
    "mths_since_last_delinq",
    "mths_since_recent_inq",
]

INTERNAL_DIAGNOSTIC_FEATURES = ["int_rate", "grade", "sub_grade"]

BIN_SPECS = {
    "fico_range_low": [0, 620, 660, 680, 700, 720, 740, 760, 850],
    "dti": [-np.inf, 10, 15, 20, 25, 30, 35, 50, np.inf],
    "annual_inc": [0, 30_000, 50_000, 75_000, 100_000, 150_000, 250_000, np.inf],
    "loan_amnt": [0, 5_000, 10_000, 15_000, 20_000, 25_000, 35_000, np.inf],
    "revol_util": [-np.inf, 20, 40, 60, 80, 100, 150, np.inf],
    "inq_last_6mths": [-np.inf, 0.5, 1.5, 2.5, 3.5, 5.5, np.inf],
}


def auc_rank(y_true: pd.Series, score: pd.Series) -> float | None:
    mask = y_true.notna() & score.notna()
    y = y_true.loc[mask].astype(int)
    score = score.loc[mask]
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0 or negatives == 0:
        return None
    ranks = score.rank(method="average")
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def save_figure(fig: plt.Figure, name: str) -> None:
    path = EDA_DIR / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _prepare_numeric(frame: pd.DataFrame, issue_date: pd.Series) -> pd.DataFrame:
    frame = frame.copy()
    for column in NUMERIC_FEATURES:
        if column not in frame:
            continue
        if column == "revol_util":
            frame[column] = parse_percent(frame[column])
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if "int_rate" in frame:
        frame["int_rate"] = parse_percent(frame["int_rate"])

    if "earliest_cr_line" in frame:
        earliest = parse_issue_date(frame["earliest_cr_line"])
        frame["credit_history_months"] = (
            (issue_date.dt.year - earliest.dt.year) * 12 + (issue_date.dt.month - earliest.dt.month)
        ).clip(lower=0)

    if {"loan_amnt", "annual_inc"} <= set(frame.columns):
        frame["loan_to_income"] = frame["loan_amnt"] / frame["annual_inc"].replace(0, np.nan)

    grade_map = {grade: index for index, grade in enumerate("ABCDEFG", start=1)}
    if "grade" in frame:
        frame["grade_score"] = frame["grade"].map(grade_map)
    if "sub_grade" in frame:
        frame["sub_grade_score"] = frame["sub_grade"].str[0].map(
            {g: i for i, g in enumerate("ABCDEFG")}
        ) * 5 + pd.to_numeric(frame["sub_grade"].str[1], errors="coerce")
    return frame


def run_eda(chunk_size: int, max_sample_rows: int) -> None:
    ensure_output_dirs()
    catalog_path = AUDIT_DIR / "column_catalog.csv"
    cohort_summary_path = COHORT_DIR / "candidate_cohort_summary.json"
    proposed_split_path = COHORT_DIR / "proposed_time_split.csv"
    if (
        not catalog_path.exists()
        or not cohort_summary_path.exists()
        or not proposed_split_path.exists()
    ):
        raise FileNotFoundError(
            "Run scripts/02_validate_raw_data.py, 03_build_column_catalog.py, "
            "and 04_analyze_target_cohorts.py first."
        )

    catalog = pd.read_csv(catalog_path)
    proposed_split = pd.read_csv(proposed_split_path)

    candidate_features = catalog.loc[
        catalog["model_b_role"].isin(["candidate_feature", "candidate_with_review"]),
        "column",
    ].tolist()

    header = pd.read_csv(RAW_FILE, nrows=0, compression=None)
    available = set(header.columns)
    requested = set(candidate_features)
    requested.update(CATEGORICAL_FEATURES)
    requested.update(NUMERIC_FEATURES)
    requested.update(INTERNAL_DIAGNOSTIC_FEATURES)
    requested.update(["issue_d", "term", "loan_status", "earliest_cr_line", "id"])
    usecols = sorted(requested & available)

    expected_rows = max(
        int(
            proposed_split.loc[
                proposed_split["issue_year"] <= DEVELOPMENT_END_YEAR,
                "modelable_rows",
            ].sum()
        ),
        1,
    )
    sample_probability = min(1.0, max_sample_rows / expected_rows)

    category_counts: Counter[tuple[str, str, int]] = Counter()
    binned_counts: Counter[tuple[str, str, int]] = Counter()
    period_counts: Counter[tuple[str, str, int]] = Counter()
    missing_counts: Counter[tuple[str, int]] = Counter()
    target_counts: Counter[int] = Counter()
    development_target_counts: Counter[int] = Counter()
    sample_parts: list[pd.DataFrame] = []
    modelable_rows = 0
    development_rows = 0

    for chunk_number, chunk in enumerate(
        read_raw_chunks(usecols=usecols, chunk_size=chunk_size), start=1
    ):
        issue_date_all = parse_issue_date(chunk["issue_d"])
        term_all = parse_term_months(chunk["term"])
        status_all = chunk["loan_status"].astype("string")
        year_all = issue_date_all.dt.year

        mask = (
            (term_all == CANDIDATE_TERM)
            & year_all.between(CANDIDATE_START_YEAR, CANDIDATE_END_YEAR)
            & status_all.isin(NEGATIVE_STATUSES | POSITIVE_STATUSES)
        )
        if not mask.any():
            continue

        cohort = chunk.loc[mask].copy()
        issue_date = issue_date_all.loc[mask]
        cohort["target"] = status_all.loc[mask].isin(POSITIVE_STATUSES).astype("uint8")
        cohort["issue_year"] = issue_date.dt.year.astype(int)
        cohort["issue_quarter"] = issue_date.dt.to_period("Q").astype(str)
        cohort = _prepare_numeric(cohort, issue_date)

        modelable_rows += len(cohort)
        for target, count in cohort["target"].value_counts().items():
            target_counts[int(target)] += int(count)

        for period_column in ("issue_year", "issue_quarter"):
            grouped = cohort.groupby([period_column, "target"], dropna=False).size()
            for (period, target), count in grouped.items():
                period_counts[(period_column, str(period), int(target))] += int(count)

        development = cohort.loc[cohort["issue_year"] <= DEVELOPMENT_END_YEAR].copy()
        development_rows += len(development)
        for target, count in development["target"].value_counts().items():
            development_target_counts[int(target)] += int(count)

        for feature in CATEGORICAL_FEATURES:
            if feature not in development:
                continue
            values = development[feature].astype("string").fillna("<MISSING>")
            grouped = (
                pd.DataFrame({"value": values, "target": development["target"]})
                .groupby(["value", "target"], dropna=False)
                .size()
            )
            for (value, target), count in grouped.items():
                category_counts[(feature, str(value), int(target))] += int(count)

        for feature, bins in BIN_SPECS.items():
            if feature not in development:
                continue
            bucket = pd.cut(development[feature], bins=bins, include_lowest=True)
            grouped = (
                pd.DataFrame({"bucket": bucket.astype("string"), "target": development["target"]})
                .groupby(["bucket", "target"], dropna=False)
                .size()
            )
            for (bucket_name, target), count in grouped.items():
                binned_counts[(feature, str(bucket_name), int(target))] += int(count)

        for target_value in (0, 1):
            target_frame = development.loc[development["target"] == target_value]
            for feature in candidate_features:
                if feature in target_frame:
                    missing_counts[(feature, target_value)] += int(
                        target_frame[feature].isna().sum()
                    )

        if sample_probability >= 1:
            sampled = development
        else:
            sampled = development.sample(
                frac=sample_probability,
                random_state=42 + chunk_number,
            )
        sample_columns = [
            column
            for column in (
                NUMERIC_FEATURES
                + [
                    "credit_history_months",
                    "loan_to_income",
                    "int_rate",
                    "grade_score",
                    "sub_grade_score",
                    "target",
                    "issue_year",
                ]
                + CATEGORICAL_FEATURES
            )
            if column in sampled
        ]
        sample_parts.append(sampled[sample_columns].copy())

        if chunk_number % 10 == 0:
            print(f"EDA scanned through {chunk_number * chunk_size:,} raw rows", flush=True)

    if not sample_parts:
        raise ValueError("Candidate cohort is empty; check status and date definitions.")

    sample = pd.concat(sample_parts, ignore_index=True)
    if len(sample) > max_sample_rows:
        sample = sample.sample(n=max_sample_rows, random_state=42).reset_index(drop=True)

    target_summary = pd.DataFrame(
        [
            {
                "target": target,
                "label": "default" if target == 1 else "fully_paid",
                "row_count": count,
                "row_share": count / modelable_rows,
            }
            for target, count in sorted(target_counts.items())
        ]
    )
    write_csv(target_summary, EDA_DIR / "target_distribution.csv")

    development_target_summary = pd.DataFrame(
        [
            {
                "target": target,
                "label": "default" if target == 1 else "fully_paid",
                "row_count": count,
                "row_share": count / development_rows,
            }
            for target, count in sorted(development_target_counts.items())
        ]
    )
    write_csv(
        development_target_summary,
        EDA_DIR / "development_target_distribution.csv",
    )

    period_frame = pd.DataFrame(
        [
            {
                "period_type": period_type,
                "period": period,
                "target": target,
                "row_count": count,
            }
            for (period_type, period, target), count in period_counts.items()
        ]
    )
    period_rates = (
        period_frame.pivot_table(
            index=["period_type", "period"],
            columns="target",
            values="row_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={0: "fully_paid_rows", 1: "default_rows"})
        .reset_index()
    )
    for column in ("fully_paid_rows", "default_rows"):
        if column not in period_rates:
            period_rates[column] = 0
    period_rates["total_rows"] = period_rates["fully_paid_rows"] + period_rates["default_rows"]
    period_rates["default_rate"] = period_rates["default_rows"] / period_rates["total_rows"]
    write_csv(period_rates, EDA_DIR / "default_rate_by_period.csv")

    category_frame = pd.DataFrame(
        [
            {"feature": feature, "value": value, "target": target, "row_count": count}
            for (feature, value, target), count in category_counts.items()
        ]
    )
    category_rates = (
        category_frame.pivot_table(
            index=["feature", "value"],
            columns="target",
            values="row_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={0: "fully_paid_rows", 1: "default_rows"})
        .reset_index()
    )
    for column in ("fully_paid_rows", "default_rows"):
        if column not in category_rates:
            category_rates[column] = 0
    category_rates["total_rows"] = (
        category_rates["fully_paid_rows"] + category_rates["default_rows"]
    )
    category_rates["default_rate"] = category_rates["default_rows"] / category_rates["total_rows"]
    write_csv(category_rates, EDA_DIR / "default_rate_by_category.csv")

    binned_frame = pd.DataFrame(
        [
            {"feature": feature, "bin": bucket, "target": target, "row_count": count}
            for (feature, bucket, target), count in binned_counts.items()
        ]
    )
    binned_rates = (
        binned_frame.pivot_table(
            index=["feature", "bin"],
            columns="target",
            values="row_count",
            aggfunc="sum",
            fill_value=0,
        )
        .rename(columns={0: "fully_paid_rows", 1: "default_rows"})
        .reset_index()
    )
    for column in ("fully_paid_rows", "default_rows"):
        if column not in binned_rates:
            binned_rates[column] = 0
    binned_rates["total_rows"] = binned_rates["fully_paid_rows"] + binned_rates["default_rows"]
    binned_rates["default_rate"] = binned_rates["default_rows"] / binned_rates["total_rows"]
    write_csv(binned_rates, EDA_DIR / "default_rate_by_numeric_bin.csv")

    missing_rows = []
    for feature in candidate_features:
        for target in (0, 1):
            denominator = development_target_counts[target]
            count = missing_counts[(feature, target)]
            missing_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "missing_count": count,
                    "missing_rate": count / denominator if denominator else None,
                }
            )
    missing_by_target = pd.DataFrame(missing_rows)
    write_csv(missing_by_target, EDA_DIR / "missingness_by_target.csv")

    missing_pivot = missing_by_target.pivot(
        index="feature", columns="target", values="missing_rate"
    ).fillna(0)
    for target in (0, 1):
        if target not in missing_pivot:
            missing_pivot[target] = 0.0
    missing_pivot["max_missing_rate"] = missing_pivot[[0, 1]].max(axis=1)
    schema_unavailable = (
        missing_pivot.loc[missing_pivot[[0, 1]].min(axis=1) >= 0.999, [0, 1]]
        .rename(
            columns={
                0: "fully_paid_missing_rate",
                1: "default_missing_rate",
            }
        )
        .rename_axis("feature")
        .reset_index()
    )
    write_csv(
        schema_unavailable,
        EDA_DIR / "schema_unavailable_candidate_features.csv",
    )

    candidate_availability = missing_pivot[[0, 1]].rename(
        columns={
            0: "fully_paid_missing_rate",
            1: "default_missing_rate",
        }
    )
    candidate_availability["unavailable_in_development_cohort"] = (
        candidate_availability.min(axis=1) >= 0.999
    )
    candidate_availability = candidate_availability.rename_axis("feature").reset_index()
    candidate_availability = candidate_availability.merge(
        catalog[["column", "model_a_role", "model_b_role"]],
        left_on="feature",
        right_on="column",
        how="left",
    ).drop(columns="column")
    write_csv(
        candidate_availability,
        EDA_DIR / "candidate_feature_availability.csv",
    )

    numeric_columns = [
        column
        for column in (NUMERIC_FEATURES + ["credit_history_months", "loan_to_income", "int_rate"])
        if column in sample
    ]
    numeric_summary = (
        sample[numeric_columns]
        .describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        .T.reset_index(names="feature")
    )
    numeric_summary.insert(1, "basis", f"sample_n={len(sample)}")
    write_csv(numeric_summary, EDA_DIR / "numeric_summary_sample.csv")

    target_numeric_parts = []
    for target, group in sample.groupby("target"):
        summary = group[numeric_columns].agg(["count", "mean", "median", "std"]).T
        summary = summary.reset_index(names="feature")
        summary.insert(1, "target", int(target))
        target_numeric_parts.append(summary)
    write_csv(
        pd.concat(target_numeric_parts, ignore_index=True),
        EDA_DIR / "numeric_summary_by_target_sample.csv",
    )

    corr_columns = [column for column in numeric_columns if sample[column].nunique() > 1]
    correlation = sample[corr_columns + ["target"]].corr(method="spearman")
    target_correlation = (
        correlation["target"]
        .drop("target")
        .sort_values(key=abs, ascending=False)
        .rename("spearman_correlation")
        .rename_axis("feature")
        .reset_index()
    )
    write_csv(target_correlation, EDA_DIR / "correlation_with_target_sample.csv")
    write_csv(
        correlation.reset_index(names="feature"),
        EDA_DIR / "numeric_correlation_matrix_sample.csv",
    )

    auc_scores = {
        "sub_grade": sample.get("sub_grade_score"),
        "grade": sample.get("grade_score"),
        "int_rate": sample.get("int_rate"),
        "fico_range_low_inverse": -sample["fico_range_low"] if "fico_range_low" in sample else None,
        "dti": sample.get("dti"),
        "revol_util": sample.get("revol_util"),
        "loan_to_income": sample.get("loan_to_income"),
    }
    auc_rows = []
    for feature, score in auc_scores.items():
        if score is None:
            continue
        auc_rows.append(
            {
                "feature": feature,
                "single_feature_auc": auc_rank(sample["target"], score),
                "basis": f"sample_n={len(sample)}",
            }
        )
    auc_frame = pd.DataFrame(auc_rows).sort_values("single_feature_auc", ascending=False)
    write_csv(auc_frame, EDA_DIR / "single_feature_auc_sample.csv")

    sns.set_theme(style="whitegrid")
    year_rates = period_rates.loc[period_rates["period_type"] == "issue_year"].copy()
    year_rates["period"] = pd.to_numeric(year_rates["period"])
    year_rates = year_rates.sort_values("period")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(year_rates["period"], year_rates["default_rate"], marker="o")
    ax.set(
        title="Candidate-cohort default rate by issue year",
        xlabel="Issue year (2016 is the future test vintage)",
        ylabel="Default rate",
    )
    ax.set_xticks(year_rates["period"])
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    save_figure(fig, "01_default_rate_by_year.png")

    for feature, filename, top_n in [
        ("sub_grade", "02_default_rate_by_sub_grade.png", 35),
        ("purpose", "03_default_rate_by_purpose.png", 20),
    ]:
        plot_data = category_rates.loc[category_rates["feature"] == feature].copy()
        if plot_data.empty:
            continue
        if feature == "sub_grade":
            order = [f"{grade}{number}" for grade in "ABCDEFG" for number in range(1, 6)]
            plot_data["sort_key"] = plot_data["value"].map(
                {value: index for index, value in enumerate(order)}
            )
            plot_data = plot_data.sort_values("sort_key")
        else:
            plot_data = plot_data.nlargest(top_n, "total_rows").sort_values(
                "default_rate", ascending=False
            )
        fig, ax = plt.subplots(figsize=(11, 5))
        sns.barplot(data=plot_data, x="value", y="default_rate", ax=ax, color="#3B6FB6")
        ax.set(title=f"Default rate by {feature}", xlabel=feature, ylabel="Default rate")
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        ax.tick_params(axis="x", rotation=45)
        save_figure(fig, filename)

    distribution_features = [
        feature
        for feature in [
            "fico_range_low",
            "dti",
            "annual_inc",
            "loan_amnt",
            "revol_util",
            "credit_history_months",
        ]
        if feature in sample
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.flat, distribution_features, strict=False):
        plot_frame = sample[[feature, "target"]].dropna().copy()
        if feature in {"annual_inc", "dti", "credit_history_months"}:
            upper_display_limit = plot_frame[feature].quantile(0.99)
            plot_frame = plot_frame.loc[plot_frame[feature] <= upper_display_limit]
        sns.histplot(
            data=plot_frame,
            x=feature,
            hue="target",
            stat="density",
            common_norm=False,
            bins=40,
            element="step",
            ax=ax,
        )
        ax.set_title(feature)
        if feature in {"annual_inc", "dti", "credit_history_months"}:
            ax.set_xlabel(f"{feature} (display capped at sample p99)")
    for ax in axes.flat[len(distribution_features) :]:
        ax.axis("off")
    fig.suptitle("Key feature distributions by outcome")
    fig.tight_layout()
    save_figure(fig, "04_key_numeric_distributions.png")

    top_corr = target_correlation.head(15)["feature"].tolist()
    if top_corr:
        heatmap = correlation.loc[top_corr + ["target"], top_corr + ["target"]]
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(heatmap, cmap="RdBu_r", center=0, ax=ax)
        ax.set_title("Spearman correlation: strongest target associations")
        save_figure(fig, "05_correlation_heatmap.png")

    missing_plot_pool = missing_pivot.loc[missing_pivot["max_missing_rate"] < 0.999]
    missing_plot = missing_plot_pool.nlargest(20, "max_missing_rate").drop(
        columns="max_missing_rate"
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    missing_plot.sort_values(0).plot.barh(ax=ax)
    ax.set(
        title="Missingness by outcome in the development cohort",
        xlabel="Missing rate",
    )
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(["Fully paid", "Default"])
    save_figure(fig, "06_missingness_by_target.png")

    default_rate = target_counts[1] / modelable_rows
    development_default_rate = development_target_counts[1] / development_rows
    top_auc = auc_frame.iloc[0].to_dict() if not auc_frame.empty else {}
    summary = {
        "candidate_rows": modelable_rows,
        "default_rows": target_counts[1],
        "default_rate": default_rate,
        "development_years": f"{CANDIDATE_START_YEAR}-{DEVELOPMENT_END_YEAR}",
        "development_rows": development_rows,
        "development_default_rows": development_target_counts[1],
        "development_default_rate": development_default_rate,
        "sample_rows_for_distribution_analysis": len(sample),
        "sample_probability": sample_probability,
        "schema_unavailable_candidate_feature_count": len(schema_unavailable),
        "schema_unavailable_candidate_features": schema_unavailable["feature"].tolist(),
        "highest_single_feature_auc": top_auc,
        "top_absolute_spearman_correlations": target_correlation.head(10).to_dict(orient="records"),
    }
    write_json(summary, EDA_DIR / "eda_summary.json")
    write_markdown_summary(
        summary,
        auc_frame,
        target_correlation,
        missing_pivot,
        schema_unavailable,
    )

    print("EDA complete:")
    print(f"  candidate rows : {modelable_rows:,}")
    print(f"  default rate   : {default_rate:.2%}")
    print(f"  development    : {development_rows:,} rows through {DEVELOPMENT_END_YEAR}")
    print(f"  plot sample    : {len(sample):,}")
    print(f"  output         : {EDA_DIR}")


def write_markdown_summary(
    summary: dict,
    auc_frame: pd.DataFrame,
    target_correlation: pd.DataFrame,
    missing_pivot: pd.DataFrame,
    schema_unavailable: pd.DataFrame,
) -> None:
    top_auc_lines = "\n".join(
        f"- `{row.feature}`: AUC `{row.single_feature_auc:.4f}`" for row in auc_frame.itertuples()
    )
    top_corr_lines = "\n".join(
        f"- `{row.feature}`: Spearman `{row.spearman_correlation:.4f}`"
        for row in target_correlation.head(10).itertuples()
    )
    missing_delta = missing_pivot.copy()
    if 0 not in missing_delta:
        missing_delta[0] = 0.0
    if 1 not in missing_delta:
        missing_delta[1] = 0.0
    missing_delta["absolute_gap"] = (missing_delta[1] - missing_delta[0]).abs()
    missing_lines = "\n".join(
        f"- `{index}`: fully paid `{row[0]:.1%}`, default `{row[1]:.1%}`"
        for index, row in missing_delta.nlargest(10, "absolute_gap").iterrows()
    )
    unavailable_lines = "\n".join(f"- `{feature}`" for feature in schema_unavailable["feature"])

    content = f"""# Lending Club data-discovery findings

## Candidate cohort

- Population: approved 36-month loans issued from 2012 through 2016.
- Labels: `Fully Paid = 0`; `Charged Off` or `Default = 1`.
- Modelable rows: {summary["candidate_rows"]:,}.
- Default rate: {summary["default_rate"]:.2%}.
- Development rows (2012-2015): {summary["development_rows"]:,}.
- Development default rate: {summary["development_default_rate"]:.2%}.
- Development distribution/correlation sample: {summary["sample_rows_for_distribution_analysis"]:,} rows.

This cohort is proposed for exploration because the 2020Q3 snapshot leaves later and
60-month vintages materially right-censored. Final cohort approval should happen before
the modeling pipeline is implemented. The issue-year plot includes 2016 only to document
the proposed cohort's temporal drift. Feature distributions, missingness-by-outcome,
correlations, and single-feature AUCs use 2012-2015 development data only; 2016 remains
the future out-of-time test vintage.

## Internal-rating signal

{top_auc_lines}

These are single-feature ranking diagnostics, not model results. Model A and Model B must
later use identical splits, algorithms, and tuning budgets. The primary difference should
be only `sub_grade`; `int_rate` belongs in a separate pricing sensitivity.

## Strong sample associations

{top_corr_lines}

Correlation is descriptive and is not a feature-selection rule. Nonlinear models may use
features with weak marginal correlation, and correlated bureau fields should be assessed
inside cross-validation rather than dropped on the full dataset.

## Missingness gaps by outcome

{missing_lines}

Missingness can represent "no prior event," conditional applicability, a schema change,
or a true unknown value. A global 30% drop threshold would mix these cases together.

## Schema availability

The following {len(schema_unavailable)} catalog candidates are completely unavailable in
the 2012-2015 development cohort and therefore cannot enter either primary model for this
experiment:

{unavailable_lines}

This is a cohort-specific availability decision, not a claim that these fields are
intrinsically useless. They may become eligible in a later-vintage experiment whose
outcomes have had enough time to mature.

## Differences from the original repository

1. The original project selected 2014-2017 before proving label maturity. The new audit
   measures maturity by issue year and term before selecting a cohort.
2. The original project retained only completed loans from immature vintages, creating
   outcome-dependent selection. The proposed cohort minimizes unresolved loans first.
3. The original project dropped every column above 30% missingness. The new workflow
   separates structural, conditional, temporal, and ordinary missingness.
4. The original project removed several bureau risk fields such as `tax_liens`,
   `acc_now_delinq`, and `collections_12_mths_ex_med` before testing their incremental value.
5. The original project retained both `sub_grade` and `int_rate`, so it could not isolate
   the value of LC's internal rating. The rebuild defines Model A and Model B explicitly.
6. The original train/validation split cut chronologically ordered rows at 80%, which can
   split the same issue month across sets. The rebuild proposes year-based boundaries.
7. Target-aware EDA is now limited to the candidate development cohort. Test-period
   distributions will be reserved for final drift and generalization evaluation.

## Potential performance improvements to test later

- Restore valid bureau features and assess them with training-only ablation.
- Encode event-history missingness explicitly instead of median-imputing every field.
- Measure schema availability by vintage and exclude features unavailable at deployment.
- Compare Model A with Model B using calibration and fixed-policy operating points, not
  only ROC-AUC.
- Treat 60-month loans as a separate extension because their maturity window and risk
  dynamics differ from 36-month loans.
"""
    path = EDA_DIR / "eda_findings.md"
    temp_path = path.with_suffix(".md.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-sample-rows", type=int, default=200_000)
    args = parser.parse_args()
    run_eda(chunk_size=args.chunk_size, max_sample_rows=args.max_sample_rows)


if __name__ == "__main__":
    main()
