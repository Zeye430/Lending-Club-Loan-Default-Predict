# Lending Club Default Risk — v2 rebuild

> **Work in progress.** This branch is a solo rebuild of a 5-person course project. The original
> team version (pandas pipeline, Databricks notebooks, Streamlit app, MCP server) is preserved
> at the git tag [`v1-course-project`](../../tree/v1-course-project).

## Why a rebuild

An audit of the raw data and the original project ([`discovery/`](discovery/)) found problems
in how the experiment was designed. The evidence is in the
[raw data audit](discovery/reports/data_audit.md) and the [EDA notebook](discovery/eda.ipynb).

- **Right-censored labels.** The 2017 test vintage was only ~71% resolved in the 2020Q3
  snapshot. Keeping only completed loans skewed the sample toward early defaults and early
  payoffs.
- **Threshold chosen on the test set**, so the reported test metrics were not a true holdout.
- **LC's own rating used as an input** (`sub_grade` + `int_rate`), so the model's incremental
  value over LC's underwriting was never measured.
- **Train/serve skew** in the deployed demo (`issue_year` as a feature, ~25 inputs filled with
  population defaults).

## Data

Source: the Kaggle snapshot of LC loans issued 2007–2020Q3 (2.9M rows, 142 columns).
`python -m lc_risk.data.cohort` streams it into `data/interim/cohort.parquet` and writes a
manifest with raw and output SHA-256 hashes, exclusion counts, and cleaning counts.

| Decision | Reason |
|---|---|
| 36-month loans only | 60-month vintages are still right-censored in the 2020Q3 snapshot |
| Issued 2013–2016 | Labels are mature. In 2012, ~30 bureau fields were only partially rolled out (~52% missing), a missingness pattern that never occurs at scoring time |
| Individual applications | Joint-applicant fields are absent before late 2015 |
| Split by whole issue year | Train 2013–14 (263k, 13.2% default), validation 2015 (283k, 14.9%), test 2016 (318k, 15.9%) |
| 62 application/bureau inputs, plus `sub_grade` for Model B | Defined once in `src/lc_risk/schema.py` and shared by the cohort build, training, API validation, and monitoring |
| No pricing or post-origination fields as inputs | `int_rate` and the payment/recovery fields are kept only for evaluation; a test enforces this |

## Plan

| Stage | Scope |
|---|---|
| M0 ✅ | Project skeleton (uv, package layout, lint, tests) |
| M1 ✅ | Cohort build: 36-month loans 2013–2016, year-based split, schema checks, data hashes |
| M2 | Single sklearn pipeline, Model A (no LC rating) vs Model B (+`sub_grade`), calibration, one-shot OOT test |
| M3 | FastAPI scoring service with reason codes, Docker image |
| M4 | Tests + GitHub Actions CI/CD, deployment on Render |
| M5 | Drift monitoring (PSI) with delayed-label simulation |
| M6 | Slim Streamlit demo, model card, architecture write-up |

## Development

```bash
uv sync --all-groups --all-extras
uv run pytest
uv run ruff check . && uv run ruff format --check .

# raw-data audit (downloads the Kaggle snapshot into data/raw/, writes discovery/reports/)
uv run python discovery/run_data_discovery.py

# build the modeling cohort, then refresh the EDA notebook
uv run python -m lc_risk.data.cohort
uv run jupyter nbconvert --to notebook --execute --inplace discovery/eda.ipynb
```
