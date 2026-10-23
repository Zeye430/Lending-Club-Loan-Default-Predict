# Lending Club Default Risk — v2 rebuild

> **Work in progress.** This branch is a solo rebuild of a 5-person course project. The original
> team version (pandas pipeline, Databricks notebooks, Streamlit app, MCP server) is preserved
> at the git tag [`v1-course-project`](../../tree/v1-course-project).

## Why a rebuild

An audit of the original project (`scripts/`) found problems in how the experiment was designed:

- **Right-censored labels.** The 2017 test vintage was only ~71% resolved in the 2020Q3
  snapshot. Keeping only completed loans skewed the sample toward early defaults and early
  payoffs.
- **Threshold chosen on the test set**, so the reported test metrics were not a true holdout.
- **LC's own rating used as an input** (`sub_grade` + `int_rate`), so the model's incremental
  value over LC's underwriting was never measured.
- **Train/serve skew** in the deployed demo (`issue_year` as a feature, ~25 inputs filled with
  population defaults).

## Plan

| Stage | Scope |
|---|---|
| M0 | Project skeleton (uv, package layout, lint, tests) |
| M1 | Cohort build: 36-month loans 2012–2016, year-based split, schema checks, data hashes |
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

# data discovery (downloads the Kaggle snapshot into data/raw/)
uv run python scripts/run_data_discovery.py
```
