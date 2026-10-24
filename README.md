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
| 62 application/bureau fields accepted; 61 used by the model (+ `sub_grade` for Model B) | Defined once in `src/lc_risk/schema.py` and shared by the cohort build, training, API validation, and monitoring. `addr_state` is accepted but excluded from the model for fairness ([amendment](docs/model_selection.md#amendment-1--remove-addr_state-decided-before-training-v110)) |
| No pricing or post-origination fields as inputs | `int_rate` and the payment/recovery fields are kept only for evaluation; a test enforces this |

## Results (v1.1.0, out-of-time test on 2016 loans)

[`models/lc-default-risk/`](models/lc-default-risk/model_card.md): XGBoost with monotone
constraints, Platt-calibrated. It uses **none** of LC's grade, LC's interest rate, or the
applicant's state.

| | this model | logistic baseline | LC's own sub-grade |
|---|---|---|---|
| AUC | **0.688** | 0.679 | 0.672 |
| Default rate when approving the best 80% | **12.34%** | 12.57% | 12.73% |
| Realized return on that book | **+7.06%** | +6.84% | +6.46% |

- **Fairness fix with almost no cost.** v1.0.0's reason codes often blamed the applicant's
  state. Dropping `addr_state` ([amendment](docs/model_selection.md#amendment-1--remove-addr_state-decided-before-training-v110))
  cost −0.0011 AUC (95% CI −0.0018 to −0.0004) and left the lending outcomes unchanged. The
  most frequent reasons are now FICO, DTI, recent account openings, and loan-to-income.
- Adding LC's `sub_grade` would add +0.012 AUC, so LC knew something the bureau data does not
  show. It is reported as an ablation and not used, per [rules fixed before testing](docs/model_selection.md).
- The ceiling is the information in the inputs, not the model: doubling the data adds about
  +0.004 AUC, ten hand-built features add +0.0014, and tuning the hyperparameters barely matters.
- Inputs stayed stable through 2016 (score PSI 0.001), yet AUC fell from 0.70 to 0.68 quarter by
  quarter. Monitoring therefore has to track realized outcomes, not just input drift.
- Fully reproducible: retraining from a clean commit matched to five decimals.

## Scoring API

```bash
docker build -t lc-default-risk .
docker run -p 8000:8000 lc-default-risk        # interactive docs at http://localhost:8000/docs
```

| Endpoint | |
|---|---|
| `GET /health` | liveness and the loaded model version |
| `GET /model` | model metadata: inputs, calibration, test metrics, provenance |
| `POST /predict` | one application → calibrated PD, top 3 reason codes, out-of-policy warnings |
| `POST /predict/batch` | up to 1,000 applications |

- **One contract.** Request fields, types, ranges, and categories are generated from
  `src/lc_risk/schema.py`. Fields that were always present in training are required; only
  fields where a null means something (e.g. no delinquency on file) may be omitted. Unknown
  fields are rejected, so a typo cannot silently become a missing value.
- **Parity tested.** A test sends synthetic applications through HTTP and checks the
  probabilities against offline scoring of the same rows (max difference < 1e-6). CI also
  loads the committed model bundle and fails if a code change breaks it.
- **Safe startup.** The service refuses to start if the bundle's SHA-256 or its pinned
  scikit-learn/xgboost versions do not match.
- **Footprint.** 163 MB compressed image (CPU-only xgboost, no training dependencies), runs as
  a non-root user. Single-request latency is p50 19 ms / p95 37 ms in Docker on a laptop, and a
  batch of 1,000 takes about 0.3 s. Each request writes one JSON log line with latency, model
  version, and mean PD, and no applicant data.

## Plan

| Stage | Scope |
|---|---|
| M0 ✅ | Project skeleton (uv, package layout, lint, tests) |
| M1 ✅ | Cohort build: 36-month loans 2013–2016, year-based split, schema checks, data hashes |
| M2 ✅ | Single sklearn pipeline, Model A (no LC rating) vs Model B (+`sub_grade`), calibration, one-shot OOT test |
| M3 ✅ | FastAPI scoring service with reason codes, Docker image |
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

# train (Optuna, ~40 min), score the test year once, diagnose the ceiling, release
uv run python -m lc_risk.training.train --trials 30 --run-id <id>
uv run python -m lc_risk.training.evaluate --run <id>
uv run python -m lc_risk.training.diagnostics --run <id>
uv run python -m lc_risk.training.release --run <id> --model xgboost_A --version <x.y.z>

# serve locally without Docker
MODEL_DIR=models/lc-default-risk uv run uvicorn --factory lc_risk.serving.app:create_app --reload

# build the modeling cohort, then refresh the EDA notebook
uv run python -m lc_risk.data.cohort
uv run jupyter nbconvert --to notebook --execute --inplace discovery/eda.ipynb
```
