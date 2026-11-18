# Architecture

How the pieces fit together, and the engineering decisions behind each one. For results, see
the [README](../README.md) and the [model card](../models/lc-default-risk/model_card.md).

The end-to-end diagram is in the [README](../README.md#architecture).

## One input contract

`src/lc_risk/schema.py` defines every input field once: type, range, categories, description,
and whether a null is meaningful. Every stage reads it:

| Consumer | Use |
|---|---|
| cohort build | validates 864k loans; fails if a required field is missing in more than 0.5% of training rows |
| training | column lists, fixed category sets, which fields the model may use |
| API | request model generated from it: types, ranges, dropdowns, required fields, descriptions in `/docs` |
| monitoring | per-field drift and data-quality checks |
| demo | reads the same contract through the API's OpenAPI document |

The model also stores the field list it was fitted with, so a later schema change cannot alter a
released model.

## Data

Source: the Kaggle snapshot of LC loans issued 2007–2020Q3 (2.9M rows, 142 columns).
`python -m lc_risk.data.cohort` streams it into `data/interim/cohort.parquet` and writes a
manifest with the SHA-256 of the input and the output, exclusion counts, and cleaning counts.
Rebuilding from the same commit produces a byte-identical file.

| Decision | Reason |
|---|---|
| 36-month loans only | 60-month vintages are still right-censored in the 2020Q3 snapshot |
| Issued 2013–2016 | Labels are mature. In 2012, ~30 bureau fields were only partly rolled out, so they were ~52% missing; that pattern never occurs at scoring time |
| Individual applications | Joint-applicant fields are absent before late 2015 |
| Split by whole issue year | Train 2013–14 (263k, 13.2% default), validation 2015 (283k, 14.9%), test 2016 (318k, 15.9%) |
| 62 fields accepted, 61 used | `addr_state` is recorded but excluded from the model ([amendment 1](model_selection.md#amendment-1--remove-addr_state-decided-before-training-v110)) |
| No pricing or post-origination inputs | `int_rate` and the payment/recovery fields are kept for evaluation only; a test enforces this |

The evidence for each decision is in the [raw data audit](../discovery/reports/data_audit.md)
and the [EDA notebook](../discovery/eda.ipynb).

## Model and release

- **Protocol.** Selection rules were written in [model_selection.md](model_selection.md) before
  the test year was scored. `evaluate` refuses to score a run twice. The shipped model is
  exactly the evaluated one.
- **Bundle.** `models/lc-default-risk/` holds the pickled pipeline (features, XGBoost, and Platt
  calibration in one object), `metadata.json` (training commit, source hash, cohort hash,
  library versions, metrics), and the generated model card and reports. A new model version
  is a reviewed commit.
- **Provenance.** Each training run records the git commit, a dirty flag, and a SHA-256 of the
  package source taken at start. Retraining from a clean commit matched the earlier run to
  five decimals.

## Scoring API

| Endpoint | |
|---|---|
| `GET /` | redirects to the interactive docs |
| `GET /health` | liveness, the deployed commit, and the model version |
| `GET /model` | inputs, calibration, test metrics, provenance |
| `POST /predict` | one application → calibrated PD, top 3 reason codes, out-of-policy warnings |
| `POST /predict/batch` | up to 1,000 applications |

- **Strict requests.** Unknown fields are rejected, so a typo cannot quietly become a missing
  value. Fields that were always present in training are required.
- **Parity.** A test sends applications through HTTP and compares the probabilities with
  offline scoring of the same rows; the maximum difference is below 1e-6. Another test loads
  the committed bundle on every CI run.
- **Safe startup.** The service refuses to start if the bundle's SHA-256 or its pinned
  scikit-learn/xgboost versions do not match.
- **Footprint.** The image is 163 MB compressed: CPU-only xgboost and no training stack. It
  runs as a non-root user. Latency on a laptop in Docker is p50 19 ms and p95 37 ms for one
  request, and a batch of 1,000 takes about 0.3 s. Each request writes one JSON log line with
  latency, model version and mean PD, and no applicant data.

## CI/CD

```
PR ──► lint + tests ──► image build + smoke test ──► merge (blocked unless both pass)
                                                           │
main ──► same checks ──► deploy: Render hook ──► wait for /health to report this commit
                                                    ──► score a live request
```

- **CI** runs on every push and pull request. The smoke test starts the container and scores
  the example application taken from the container's own OpenAPI document.
- **CD** runs on `main` only, after both checks pass. It polls `/health` until the live
  service reports the exact commit CI tested, so a deploy that did not land turns the pipeline
  red.
- **Branch protection.** A ruleset on `main` requires pull requests and both checks, and
  blocks force pushes and deletion.

## Demo

`app/` is a Streamlit app deployed on Streamlit Community Cloud ([live](https://lending-club-loan-default-predict-6jekialmnxiawsqpfusvgd.streamlit.app/)). It is a thin HTTP client: it
installs only `app/requirements.txt`, never loads the model, and builds its form from the API's
OpenAPI document. Its client is contract-tested against the real API in CI, and a Streamlit
`AppTest` drives the full UI against the in-process service.

## Monitoring

Five layers, from fastest to most complete. See the [playbook](monitoring/README.md) and the
[2016 replay](monitoring/report.md).
