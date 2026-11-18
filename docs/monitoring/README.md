# Monitoring playbook — lc-default-risk

How the scoring service is monitored, what each alert means, and what to do when one fires.
The [replay report](report.md) applies these rules to 2016, treating that year as production
traffic.

## Why five layers

A default model's outcome is known only ~40 months after a loan is scored. Checks on the
inputs are available immediately but miss changes in how inputs relate to outcomes. Checks
on outcomes catch those changes but arrive years late. Each layer trades speed for coverage:

| Layer | Metric | Visible | Catches | Misses |
|---|---|---|---|---|
| Data quality | share of applications the API rejects | same day | broken upstream feeds, schema changes | anything the contract allows |
| Input drift | max PSI over input fields vs the reference year | same day | shifts in who applies | changes in how inputs relate to default |
| Score drift | PSI of predicted PD | same day | shifts in the scored risk mix | the same |
| Early performance | defaults within 12 months on book: actual / expected, and AUC | ~13 months after issue | most outcome shifts, about two years early | defaults that come late in a loan's life |
| Final performance | lifetime defaults: actual / expected, and AUC | ~40 months after issue | everything, definitively | nothing, but far too late to steer by |

The reference year is the model's calibration year (2015). Performance metrics pool the
trailing three issue months, because one month holds only ~1,000 early defaults and its AUC
is too noisy to alert on.

## Rules

Defined in [`src/lc_risk/monitoring/rules.py`](../../src/lc_risk/monitoring/rules.py):

| Metric | Warning | Critical | Basis |
|---|---|---|---|
| `rejected_share` | ≥ 0.5% | ≥ 2% | the input contract tolerates up to 0.5% missing per required field |
| `max_input_psi` | ≥ 0.10 | ≥ 0.25 | industry convention |
| `score_psi` | ≥ 0.10 | ≥ 0.25 | industry convention |
| `early_ae`, `final_ae` | off by ≥ 10% | off by ≥ 20% | usual recalibration triggers |
| `early_auc_drop`, `final_auc_drop` | ≥ 0.01 | ≥ 0.02 | about one and two times the quarter-to-quarter spread of 2015 early AUC |

**Batch size.** PSI has sampling noise that grows as batches shrink, and taking the maximum
over ~60 fields inflates it further. These thresholds assume batches of thousands of loans;
at a few hundred loans per batch, `max_input_psi` crosses 0.10 with no real drift. Pool small
batches before applying the rule.

## Response

| Alert | First check | Then |
|---|---|---|
| `rejected_share` | which fields fail (the service returns 422 with the field) and which upstream system sends them | fix the feed; make a field nullable only if a null genuinely means something, and retrain if it does |
| `max_input_psi` / `score_psi` | which fields moved and whether it was a product or marketing change (new channel, new state, new loan sizes) | if the new population is outside the training range, flag those applications (the service already warns past LC's policy edges) and plan a retrain on data that covers it |
| `early_ae` | whether the shift is one vintage or a trend, and which segments drive it | short term, recalibrate on the newest matured early labels (the Platt step only, no retrain); report the expected impact on approvals |
| `early_auc_drop` | segment AUCs: purpose, FICO band, income band | ranking has degraded, so recalibration will not fix it; retrain on the most recent vintages with mature labels, with a new out-of-time holdout |
| `final_ae` / `final_auc_drop` | whether early monitoring already raised the same alert | confirms or rules out the early signal; feed the confirmed numbers into the next model card |

Any critical alert also triggers a review of the approval cutoff, because a mis-calibrated PD
changes how many applicants clear a fixed threshold.

## What the 2016 replay showed

- No data-quality, input-drift or score-drift alert fired in any month. The applicants looked
  like 2015.
- Early-label AUC crossed the warning line in data visible in July 2017. The lifetime AUC
  showed the same drop only in November 2019, **28 months later**.
- Early actual / expected fired from the first vintage (1.18) and peaked at 1.28, while the
  lifetime ratio peaked at 1.11. Defaults rose and also arrived earlier in the loan's life, so
  the early ratio exaggerates the size of a level shift. Use it to decide *whether* to
  recalibrate, and the matured labels to decide *by how much*.

Rerun the replay:

```bash
uv run python -m lc_risk.monitoring.simulate      # writes docs/monitoring/report.md
```
