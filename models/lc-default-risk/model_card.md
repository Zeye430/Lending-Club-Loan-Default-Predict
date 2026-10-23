# Model card — lc-default-risk v1.0.0

## What it does

Predicts the probability that a 36-month individual Lending Club loan is charged off, from information available when the application is scored: 62 application and credit-bureau fields. It does **not** use LC's grade or interest rate. Output: a calibrated probability of default (PD) plus the top risk-increasing factors for that applicant (reason codes).

Algorithm: XGBoost (962 trees, max depth 3), with monotone constraints on 12 features, followed by Platt calibration (slope 0.943, intercept +0.003).

## Intended use

- Ranking and pricing support for 36-month unsecured personal loans similar to LC's 2013–2016 book; portfolio-level expected-loss estimates.
- **Not** for: 60-month loans, joint applications, applicants outside LC's historical approval range (e.g. FICO < 660, DTI ≥ 40%), or as the sole basis of an adverse decision without human review.

## Data

- Train: loans issued 2013–2014; calibration/validation: 2015; out-of-time test: 2016 (318,223 loans, 15.9% default rate).
- Cohort definition and evidence: `discovery/reports/data_audit.md` and `discovery/eda.ipynb`. Selection rules, fixed before testing: `docs/model_selection.md`.

## Performance on the out-of-time test year

| | this model | logistic baseline | LC sub-grade ranking |
|---|---|---|---|
| AUC | 0.6895 | 0.6804 | 0.6723 |
| Gini | 0.3790 | 0.3609 | 0.3447 |
| KS | 0.2742 | | |

Validation AUC was 0.6973.
XGBoost vs the logistic baseline: +0.0091 AUC (95% CI +0.0080 to +0.0101).

**Value of LC's grade (ablation).** Adding `sub_grade` to the same model changes test AUC by +0.0117 (95% CI +0.0105 to +0.0128). This model deliberately does not use it; see `docs/model_selection.md`.

## Calibration

Test year: mean predicted PD 14.72% vs observed default rate 15.88%; expected calibration error 0.0121; Brier score 0.1255.

![calibration](figures/calibration_test.png)

By issue quarter of the test year:

| quarter | loans | AUC | mean PD | default rate |
|---|---|---|---|---|
| 2016Q1 | 94,592 | 0.702 | 14.8% | 15.6% |
| 2016Q2 | 73,666 | 0.693 | 14.9% | 16.1% |
| 2016Q3 | 73,574 | 0.682 | 15.0% | 16.6% |
| 2016Q4 | 76,391 | 0.677 | 14.2% | 15.3% |

## Lending impact

Approving the 80% of test-year applicants with the lowest predicted risk:

| ranking | default rate among approved | realized return on approved book |
|---|---|---|
| this model | 12.35% | +7.06% |
| LC sub-grade | 12.73% | +6.46% |
| approve everyone | 15.88% | +5.91% |

Realized return uses each loan's actual repayments and recoveries over its life, not annualized.

![approval](figures/approval_test.png)

## Explainability

Monotone constraints: `fico_range_low` (higher → less risk), `annual_inc` (higher → less risk), `mort_acc` (higher → less risk), `credit_history_months` (higher → less risk), `dti` (higher → more risk), `loan_to_income` (higher → more risk), `revol_util` (higher → more risk), `inq_last_6mths` (higher → more risk), `acc_open_past_24mths` (higher → more risk), `num_tl_op_past_12m` (higher → more risk), `delinq_2yrs` (higher → more risk), `pub_rec_bankruptcies` (higher → more risk).

Reason codes are exact TreeSHAP contributions computed by XGBoost, scaled by the calibration slope, so they are in calibrated log-odds.

| top drivers (mean abs contribution, test year) | |
|---|---|
| fico_range_low | 0.201 |
| dti | 0.146 |
| acc_open_past_24mths | 0.140 |
| loan_to_income | 0.132 |
| annual_inc | 0.114 |
| num_tl_op_past_12m | 0.095 |
| percent_bc_gt_75 | 0.095 |
| total_il_high_credit_limit | 0.094 |
| addr_state | 0.092 |
| purpose | 0.085 |

## Where the performance ceiling is

Validation-year diagnostics (`lc_risk.training.diagnostics`):

| lever | effect on validation AUC |
|---|---|
| more training rows | 0.6818 (10%) → 0.6895 (25%) → 0.6940 (50%) → 0.6969 (100%); about +0.0046 per doubling |
| 10 hand-built ratio features | +0.0015 |
| hyperparameter tuning | best of 30 trials was #4 (random warm-up); top-5 spread 0.0009 |
| new information: LC's sub_grade | +0.0106 |

More rows, more features, and more tuning each add little; new information adds more. The ceiling is the information in application-time bureau data for an already-approved population, not the model. Gains would come from new sources (cash-flow data, trended bureau history, the bureau fields LC added in late 2015) rather than from modeling.

## Limitations and risks

- **Approved loans only.** Rejected applicants are absent from the data, so the model has not learned risk outside LC's approval range (no reject inference).
- **Drift.** Default rates rose within every LC grade from 2013 to 2016, and the calibration was fit on 2015. Expect under-prediction if the trend continues; recalibrate on the latest matured vintage.
- **Selection-encoded inputs.** `verification_status` reflects LC choosing to verify riskier applicants; it is predictive but not causal.
- **Fairness.** The data holds no protected attributes, so disparate impact cannot be measured directly. `addr_state` can proxy for them; review its contribution before any real use.
- **Serialization.** The model is a pickle; it loads only with the pinned scikit-learn and xgboost versions (checked at load time).

## Monitoring

Within the test year, AUC fell from 0.702 (2016Q1) to 0.677 (2016Q4), while the score PSI between validation and test was only 0.0013. **Input and score drift monitoring alone would not have caught this.** In production, track input PSI per field and score PSI per batch (alert above 0.25), and compare predicted PD with the realized default rate and AUC as each vintage's labels mature.

## Follow-up work

- Refit on 2013–2015 and recalibrate on the newest matured vintage, with a fresh holdout.
- A separate model for 60-month loans on an earlier cohort.

## Provenance

- Training run `v2-release`; code commit `606dea4e8e106a26f503b80e86c3e8aefa15f96e` (dirty: False), package source SHA-256 `232b7aaf88798adcba8da72bd87dd1f5fdc62cb6c0bc31c6ac61c4d0b40a412f`
- Cohort `cohort.parquet` SHA-256 `1c10b1db261dc015f802b05ef050bbcf642cd26633575cfe71859d69a05d1d02`
- Libraries: scikit-learn 1.9.1, xgboost 3.4.1, optuna 5.0.0, pandas 3.1.0, numpy 2.5.4
- Full reports: [validation](validation_report.md), [test](test_report.md)
