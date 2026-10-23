# Test report — run `v2-release`

Out-of-time test: 318,223 loans issued in 2016, default rate 15.88%. Models were tuned on train/validation only and calibrated on the validation year. Evaluated once at 2026-10-23T13:34:32+00:00.

## Discrimination and calibration

| model | AUC | Gini | KS | Brier | ECE | mean PD |
|---|---|---|---|---|---|---|
| logistic_A | 0.6804 | 0.3609 | 0.2608 | 0.1263 | 0.0128 | 0.1460 |
| logistic_B | 0.6966 | 0.3932 | 0.2854 | 0.1247 | 0.0107 | 0.1481 |
| xgboost_A | 0.6895 | 0.3790 | 0.2742 | 0.1255 | 0.0121 | 0.1472 |
| xgboost_A_unconstrained | 0.6900 | 0.3801 | 0.2752 | 0.1254 | 0.0118 | 0.1473 |
| xgboost_B | 0.7012 | 0.4025 | 0.2921 | 0.1243 | 0.0101 | 0.1489 |
| xgboost_B_unconstrained | 0.7017 | 0.4034 | 0.2920 | 0.1242 | 0.0101 | 0.1488 |
| lc_sub_grade | 0.6723 | 0.3447 | 0.2475 | — | — | — |

`lc_sub_grade` ranks loans by LC's own grade; it has no probability, so no calibration.

## Paired bootstrap comparisons (200 resamples)

| comparison | AUC difference | 95% CI low | 95% CI high |
|---|---|---|---|
| B_vs_A_xgboost (value of LC sub_grade) | 0.0117 | 0.0105 | 0.0128 |
| xgboost_vs_logistic_A | 0.0091 | 0.0080 | 0.0101 |
| unconstrained_vs_monotone_A | 0.0005 | -0.0001 | 0.0010 |

## Lending policy: approve the lowest-risk share

Realized return = (total repaid incl. recoveries − funded) / funded, from actual cash flows over the life of the loans (not annualized).

| ranking | approval rate | bad rate | realized return |
|---|---|---|---|
| LC sub_grade | 70% | 11.62% | +6.54% |
| LC sub_grade | 80% | 12.73% | +6.46% |
| LC sub_grade | 90% | 14.05% | +6.27% |
| LC sub_grade | 100% | 15.88% | +5.91% |
| logistic_A | 70% | 11.30% | +7.15% |
| logistic_A | 80% | 12.55% | +6.86% |
| logistic_A | 90% | 13.94% | +6.50% |
| logistic_A | 100% | 15.88% | +5.91% |
| xgboost_A | 70% | 11.06% | +7.32% |
| xgboost_A | 80% | 12.35% | +7.06% |
| xgboost_A | 90% | 13.83% | +6.67% |
| xgboost_A | 100% | 15.88% | +5.91% |
| xgboost_B | 70% | 10.68% | +7.31% |
| xgboost_B | 80% | 12.06% | +7.07% |
| xgboost_B | 90% | 13.67% | +6.66% |
| xgboost_B | 100% | 15.88% | +5.91% |

![approval](figures/approval_test.png)

## Calibration and stability of the primary model

![calibration](figures/calibration_test.png)

Score PSI, validation → test: 0.0013

| quarter | loans | auc | mean_pd | default_rate |
|---|---|---|---|---|
| 2016Q1 | 94592 | 0.7017 | 0.1480 | 0.1557 |
| 2016Q2 | 73666 | 0.6932 | 0.1489 | 0.1613 |
| 2016Q3 | 73574 | 0.6819 | 0.1496 | 0.1664 |
| 2016Q4 | 76391 | 0.6774 | 0.1424 | 0.1531 |

## What drives the primary model (mean |SHAP| in log-odds)

| feature | mean abs contribution |
|---|---|
| fico_range_low | 0.2006 |
| dti | 0.1455 |
| acc_open_past_24mths | 0.1396 |
| loan_to_income | 0.1315 |
| annual_inc | 0.1142 |
| num_tl_op_past_12m | 0.0949 |
| percent_bc_gt_75 | 0.0948 |
| total_il_high_credit_limit | 0.0943 |
| addr_state | 0.0916 |
| purpose | 0.0851 |
| loan_amnt | 0.0819 |
| home_ownership | 0.0803 |
| bc_util | 0.0702 |
| mths_since_recent_inq | 0.0701 |
| pct_tl_nvr_dlq | 0.0676 |
