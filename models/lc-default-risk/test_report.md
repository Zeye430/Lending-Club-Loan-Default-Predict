# Test report — run `v1.1-no-state`

Out-of-time test: 318,223 loans issued in 2016, default rate 15.88%. Models were tuned on train/validation only and calibrated on the validation year. Evaluated once at 2026-10-24T01:25:15+00:00.

## Discrimination and calibration

| model | AUC | Gini | KS | Brier | ECE | mean PD |
|---|---|---|---|---|---|---|
| logistic_A | 0.6790 | 0.3581 | 0.2594 | 0.1264 | 0.0131 | 0.1457 |
| logistic_B | 0.6955 | 0.3910 | 0.2832 | 0.1248 | 0.0109 | 0.1480 |
| xgboost_A | 0.6884 | 0.3768 | 0.2715 | 0.1256 | 0.0120 | 0.1472 |
| xgboost_A_unconstrained | 0.6886 | 0.3772 | 0.2720 | 0.1255 | 0.0114 | 0.1475 |
| xgboost_B | 0.7004 | 0.4007 | 0.2904 | 0.1244 | 0.0103 | 0.1488 |
| xgboost_B_unconstrained | 0.7004 | 0.4008 | 0.2899 | 0.1243 | 0.0099 | 0.1489 |
| lc_sub_grade | 0.6723 | 0.3447 | 0.2475 | — | — | — |

`lc_sub_grade` ranks loans by LC's own grade; it has no probability, so no calibration.

## Paired bootstrap comparisons (200 resamples)

| comparison | AUC difference | 95% CI low | 95% CI high |
|---|---|---|---|
| B_vs_A_xgboost (value of LC sub_grade) | 0.0120 | 0.0109 | 0.0130 |
| xgboost_vs_logistic_A | 0.0094 | 0.0082 | 0.0103 |
| unconstrained_vs_monotone_A | 0.0002 | -0.0002 | 0.0007 |

## Lending policy: approve the lowest-risk share

Realized return = (total repaid incl. recoveries − funded) / funded, from actual cash flows over the life of the loans (not annualized).

| ranking | approval rate | bad rate | realized return |
|---|---|---|---|
| LC sub_grade | 70% | 11.62% | +6.54% |
| LC sub_grade | 80% | 12.73% | +6.46% |
| LC sub_grade | 90% | 14.05% | +6.27% |
| LC sub_grade | 100% | 15.88% | +5.91% |
| logistic_A | 70% | 11.35% | +7.11% |
| logistic_A | 80% | 12.57% | +6.84% |
| logistic_A | 90% | 13.95% | +6.50% |
| logistic_A | 100% | 15.88% | +5.91% |
| xgboost_A | 70% | 11.09% | +7.32% |
| xgboost_A | 80% | 12.34% | +7.06% |
| xgboost_A | 90% | 13.84% | +6.67% |
| xgboost_A | 100% | 15.88% | +5.91% |
| xgboost_B | 70% | 10.72% | +7.27% |
| xgboost_B | 80% | 12.08% | +7.03% |
| xgboost_B | 90% | 13.68% | +6.67% |
| xgboost_B | 100% | 15.88% | +5.91% |

![approval](figures/approval_test.png)

## Calibration and stability of the primary model

![calibration](figures/calibration_test.png)

Score PSI, validation → test: 0.0013

| quarter | loans | auc | mean_pd | default_rate |
|---|---|---|---|---|
| 2016Q1 | 94592 | 0.7012 | 0.1480 | 0.1557 |
| 2016Q2 | 73666 | 0.6921 | 0.1490 | 0.1613 |
| 2016Q3 | 73574 | 0.6794 | 0.1493 | 0.1664 |
| 2016Q4 | 76391 | 0.6768 | 0.1423 | 0.1531 |

## What drives the primary model (mean |SHAP| in log-odds)

| feature | mean abs contribution |
|---|---|
| fico_range_low | 0.1957 |
| dti | 0.1418 |
| acc_open_past_24mths | 0.1375 |
| loan_to_income | 0.1284 |
| annual_inc | 0.1189 |
| num_tl_op_past_12m | 0.1002 |
| total_il_high_credit_limit | 0.0957 |
| percent_bc_gt_75 | 0.0914 |
| home_ownership | 0.0855 |
| purpose | 0.0845 |
| loan_amnt | 0.0836 |
| bc_open_to_buy | 0.0794 |
| bc_util | 0.0749 |
| mths_since_recent_inq | 0.0714 |
| mo_sin_rcnt_rev_tl_op | 0.0692 |
