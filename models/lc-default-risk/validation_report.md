# Validation report — run `v1.1-no-state`

Train 2013–2014, validation 2015. The test year was not used. Calibration is fit on the validation year, so calibration metrics here are shown **before** calibration; calibrated metrics are only meaningful on the test year.

| model | AUC | Gini | KS | mean PD (raw) | ECE (raw) |
|---|---|---|---|---|---|
| logistic_A | 0.6859 | 0.3718 | 0.2706 | 0.1354 | 0.0137 |
| xgboost_A | 0.6959 | 0.3919 | 0.2845 | 0.1375 | 0.0119 |
| xgboost_A_unconstrained | 0.6972 | 0.3944 | 0.2858 | 0.1375 | 0.0115 |
| logistic_B | 0.7012 | 0.4024 | 0.2920 | 0.1322 | 0.0167 |
| xgboost_B | 0.7060 | 0.4120 | 0.3017 | 0.1334 | 0.0156 |
| xgboost_B_unconstrained | 0.7070 | 0.4140 | 0.3008 | 0.1340 | 0.0151 |

Validation default rate: 0.1489. Optuna trials per variant: 30.

## Bad rate at 80% approval

| ranking | bad rate |
|---|---|
| LC sub_grade | 0.1189 |
| xgboost_A | 0.1142 |
| xgboost_B | 0.1121 |

![approval](figures/approval_validation.png)
![calibration](figures/calibration_raw_validation.png)
