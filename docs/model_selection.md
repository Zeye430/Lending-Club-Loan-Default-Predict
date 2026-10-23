# Model selection protocol

Written **before** the main training run (`v2-main`) finished and before the test year was
scored. The rules below decide which model ships; the test year is used only to report its
performance.

## Candidates

| Name | Inputs | Algorithm |
|---|---|---|
| `logistic_A` | 62 application/bureau fields + 2 derived | L2 logistic regression (scorecard-style baseline) |
| `xgboost_A` | same as A | XGBoost with monotone constraints, Optuna-tuned |
| `xgboost_A_unconstrained` | same as A | same parameters, constraints removed |
| `logistic_B`, `xgboost_B`, `xgboost_B_unconstrained` | A + LC `sub_grade` | as above |

Every variant gets the same splits, the same 30-trial Optuna budget, and Platt calibration
on the 2015 validation year.

## Rules, fixed in advance

1. **The shipped model uses input set A.** `sub_grade` is the output of LC's existing
   underwriting model. A risk model meant to stand on its own, and to be scored by a
   service that receives only application and bureau data, cannot depend on it. Model B is
   an ablation: it measures how much LC's rating adds on top of the bureau data.
2. **Monotone constraints are kept** unless they cost more than 0.005 validation AUC against
   the unconstrained model with the same parameters. They make the model's behavior
   defensible: for example, a higher FICO can never raise the predicted risk.
3. **XGBoost ships over the logistic baseline** only if it wins on validation AUC. If it
   does not, the logistic model ships. The reason codes would then come from coefficients.
4. **The test year is scored once** (`lc_risk.training.evaluate` refuses a second run).
   Its results are reported, not used to choose between models or to tune anything.
5. **The shipped model is exactly the evaluated model**: trained on 2013–14 and calibrated
   on 2015. Refitting on 2013–15 before shipping would produce a model with no held-out
   evidence. That refit is listed as follow-up work in the model card.
