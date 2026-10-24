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

## Outcome (run `v2-release`)

Applied to the validation year, before scoring the test year:

| Rule | Validation evidence | Result |
|---|---|---|
| 1. Input set A | — | A |
| 2. Keep constraints unless they cost > 0.005 AUC | unconstrained 0.6989 vs constrained 0.6973 (cost 0.0016) | keep |
| 3. XGBoost only if it beats logistic | 0.6973 vs 0.6879 | XGBoost |

Shipped: `xgboost_A` → `models/lc-default-risk/` v1.0.0. Test year (scored once): AUC 0.6895.

Reproducibility: the run was repeated from a clean commit (`606dea4`). All six models matched the
earlier run to five decimals of validation AUC, with identical hyperparameters and calibration.
Rebuilding the cohort twice produced byte-identical parquet files.

## Amendment 1 — remove `addr_state` (decided before training v1.1.0)

**Why.** Reviewing v1.0.0's reason codes on sample applicants showed `addr_state` among the
top three risk factors for four of five applicants. That is a problem for two reasons that
have nothing to do with test performance:

- geography can proxy for protected characteristics (fair-lending risk);
- "your state of residence" is neither a specific nor an actionable reason to give an applicant.

**Decision.** `addr_state` stays in the schema, so the service still accepts and validates it
and monitoring can break results down by state. It is listed in `schema.EXCLUDED_FROM_MODEL`
and is no longer a model input. The same was done for Model B so the A-vs-B ablation stays
like-for-like.

**Protocol.** Rules 1–5 above apply unchanged to the new run. This is the **second** use of the
2016 test year. The decision to drop the field was made from v1.0.0's reason codes, not from
its test metrics. The v1.0.0 test results stay on record (commit `dfcb634`) so the cost of the
change can be read directly as v1.1.0 minus v1.0.0.
