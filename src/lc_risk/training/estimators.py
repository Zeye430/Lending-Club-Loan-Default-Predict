"""Pipeline factories for the candidate models. Each returns features -> estimator."""

from __future__ import annotations

import numpy as np
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from lc_risk.features import FeatureBuilder, monotone_constraints

SEED = 42


def xgboost_pipeline(
    include_lc_rating: bool, params: dict, monotone: bool = True, n_jobs: int = -1
) -> Pipeline:
    features = FeatureBuilder(include_lc_rating=include_lc_rating)
    names = list(features.get_feature_names_out())
    model = xgb.XGBClassifier(
        **params,
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=1,
        monotone_constraints=monotone_constraints(names) if monotone else None,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=n_jobs,
    )
    return Pipeline([("features", features), ("model", model)])


def _signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))


def logistic_pipeline(include_lc_rating: bool, C: float = 1.0) -> Pipeline:
    """Scorecard-style baseline: log-scaled numerics with missing indicators, one-hot categories."""
    numeric = make_pipeline(
        FunctionTransformer(_signed_log1p, feature_names_out="one-to-one"),
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    categorical = OneHotEncoder(handle_unknown="ignore", min_frequency=50)
    prep = ColumnTransformer(
        [
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_include="category")),
        ]
    )
    model = LogisticRegression(C=C, max_iter=2000)
    features = FeatureBuilder(include_lc_rating=include_lc_rating)
    return Pipeline([("features", features), ("prep", prep), ("model", model)])
