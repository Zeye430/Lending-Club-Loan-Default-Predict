"""The deployable model: feature pipeline + classifier + probability calibration.

``RiskModel`` is the single object that training saves and the service loads. All classes
it references live in this package, so unpickling only requires ``lc_risk`` to be
installed, never a training script.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

_EPS = 1e-7


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass(frozen=True)
class PlattCalibrator:
    """calibrated log-odds = slope * raw log-odds + intercept.

    Two parameters fit on a recent labeled period. It is monotonic, so ranking (AUC)
    is unchanged; it corrects the level and spread of predicted probabilities.
    """

    slope: float
    intercept: float

    @classmethod
    def fit(cls, raw_log_odds: np.ndarray, y: np.ndarray) -> PlattCalibrator:
        lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularized
        lr.fit(np.asarray(raw_log_odds).reshape(-1, 1), y)
        return cls(slope=float(lr.coef_[0, 0]), intercept=float(lr.intercept_[0]))

    def __call__(self, raw_log_odds: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(raw_log_odds) + self.intercept


@dataclass(frozen=True)
class Reason:
    feature: str
    value: float | str | None
    contribution: float  # change in calibrated log-odds; positive raises risk


class RiskModel:
    def __init__(self, pipeline: Pipeline, calibrator: PlattCalibrator | None = None):
        self.pipeline = pipeline
        self.calibrator = calibrator

    @property
    def features(self):
        return self.pipeline.named_steps["features"]

    @property
    def estimator(self):
        return self.pipeline.steps[-1][1]

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.get_feature_names_out())

    def raw_log_odds(self, X: pd.DataFrame) -> np.ndarray:
        return logit(self.pipeline.predict_proba(X)[:, 1])

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability of default."""
        z = self.raw_log_odds(X)
        return sigmoid(self.calibrator(z) if self.calibrator else z)

    def with_calibration(self, X: pd.DataFrame, y: np.ndarray) -> RiskModel:
        return RiskModel(self.pipeline, PlattCalibrator.fit(self.raw_log_odds(X), y))

    def explain(self, X: pd.DataFrame, top_k: int = 3) -> list[list[Reason]]:
        """Top risk-increasing features per row, from XGBoost's exact SHAP contributions."""
        if not isinstance(self.estimator, xgb.XGBClassifier):
            raise NotImplementedError("reason codes require the XGBoost model")
        features = self.features.transform(X)
        contributions = self.estimator.get_booster().predict(
            xgb.DMatrix(features, enable_categorical=True), pred_contribs=True
        )[:, :-1]  # last column is the bias term
        if self.calibrator:
            contributions = contributions * self.calibrator.slope

        names = features.columns
        reasons = []
        for row, contrib in zip(features.itertuples(index=False), contributions, strict=True):
            order = np.argsort(-contrib)[:top_k]
            reasons.append(
                [
                    Reason(names[i], _plain(row[i]), round(float(contrib[i]), 4))
                    for i in order
                    if contrib[i] > 0
                ]
            )
        return reasons


def _plain(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value.item() if isinstance(value, np.generic) else value
