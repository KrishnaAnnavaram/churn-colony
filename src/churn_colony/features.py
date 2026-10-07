"""Leak-free engineered features and the leakage screen.

Every engineered feature uses only the columns of one customer row. The only fitted value
(the high-value price limit) is learned in `fit`, so inside a Pipeline it comes from the
training rows only. The transformer refuses a frame that contains the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score

from churn_colony import TARGET
from churn_colony.schema import ADDON_COLUMNS, CATEGORICAL, FEATURE_COLUMNS

LONG_TERM_MONTHS = 24
TENURE_BANDS = (-1, 12, 24, 48, 10_000)
TENURE_BAND_LABELS = ("0-12", "13-24", "25-48", "49+")

ENGINEERED_NUMERIC = ("avg_monthly_spend", "spend_drift", "n_addons", "n_streaming")
ENGINEERED_BINARY = ("long_term", "high_value", "autopay", "fiber_no_support")
ENGINEERED_CATEGORICAL = ("tenure_band",)

RAW_NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
RAW_CATEGORICAL = tuple(CATEGORICAL)

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC + ENGINEERED_BINARY
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL


class TargetInFeatures(ValueError):
    """The feature frame contains the target column."""


class TelcoFeatures(TransformerMixin, BaseEstimator):
    """Add row-wise features to the raw Telco columns. Never reads the target."""

    def __init__(self, high_value_quantile: float = 0.75):
        self.high_value_quantile = high_value_quantile

    def _check(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoFeatures needs a pandas DataFrame")
        bad = [c for c in X.columns if str(c).strip().lower() == TARGET.lower()]
        if bad:
            raise TargetInFeatures(f"the target column {bad[0]!r} must not be in the features")
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")

    def fit(self, X: pd.DataFrame, y=None):
        self._check(X)
        self.high_value_threshold_ = float(np.quantile(X["MonthlyCharges"].astype(float), self.high_value_quantile))
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check(X)
        if not hasattr(self, "high_value_threshold_"):
            raise RuntimeError("call fit before transform")
        out = X.loc[:, list(FEATURE_COLUMNS)].copy()
        tenure = out["tenure"].astype(float)
        monthly = out["MonthlyCharges"].astype(float)
        total = out["TotalCharges"].astype(float)
        out["avg_monthly_spend"] = np.where(tenure > 0, total / tenure.clip(lower=1), monthly)
        out["spend_drift"] = monthly - out["avg_monthly_spend"]
        out["n_addons"] = sum((out[c] == "Yes").astype(int) for c in ADDON_COLUMNS)
        out["n_streaming"] = (out["StreamingTV"] == "Yes").astype(int) + (out["StreamingMovies"] == "Yes").astype(int)
        out["long_term"] = (tenure >= LONG_TERM_MONTHS).astype(int)
        out["high_value"] = (monthly >= self.high_value_threshold_).astype(int)
        out["autopay"] = out["PaymentMethod"].str.contains("automatic", regex=False).astype(int)
        out["fiber_no_support"] = ((out["InternetService"] == "Fiber optic") & (out["TechSupport"] == "No")).astype(int)
        out["tenure_band"] = pd.cut(tenure, bins=TENURE_BANDS, labels=TENURE_BAND_LABELS).astype(str)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.asarray(
            list(FEATURE_COLUMNS) + list(ENGINEERED_NUMERIC) + list(ENGINEERED_BINARY) + list(ENGINEERED_CATEGORICAL),
            dtype=object,
        )


def single_feature_auc(X: pd.DataFrame, y) -> pd.Series:
    """Ranking power of each column alone, as max(AUC, 1 - AUC).

    A text column is scored by the churn rate of its level (in-sample), which gives an upper
    bound. The screen is for leaks, not for feature choice.
    """
    y = np.asarray(y).astype(int)
    scores = {}
    for col in X.columns:
        values = X[col]
        if not pd.api.types.is_numeric_dtype(values) or isinstance(values.dtype, pd.CategoricalDtype):
            rates = pd.Series(y, index=X.index).groupby(values.astype(str)).transform("mean")
            score = rates.to_numpy(dtype=float)
        else:
            score = values.to_numpy(dtype=float)
        if np.unique(score).size < 2:
            scores[col] = 0.5
            continue
        auc = roc_auc_score(y, score)
        scores[col] = max(auc, 1 - auc)
    return pd.Series(scores).sort_values(ascending=False)


def leakage_screen(X: pd.DataFrame, y, threshold: float = 0.95) -> list[str]:
    """Names of the columns that alone separate churners from stayers almost perfectly."""
    scores = single_feature_auc(X, y)
    return [c for c, s in scores.items() if s >= threshold]
