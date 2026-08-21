"""ML pipeline construction.

Builds (features, labels) from raw price data and instantiates models from
config. Centralizing this prevents the train-prod skew problem: the same
function builds features for training, walk-forward testing, and live
inference.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from quant_ml.features.technical import build_feature_matrix


def make_classifier(model_type: str, params: dict[str, Any] | None = None) -> ClassifierMixin:
    """Factory for sklearn classifiers."""
    params = params or {}
    if model_type == "random_forest":
        defaults = {"n_estimators": 200, "max_depth": 5, "random_state": 42, "n_jobs": -1}
        return RandomForestClassifier(**{**defaults, **params})
    if model_type == "logistic_regression":
        defaults = {"max_iter": 1000, "random_state": 42}
        return LogisticRegression(**{**defaults, **params})
    if model_type == "gradient_boosting":
        defaults = {"n_estimators": 200, "max_depth": 3, "random_state": 42}
        return GradientBoostingClassifier(**{**defaults, **params})
    raise ValueError(f"Unknown model_type: {model_type}")


def build_dataset(
    prices: pd.DataFrame,
    horizon: int = 1,
    close_col: str = "Adj Close",
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for ML training.

    Rows without a known future price are excluded rather than being
    incorrectly treated as class 0.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    if close_col not in prices.columns:
        raise KeyError(f"Missing close column: {close_col}")

    features = build_feature_matrix(
        prices,
        close_col=close_col,
    )

    forward_return = (
        prices[close_col].shift(-horizon)
        / prices[close_col]
        - 1
    )

    # Preserve NaN where the future price does not exist.
    target = pd.Series(
        np.nan,
        index=prices.index,
        dtype=float,
        name="target",
    )

    valid_future = forward_return.notna()

    target.loc[valid_future] = (
        forward_return.loc[valid_future] > 0
    ).astype(int)

    df = features.join(target)

    # Remove incomplete features and rows without future labels.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    return X, y
