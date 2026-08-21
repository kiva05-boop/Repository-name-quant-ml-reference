"""Tests for the ML pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from quant_ml.ml.pipeline import build_dataset, make_classifier


def make_prices(n: int = 120) -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(42)
    close = pd.Series(
        100 + np.cumsum(rng.normal(0, 1, n)),
        index=idx,
    )

    return pd.DataFrame(
        {
            "Open": close,
            "High": close,
            "Low": close,
            "Close": close,
            "Adj Close": close,
            "Volume": 1_000_000,
        },
        index=idx,
    )


class TestMakeClassifier:
    def test_random_forest(self) -> None:
        model = make_classifier("random_forest")
        assert isinstance(model, RandomForestClassifier)

    def test_logistic_regression(self) -> None:
        model = make_classifier("logistic_regression")
        assert isinstance(model, LogisticRegression)

    def test_gradient_boosting(self) -> None:
        model = make_classifier("gradient_boosting")
        assert isinstance(model, GradientBoostingClassifier)

    def test_parameters_override_defaults(self) -> None:
        model = make_classifier(
            "random_forest",
            {"n_estimators": 10, "max_depth": 2},
        )
        assert model.n_estimators == 10
        assert model.max_depth == 2

    def test_unknown_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            make_classifier("does_not_exist")


class TestBuildDataset:
    def test_features_and_target_are_aligned(self) -> None:
        prices = make_prices()

        X, y = build_dataset(prices, horizon=5)

        assert len(X) == len(y)
        assert X.index.equals(y.index)

    def test_no_nan_or_inf_in_features(self) -> None:
        prices = make_prices()

        X, y = build_dataset(prices, horizon=5)

        assert not X.isna().any().any()
        assert np.isfinite(X.to_numpy()).all()
        assert not y.isna().any()

    def test_horizon_rows_are_not_labeled_as_zero(self) -> None:
        """Rows without a future price must never become class 0."""
        prices = make_prices(120)
        horizon = 5

        X, y = build_dataset(prices, horizon=horizon)

        # The final horizon rows have no future observations.
        expected_last_labeled_index = prices.index[-horizon - 1]

        assert X.index[-1] == expected_last_labeled_index
        assert y.index[-1] == expected_last_labeled_index

    def test_target_is_binary(self) -> None:
        prices = make_prices()

        _, y = build_dataset(prices, horizon=5)

        assert set(y.unique()).issubset({0, 1})

    def test_horizon_changes_number_of_labeled_rows(self) -> None:
        prices = make_prices()

        X1, y1 = build_dataset(prices, horizon=1)
        X5, y5 = build_dataset(prices, horizon=5)

        assert len(X1) > len(X5)
        assert len(y1) == len(X1)
        assert len(y5) == len(X5)
