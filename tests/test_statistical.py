"""Tests for the ETS / SARIMAX comparison (step 7.5)."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.train_statistical import MODELS, backtest_statistical, compare, forecast_one

FAST = {"n_estimators": 30, "num_leaves": 8, "min_child_samples": 5}


@pytest.fixture(scope="module")
def data():
    transactions, users = generate_dataset(n_users=6, months=30, start="2024-01-01", seed=5)
    table, info = build_features(aggregate_user_month_category(transactions))
    return table, info["feature_columns"], users


@pytest.mark.parametrize("model", list(MODELS))
def test_follows_a_seasonal_series(model):
    history = (1000 + 200 * (np.arange(35) % 12 == 11)).astype(float)   # December bonus every year
    december, fallback = forecast_one(history, model)                    # month 35 is a December
    october, _ = forecast_one(history[:-1], model)                       # month 34 is an October
    assert not fallback and december > october + 25


@pytest.mark.parametrize("model", list(MODELS))
def test_flat_or_short_series_falls_back_to_last_month(model):
    assert forecast_one(np.full(24, 2500.0), model) == (2500.0, True)
    assert forecast_one(np.array([10.0, 20.0, 30.0]), model) == (30.0, True)


def test_forecasts_never_negative():
    history = np.array([500, 400, 300, 200, 100, 50, 20, 10, 5, 1, 0.5, 0.1], dtype=float)
    for model in MODELS:
        assert forecast_one(history, model)[0] >= 0


def test_uses_only_history_up_to_the_origin(data):
    table, _, _ = data
    before = backtest_statistical(table, "expenses", "ets", last_n_folds=2, n_jobs=1)
    future = table["month"] > before["month"].max()
    scrambled = table.copy()
    scrambled.loc[future, ["value", "lag_1"]] = 1e9
    after = backtest_statistical(scrambled, "expenses", "ets", last_n_folds=2, n_jobs=1)
    pd.testing.assert_series_equal(before["predicted"], after["predicted"])


def test_compare_scores_every_forecaster_on_the_same_rows(data):
    table, features, users = data
    results = compare(table, features, users, last_n_folds=2, n_jobs=1, lgbm_params=FAST)
    for series in ("income", "expenses"):
        r = results[series]
        assert set(r["overall"]["model"]) == {"last_month", "same_month_last_year", "mean_3", "ets", "sarimax", "lightgbm"}
        assert (r["overall"]["n"] == r["n"]).all()
        assert r["joined"][["ets", "sarimax", "lightgbm"]].notna().all().all()
