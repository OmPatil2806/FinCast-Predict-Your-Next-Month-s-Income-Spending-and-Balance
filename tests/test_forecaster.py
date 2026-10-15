"""Tests for the LightGBM forecaster plumbing (small, fast models)."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.train_forecaster import (
    ForecastModel, backtest_family, compare_with_baselines, family_rows, from_model_output, to_model_target,
)

MACRO = pd.DataFrame({"month": pd.period_range("2023-01", periods=48, freq="M"), "inflation_de": 2.0, "ecb_rate": 2.5})
FAST = {"n_estimators": 30, "num_leaves": 8, "min_child_samples": 5}


@pytest.fixture(scope="module")
def data():
    transactions, users = generate_dataset(n_users=8, months=24, start="2024-01-01", seed=11)
    table, info = build_features(aggregate_user_month_category(transactions), macro=MACRO, country="DE")
    return table, info["feature_columns"], users


def test_relative_target_round_trips(data):
    table, _, _ = data
    rows = family_rows(table, "spend").dropna(subset=["target"]).head(500)
    back = from_model_output(rows, to_model_target(rows, "relative").to_numpy(), "relative")
    np.testing.assert_allclose(back, rows["target"].clip(lower=0), rtol=1e-9, atol=1e-6)


def test_families_split_income_from_spending(data):
    table, _, _ = data
    assert set(family_rows(table, "income")["series"]) == {"income"}
    assert "income" not in set(family_rows(table, "spend")["series"])


@pytest.mark.parametrize("family, mode", [("income", "raw"), ("spend", "relative")])
def test_model_fits_and_predicts_non_negative(data, family, mode):
    table, features, _ = data
    rows = family_rows(table, family).dropna(subset=["target"])
    model = ForecastModel(family, mode, 0.5, features, sorted(rows["series"].unique()), FAST).fit(rows)
    predicted = model.predict(rows)
    assert len(predicted) == len(rows) and (predicted >= 0).all()


def test_backtest_trains_only_on_the_past(data):
    table, features, users = data
    seen = []
    predictions = backtest_family(table, "spend", "raw", features, params=FAST, last_n_folds=3,
                                  on_fold=lambda n, month, p: seen.append(month))
    assert len(seen) == 3 and seen == sorted(seen)
    assert set(predictions["target_month"]) == set(seen)
    comparison = compare_with_baselines(predictions, table, users)
    assert set(comparison["per_series"]["series"]) == set(predictions["series"])
    assert {"model_mae", "baseline_mae", "improvement"} <= set(comparison["per_series"].columns)
