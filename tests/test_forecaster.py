"""Tests for the LightGBM forecaster plumbing (small, fast models)."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.train_forecaster import (
    ForecastModel, backtest_family, backtest_family_quantiles, calibrate_width, compare_with_baselines, family_rows,
    from_model_output, to_model_target, widen,
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


def test_quantile_ranges_are_ordered(data):
    table, features, _ = data
    predictions = backtest_family_quantiles(table, "spend", "raw", features, params=FAST, last_n_folds=2)
    assert predictions["target_month"].nunique() == 2
    assert (predictions["lower"] <= predictions["predicted"]).all()
    assert (predictions["predicted"] <= predictions["upper"]).all()


def test_widen_scales_distance_from_median_and_keeps_lower_non_negative():
    p = pd.DataFrame({"lower": [80.0, 5.0], "predicted": [100.0, 10.0], "upper": [130.0, 20.0]})
    wider = widen(p, 2.0)
    assert wider["lower"].tolist() == [60.0, 0.0]          # 10 - 2*5 = 0, never below zero
    assert wider["upper"].tolist() == [160.0, 30.0]
    assert widen(p, 1.0)[["lower", "upper"]].equals(p[["lower", "upper"]])


def test_calibrate_width_widens_a_too_narrow_range():
    rng = np.random.default_rng(0)
    months = pd.period_range("2025-01", periods=10, freq="M")
    n = 400
    actual = rng.normal(100, 10, size=len(months) * n)
    p = pd.DataFrame({"target_month": np.repeat(months, n), "actual": actual, "predicted": 100.0,
                      "lower": 95.0, "upper": 105.0})            # ~38 % coverage, far too narrow
    factor, report = calibrate_width(p, target=0.80)
    assert factor > 2
    assert report["coverage_raw_check"] < 0.5
    assert abs(report["coverage_calibrated_check"] - 0.80) < 0.05
    assert report["check_months"].startswith(str(months[5]))
