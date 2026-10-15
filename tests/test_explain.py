"""Tests for the SHAP explanations of the forecaster (step 7.7), with tiny models."""

import numpy as np
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.explain_forecaster import SCALE, explain_family, save_plots

FAST = {"n_estimators": 30, "num_leaves": 8, "min_child_samples": 5}


@pytest.fixture(scope="module")
def data():
    transactions, _ = generate_dataset(n_users=6, months=20, start="2024-01-01", seed=9)
    table, info = build_features(aggregate_user_month_category(transactions))
    return table, info["feature_columns"]


@pytest.fixture(scope="module", params=["income", "spend"])
def result(request, data):
    table, features = data
    return explain_family(table, request.param, features, FAST)


def test_shap_values_add_up_to_the_model_output(result):
    assert result["additivity_error"] < 1e-6


def test_explains_only_the_last_test_month(result, data):
    table, _ = data
    assert set(result["rows"]["target_month"].astype(str)) == {result["test_month"]}
    assert result["test_month"] == str(table.dropna(subset=["target"])["target_month"].max())


def test_importance_covers_every_input_and_shares_sum_to_one(result, data):
    _, features = data
    expected = set(features) | ({"series"} if result["family"] == "spend" else set())
    assert set(result["importance"]["feature"]) == expected
    assert np.isclose(result["importance"]["share"].sum(), 1.0)
    assert result["importance"]["mean_abs_shap"].is_monotonic_decreasing


def test_example_is_a_forecast_of_the_main_series(result):
    series = result["rows"].loc[result["example"], "series"]
    assert series == ("income" if result["family"] == "income" else "expenses")
    assert result["explanation"].values.shape == (len(result["rows"]), len(result["importance"]))
    assert SCALE[result["mode"]] in (1.0, 100.0)


def test_plots_are_written(result, tmp_path):
    paths = save_plots(result, tmp_path)
    assert len(paths) == 4
    assert all(p.exists() and p.stat().st_size > 5_000 for p in paths)
