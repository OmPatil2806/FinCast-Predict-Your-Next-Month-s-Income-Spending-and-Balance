"""Tests for the baseline forecasts and their score file."""

import json

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.train_baselines import BASELINES, evaluate, save_scores, series_group

MACRO = pd.DataFrame({"month": pd.period_range("2023-01", periods=48, freq="M"), "inflation_de": 2.0, "ecb_rate": 2.5})


def test_baseline_formulas_and_fallbacks():
    t = pd.DataFrame({"lag_1": [100.0, 100.0], "lag_12": [80.0, np.nan], "roll_mean_3": [90.0, np.nan]})
    assert BASELINES["last_month"](t).tolist() == [100.0, 100.0]
    assert BASELINES["same_month_last_year"](t).tolist() == [80.0, 100.0]   # falls back to last month
    assert BASELINES["mean_3"](t).tolist() == [90.0, 100.0]


def test_series_group():
    assert series_group(pd.Series(["income", "expenses", "expenses:Rent"])).tolist() == ["income", "expenses", "category"]


@pytest.fixture(scope="module")
def results():
    transactions, users = generate_dataset(n_users=8, months=24, start="2024-01-01", seed=5)
    monthly = aggregate_user_month_category(transactions)
    table, _ = build_features(monthly, macro=MACRO, country="DE")
    return evaluate(table, users=users, monthly=monthly)


def test_evaluate_scores_every_baseline_on_every_test_month(results):
    assert results["test_months"] == 11                       # 23 target months, first 12 only train
    assert set(results["overall"]["baseline"]) == set(BASELINES)
    per_baseline = results["predictions"].groupby("baseline").size()
    assert per_baseline.nunique() == 1                        # same rows for every baseline
    assert set(results["new_users"]["history_months"]) == {3, 6, 12}
    # salaried income barely changes, so "last month" is close to perfect there
    salaried = results["by_archetype"].query("series_group == 'income' and archetype == 'salaried_single' and baseline == 'last_month'")
    if len(salaried):
        assert salaried["mae"].iloc[0] < 50


def test_best_by_series_picks_lowest_mae(results, tmp_path):
    by_series = results["by_series"]
    for row in results["best_by_series"].itertuples():
        assert row.mae == pytest.approx(by_series[by_series["series"] == row.series]["mae"].min())
    path = save_scores(results, tmp_path / "scores.json")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert set(saved["best_by_series"]) == set(results["best_by_series"]["series"])
    assert saved["metric"] == "mae"
