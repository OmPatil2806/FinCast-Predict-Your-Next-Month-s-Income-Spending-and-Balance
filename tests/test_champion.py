"""Tests for the champion rule (step 7.8)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.select_champion import (
    apply_champions, choose_champions, choose_width, save_champions, select_family,
)

FAST = {"n_estimators": 30, "num_leaves": 8, "min_child_samples": 5}


def test_lightgbm_needs_the_full_threshold():
    per_series = pd.DataFrame({"series": ["a", "b", "c"], "model_mae": [80.0, 91.0, 0.5],
                               "best_baseline": ["mean_3", "last_month", "last_month"],
                               "baseline_mae": [100.0, 100.0, 0.0], "improvement": [0.20, 0.09, 0.0]})
    out = choose_champions(per_series, threshold=0.10)
    assert out["champion"].tolist() == ["lightgbm", "last_month", "last_month"]
    assert "baseline kept" in out.loc[1, "reason"]


@pytest.mark.parametrize("factor, raw, calibrated, expected", [
    (1.3, 0.70, 0.79, 1.3),       # calibration helped -> keep it
    (1.3, 0.85, 0.93, 1.0),       # calibration overshot -> raw width
    (1.0, 0.82, 0.82, 1.0),
])
def test_width_factor_only_kept_when_it_helps(factor, raw, calibrated, expected):
    chosen, reason = choose_width(factor, {"coverage_raw_check": raw, "coverage_calibrated_check": calibrated})
    assert chosen == expected and reason


def test_baseline_champion_point_sits_inside_the_range():
    rows = pd.DataFrame({"series": ["expenses:Rent", "expenses:Health"], "lag_1": [1200.0, 50.0],
                         "lag_12": [1100.0, 40.0], "roll_mean_3": [1200.0, 45.0]})
    out = apply_champions(rows, lower=np.array([900.0, 20.0]), median=np.array([1000.0, 30.0]),
                          upper=np.array([1100.0, 60.0]),
                          champions={"expenses:Rent": "last_month", "expenses:Health": "lightgbm"}, factor=1.0)
    assert out["predicted"].tolist() == [1200.0, 30.0]
    assert out["champion"].tolist() == ["last_month", "lightgbm"]
    assert (out["lower"] <= out["predicted"]).all() and (out["predicted"] <= out["upper"]).all()
    assert out.loc[0, "upper"] == 1200.0                      # stretched to hold the baseline point


@pytest.fixture(scope="module")
def data():
    transactions, users = generate_dataset(n_users=6, months=26, start="2024-01-01", seed=13)
    table, info = build_features(aggregate_user_month_category(transactions))
    return table, info["feature_columns"], users


def test_select_family_end_to_end(data, tmp_path):
    table, features, users = data
    result = select_family(table, "spend", features, 0.10, FAST, users)
    champions = result["champions"]
    assert set(champions["series"]) == set(table.loc[table["series"] != "income", "series"])
    assert set(champions["champion"]) <= {"lightgbm", "last_month", "same_month_last_year", "mean_3"}
    assert result["system"]["coverage"] > 0 and result["check"]["series"] == len(champions)
    saved = json.loads(save_champions({"spend": result}, 0.10, tmp_path / "c.json").read_text())
    assert set(saved["families"]["spend"]["series"]) == set(champions["series"])
