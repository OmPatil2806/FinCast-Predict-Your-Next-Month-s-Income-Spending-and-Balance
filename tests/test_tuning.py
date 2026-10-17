"""Tests for the Optuna tuning of the LightGBM forecaster (step 7.6), with tiny models."""

import json

import pandas as pd
import pytest

from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.train_forecaster import DEFAULT_PARAMS, family_rows
from src.training.tune_forecaster import save_params, search_space_keys, split_folds, tune_family

FAST = {"n_estimators": 20, "num_leaves": 8, "min_child_samples": 5}


@pytest.fixture(scope="module")
def data():
    transactions, users = generate_dataset(n_users=6, months=24, start="2024-01-01", seed=3)
    table, info = build_features(aggregate_user_month_category(transactions))
    return table, info["feature_columns"]


def test_holdout_months_come_after_tune_months_and_never_overlap(data):
    table, _ = data
    tune, holdout = split_folds(family_rows(table, "spend"), tune_folds=3, holdout_folds=2)
    assert len(tune) == 3 and len(holdout) == 2
    assert max(f.test_month for f in tune) < min(f.test_month for f in holdout)
    with pytest.raises(ValueError):
        split_folds(family_rows(table, "spend"), tune_folds=50, holdout_folds=2)


def test_first_trial_is_the_default_and_params_are_saved(data, tmp_path, monkeypatch):
    table, features = data
    import src.training.tune_forecaster as tf
    small_space = lambda trial: {"n_estimators": trial.suggest_int("n_estimators", 10, 30, step=10),
                                 "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                                 "num_leaves": trial.suggest_int("num_leaves", 4, 16),
                                 "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
                                 "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                                 "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                                 "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}
    monkeypatch.setattr(tf, "search_space", small_space)
    monkeypatch.setattr(tf, "DEFAULT_PARAMS", {**DEFAULT_PARAMS, **FAST, "num_leaves": 8, "min_child_samples": 5})
    result = tune_family(table, "income", features, trials=3, tune_folds=2, holdout_folds=2, log=False)
    assert result["trials"] == 3
    assert result["best_tune_mae"] <= result["default_tune_mae"]
    assert result["use_tuned"] == (result["tuned_holdout_mae"] < result["default_holdout_mae"])

    path = save_params({"income": result}, tmp_path / "params.json")
    saved = json.loads(path.read_text(encoding="utf-8"))["families"]["income"]
    assert set(saved["params"]) == set(search_space_keys())
    assert saved["source"] == ("tuned" if result["use_tuned"] else "default")


def test_defaults_are_kept_when_tuning_loses_on_holdout(tmp_path):
    result = {"mode": "raw", "use_tuned": False, "best_params": {"n_estimators": 999},
              "holdout_months": "2026-04 to 2026-09", "default_holdout_mae": 10.0, "tuned_holdout_mae": 11.0}
    saved = json.loads(save_params({"spend": result}, tmp_path / "p.json").read_text(encoding="utf-8"))["families"]["spend"]
    assert saved["source"] == "default"
    assert saved["params"]["n_estimators"] == DEFAULT_PARAMS["n_estimators"]
