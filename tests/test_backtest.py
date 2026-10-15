"""Tests for the time-series backtest folds."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.backtest import backtest_folds, holdout_user_folds
from src.features.build_features import build_features
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category

MACRO = pd.DataFrame({"month": pd.period_range("2023-01", periods=48, freq="M"),
                      "inflation_de": 2.0, "ecb_rate": 2.5})


def build(monthly):
    return build_features(monthly, macro=MACRO, country="DE")


@pytest.fixture(scope="module")
def monthly():
    transactions, _ = generate_dataset(n_users=10, months=24, start="2024-01-01", seed=3)
    return aggregate_user_month_category(transactions)


@pytest.fixture(scope="module")
def table(monthly):
    return build(monthly)[0]


def test_folds_never_train_on_the_test_month_or_later(table):
    folds = backtest_folds(table, min_train_months=12)
    for fold in folds:
        train_months = table["target_month"].iloc[fold.train_idx]
        test_months = table["target_month"].iloc[fold.test_idx]
        assert (train_months < fold.test_month).all()
        assert (test_months == fold.test_month).all()
        assert table["target"].iloc[fold.train_idx].notna().all()
        assert table["target"].iloc[fold.test_idx].notna().all()


def test_folds_walk_forward_one_month_at_a_time(table):
    folds = backtest_folds(table, min_train_months=12)
    months = [f.test_month for f in folds]
    # 24 months of data -> 23 target months; the first 12 are training only
    assert len(folds) == 23 - 12
    assert months == sorted(months)
    assert all(b == a + 1 for a, b in zip(months, months[1:]))    # consecutive months
    sizes = [len(f.train_idx) for f in folds]
    assert sizes == sorted(sizes)                                  # expanding window


def test_holdout_users_are_unseen_and_history_is_truncated(monthly):
    folds = holdout_user_folds(monthly, build, users_share=0.3, histories=(3, 6, 12))
    for fold in folds:
        assert not set(fold.test_users) & set(fold.train["user_id"])
        assert set(fold.test["user_id"]) == set(fold.test_users)
        assert (fold.test["history_months"] == fold.history_months).all()
        assert (fold.test["month"] == fold.origin).all()
        assert (fold.train["target_month"] <= fold.origin).all()
    short = folds[0].test
    assert short["lag_12"].isna().all() and short["roll_mean_6"].isna().all()   # only 3 months known
    assert folds[2].test["roll_mean_6"].notna().all()


def test_truncated_features_match_the_full_history_where_they_overlap(monthly):
    """With 12 months of history, the 3-month features must equal those built from the full history."""
    fold = holdout_user_folds(monthly, build, users_share=0.3, histories=(12,))[0]
    full, _ = build(monthly[monthly["user_id"].isin(fold.test_users)])
    full = full[full["month"] == fold.origin].set_index(["user_id", "series"])
    test = fold.test.set_index(["user_id", "series"])
    for column in ["lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_mean_6", "target"]:
        np.testing.assert_allclose(test[column], full.loc[test.index, column])
