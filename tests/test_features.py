"""Tests for the forecasting feature table."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_targets, build_series
from src.preprocessing.aggregate_monthly import aggregate_user_month_category


def make_monthly(n_months: int = 15) -> pd.DataFrame:
    """One user, salary every month and groceries growing by 10 per month."""
    months = pd.period_range("2025-01", periods=n_months, freq="M")
    rows = []
    for i, month in enumerate(months):
        day = month.to_timestamp()
        rows.append({"user_id": "u1", "date": day, "amount": 3000.0, "category": "Income"})
        rows.append({"user_id": "u1", "date": day, "amount": -(100.0 + 10 * i), "category": "Groceries"})
        rows.append({"user_id": "u1", "date": day, "amount": -800.0, "category": "Rent"})
    return aggregate_user_month_category(pd.DataFrame(rows))


def test_series_are_income_expenses_and_categories():
    series = build_series(make_monthly(3))
    assert sorted(series["series"].unique()) == ["expenses", "expenses:Groceries", "expenses:Rent", "income"]
    jan = series[series["month"] == pd.Period("2025-01")].set_index("series")["value"]
    assert jan["income"] == 3000.0
    assert jan["expenses"] == 900.0
    assert jan["expenses:Groceries"] == 100.0


def test_target_is_next_month_value():
    table = add_targets(build_series(make_monthly(3)))
    groceries = table[table["series"] == "expenses:Groceries"]
    assert groceries["value"].tolist() == [100.0, 110.0, 120.0]
    assert groceries["target"].tolist()[:2] == [110.0, 120.0]
    assert np.isnan(groceries["target"].iloc[-1])                 # nothing to predict after the last month
    assert (groceries["target_month"] == groceries["month"] + 1).all()
