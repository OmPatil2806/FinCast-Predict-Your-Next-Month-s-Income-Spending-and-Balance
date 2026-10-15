"""Tests for the forecasting feature table."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import add_calendar_features, add_lag_features, add_targets, build_series
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


def test_lag_and_rolling_features_use_only_the_past():
    table = add_lag_features(add_targets(build_series(make_monthly(15))))
    g = table[table["series"] == "expenses:Groceries"].reset_index(drop=True)
    # groceries in month i = 100 + 10 * i
    row = g.iloc[13]                                   # origin = 14th month, target = 15th month
    assert row["value"] == 230.0 and row["target"] == 240.0
    assert row["lag_1"] == 230.0                       # month t
    assert row["lag_2"] == 220.0 and row["lag_3"] == 210.0
    assert row["lag_12"] == 120.0                      # t-11: same calendar month as the target, a year earlier
    assert row["roll_mean_3"] == pytest.approx((210 + 220 + 230) / 3)
    assert row["roll_mean_6"] == pytest.approx(sum(100 + 10 * i for i in range(8, 14)) / 6)
    assert row["roll_std_6"] == pytest.approx(pd.Series([180, 190, 200, 210, 220, 230]).std())
    assert row["history_months"] == 14


def test_short_history_gives_nan_not_future_values():
    table = add_lag_features(add_targets(build_series(make_monthly(15))))
    g = table[table["series"] == "expenses:Groceries"].reset_index(drop=True)
    assert np.isnan(g.loc[0, "lag_2"]) and np.isnan(g.loc[1, "roll_mean_3"])
    assert g["lag_12"].isna().sum() == 11              # needs 12 months of history
    assert g.loc[11, "lag_12"] == 100.0                # first month with a full year behind it


def test_calendar_features_describe_the_target_month():
    table = add_calendar_features(add_targets(build_series(make_monthly(13))), country="DE")
    row = table[(table["series"] == "income") & (table["month"] == pd.Period("2025-11"))].iloc[0]
    assert row["target_month"] == pd.Period("2025-12")
    assert row["holidays"] == 2                        # Christmas Day + Boxing Day
    assert row["working_days"] == 21 and row["days_in_month"] == 31
    assert row["month_cos"] == pytest.approx(np.cos(2 * np.pi * 11 / 12))
    # December and January are neighbours on the circle; June is opposite
    dec, jan, jun = [(np.sin(2 * np.pi * (m - 1) / 12), np.cos(2 * np.pi * (m - 1) / 12)) for m in (12, 1, 6)]
    assert np.hypot(dec[0] - jan[0], dec[1] - jan[1]) < np.hypot(jun[0] - jan[0], jun[1] - jan[1])
    assert len(table) == len(build_series(make_monthly(13)))   # merge adds columns, never rows
