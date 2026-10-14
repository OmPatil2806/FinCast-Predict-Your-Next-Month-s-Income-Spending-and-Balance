"""Tests for validate_transactions: good data passes, every kind of bad data is reported."""

import pandas as pd
import pytest

from src.preprocessing.cleaning import DataValidationError, validate_transactions

TODAY = pd.Timestamp("2026-10-14")


def make_data(**overrides) -> pd.DataFrame:
    """Two users with one transaction per month from Jan to Jun 2026."""
    months = pd.date_range("2026-01-01", periods=6, freq="MS") + pd.Timedelta(days=14)   # 15 Jan ... 15 Jun
    rows = [{"user_id": u, "date": d, "description": "REWE", "amount": -20.0}
            for u in ("user_001", "user_002") for d in months]
    df = pd.DataFrame(rows)
    for column, (index, value) in overrides.items():
        df.loc[index, column] = value
    return df


def test_good_data_passes():
    report = validate_transactions(make_data(), today=TODAY)
    assert report["rows"] == 12
    assert report["users"] == 2


@pytest.mark.parametrize("column, value, message", [
    ("date", pd.Timestamp("2027-01-01"), "dated after 2026-10-14"),
    ("description", "   ", "empty description"),
    ("amount", 75_000.0, "beyond ±€50,000"),
    ("amount", None, "missing or non-numeric amount"),
])
def test_each_problem_is_reported(column, value, message):
    with pytest.raises(DataValidationError, match=message):
        validate_transactions(make_data(**{column: (3, value)}), today=TODAY)


def test_gap_longer_than_two_months_is_reported():
    df = make_data()
    df = df[~((df["user_id"] == "user_002") & df["date"].dt.month.isin([2, 3, 4]))]  # skips Feb-Apr
    with pytest.raises(DataValidationError, match=r"1 users with a gap longer than 2 months: user_002 \(3 months\)"):
        validate_transactions(df, today=TODAY)


def test_gap_of_two_months_is_allowed():
    df = make_data()
    df = df[~((df["user_id"] == "user_002") & df["date"].dt.month.isin([2, 3]))]
    assert validate_transactions(df, today=TODAY)["users"] == 2


def test_all_problems_are_listed_together():
    df = make_data(description=(0, ""), amount=(5, 99_999.0))
    with pytest.raises(DataValidationError) as err:
        validate_transactions(df, today=TODAY)
    assert len(err.value.problems) == 2


def test_works_without_user_id_column():
    df = make_data().drop(columns="user_id")
    assert validate_transactions(df, today=TODAY)["users"] == 1
