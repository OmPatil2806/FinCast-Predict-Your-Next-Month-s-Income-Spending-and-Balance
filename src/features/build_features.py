"""Forecasting features: everything the models see, built the same way for training and the live app.

Shape: one row per user x forecast origin month t x series, where a series is what we forecast:
    income                 total monthly income
    expenses               total monthly spending
    expenses:<Category>    spending in one category (e.g. expenses:Groceries)

A row is "the forecast made at the end of month t for the target month t+1":
    value         the series in month t (already known)
    target        the series in month t+1 (what the model must predict; NaN for the last month)
Feature names count back from the target month: lag_1 = month t, lag_12 = same month last year.

Step 6.1: series + targets. 6.2: lags and rolling stats. 6.3: calendar of the target month.
(6.4-6.5 add macro and profile features.)
"""

import numpy as np
import pandas as pd

from src.preprocessing.aggregate_monthly import TOTAL
from src.utils.helpers import holidays_per_month

INCOME = "income"
EXPENSES = "expenses"
KEYS = ["user_id", "series", "month"]


# ---------- step 6.1: series and targets ----------

def build_series(monthly: pd.DataFrame) -> pd.DataFrame:
    """Long table user_id, month, series, value from the user x month x category table (step 5.7)."""
    data = monthly.copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M") if not isinstance(
        data["month"].dtype, pd.PeriodDtype) else data["month"]

    totals = data[data["category"] == TOTAL]
    income = totals.assign(series=INCOME, value=totals["income"])
    expenses = totals.assign(series=EXPENSES, value=totals["expenses"])
    categories = data[(data["category"] != TOTAL) & (data["category"] != "Income")]
    per_category = categories.assign(series=EXPENSES + ":" + categories["category"], value=categories["expenses"])

    series = pd.concat([income, expenses, per_category], ignore_index=True)[["user_id", "month", "series", "value"]]
    return series.sort_values(KEYS, ignore_index=True)


def add_targets(series: pd.DataFrame) -> pd.DataFrame:
    """Add target_month (t+1) and target (the value in t+1) within each user and series."""
    out = series.sort_values(KEYS, ignore_index=True)
    grouped = out.groupby(["user_id", "series"], sort=False)["value"]
    out["target"] = grouped.shift(-1)
    out["target_month"] = out["month"] + 1
    return out


# ---------- step 6.2: lag and rolling features (past data only) ----------

LAGS = {"lag_1": 0, "lag_2": 1, "lag_3": 2, "lag_12": 11}   # name -> months before the origin month t
ROLL_WINDOWS = (3, 6)


def add_lag_features(table: pd.DataFrame) -> pd.DataFrame:
    """Lags and rolling statistics that only use months <= t (the forecast origin).

    lag_1 = value in month t, lag_2 = t-1, lag_3 = t-2, lag_12 = t-11 (same calendar month as the
    target, one year earlier). roll_mean_3 / roll_mean_6 / roll_std_6 cover the last 3 or 6 months
    up to and including t. history_months counts the months observed so far (for cold start).
    Rows without enough history get NaN, never a value from the future.
    """
    out = table.sort_values(KEYS, ignore_index=True)
    grouped = out.groupby(["user_id", "series"], sort=False)["value"]
    for name, shift in LAGS.items():
        out[name] = grouped.shift(shift)
    for window in ROLL_WINDOWS:
        out[f"roll_mean_{window}"] = grouped.transform(lambda s, w=window: s.rolling(w, min_periods=w).mean())
    out["roll_std_6"] = grouped.transform(lambda s: s.rolling(6, min_periods=6).std())
    out["history_months"] = grouped.cumcount() + 1
    return out


# ---------- step 6.3: calendar of the target month (known in advance) ----------

CALENDAR_FEATURES = ["month_sin", "month_cos", "holidays", "working_days", "days_in_month"]


def add_calendar_features(table: pd.DataFrame, country: str | None = None) -> pd.DataFrame:
    """Calendar facts about the TARGET month t+1. This is the one place a feature describes a month
    after the origin: the calendar of next month is known in advance, so it is not leakage.

    month_sin / month_cos place the month on a circle (December sits next to January).
    """
    out = table.copy()
    target_month = out["target_month"]
    angle = 2 * np.pi * (target_month.dt.month - 1) / 12
    out["month_sin"] = np.sin(angle).round(6)
    out["month_cos"] = np.cos(angle).round(6)

    calendar = holidays_per_month(target_month.min(), target_month.max(), country=country)
    calendar = calendar.rename(columns={"month": "target_month"})
    out = out.merge(calendar[["target_month", "holidays", "working_days", "days_in_month"]],
                    on="target_month", how="left")
    return out.sort_values(KEYS, ignore_index=True)
