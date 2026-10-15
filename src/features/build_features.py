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
6.4: macro data (lagged). 6.5: profile features. build_features() runs all of them.
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


# ---------- step 6.4: macro data, lagged so it was published at forecast time ----------

MACRO_FEATURES = ["inflation_de", "ecb_rate"]


def add_macro_features(table: pd.DataFrame, macro: pd.DataFrame | None = None) -> tuple[pd.DataFrame, bool]:
    """Add German inflation and the ECB rate of month t-1 (one month before the origin).

    Inflation for month t is only published around the middle of month t+1, so at the end of
    month t the latest figure a real forecaster has is t-1; the rate is lagged the same way.
    macro defaults to the cached ECB data (load_macro). If there is none, the table is returned
    unchanged and the second value is False, so the pipeline can say the features were skipped.
    """
    if macro is None:
        from src.ingestion.macro_api import MacroDataUnavailable, load_macro
        try:
            macro = load_macro(refresh=False)
        except MacroDataUnavailable:
            return table, False

    lagged = macro[["month"] + MACRO_FEATURES].copy()
    lagged["month"] = pd.PeriodIndex(lagged["month"], freq="M") + 1   # value of t-1 joins origin month t
    out = table.merge(lagged, on="month", how="left")
    return out.sort_values(KEYS, ignore_index=True), True


# ---------- step 6.5: user profile features (from the user's own past months) ----------

FIXED_COST_SERIES = ["expenses:Rent", "expenses:Utilities", "expenses:Subscriptions"]
PROFILE_FEATURES = ["fixed_share", "income_volatility"]


def add_profile_features(table: pd.DataFrame) -> pd.DataFrame:
    """Two descriptions of the user's budget, from months <= t only:

    fixed_share        rent + utilities + subscriptions as a share of total spending, last 3 months
    income_volatility  std / mean of monthly income over the last 6 months (~0.03 salary, ~0.6 freelance)

    The synthetic archetype is deliberately NOT a feature: real app users have no such label,
    so the model has to recognise salaried vs. freelance behaviour from these numbers.
    """
    wide = table.pivot_table(index=["user_id", "month"], columns="series", values="value", aggfunc="sum")
    wide = wide.sort_index()
    by_user = wide.groupby(level="user_id", sort=False)

    fixed = wide.reindex(columns=FIXED_COST_SERIES).fillna(0).sum(axis=1)
    fixed_3 = fixed.groupby(level="user_id", sort=False).transform(lambda s: s.rolling(3, min_periods=3).sum())
    spend_3 = by_user[EXPENSES].transform(lambda s: s.rolling(3, min_periods=3).sum())
    income_mean = by_user[INCOME].transform(lambda s: s.rolling(6, min_periods=6).mean())
    income_std = by_user[INCOME].transform(lambda s: s.rolling(6, min_periods=6).std())

    profile = pd.DataFrame({
        "fixed_share": (fixed_3 / spend_3.where(spend_3 > 0)).round(4),
        "income_volatility": (income_std / income_mean.where(income_mean > 0)).round(4),
    }).reset_index()
    out = table.merge(profile, on=["user_id", "month"], how="left")
    return out.sort_values(KEYS, ignore_index=True)


# ---------- everything in one call ----------

BASE_FEATURES = list(LAGS) + [f"roll_mean_{w}" for w in ROLL_WINDOWS] + ["roll_std_6", "history_months"]


def build_features(monthly: pd.DataFrame, macro: pd.DataFrame | None = None,
                   country: str | None = None) -> tuple[pd.DataFrame, dict]:
    """The full feature table from the user x month x category table (step 5.7).

    Returns (table, info): info["feature_columns"] lists the model inputs (plus the categorical
    'series'), info["macro"] says whether the ECB features could be added.
    """
    table = add_targets(build_series(monthly))
    table = add_lag_features(table)
    table = add_calendar_features(table, country=country)
    table, macro_added = add_macro_features(table, macro)
    table = add_profile_features(table)
    features = BASE_FEATURES + CALENDAR_FEATURES + (MACRO_FEATURES if macro_added else []) + PROFILE_FEATURES
    return table, {"feature_columns": features, "categorical": ["series"], "macro": macro_added}
