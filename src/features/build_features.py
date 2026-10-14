"""Forecasting features: everything the models see, built the same way for training and the live app.

Shape: one row per user x forecast origin month t x series, where a series is what we forecast:
    income                 total monthly income
    expenses               total monthly spending
    expenses:<Category>    spending in one category (e.g. expenses:Groceries)

A row is "the forecast made at the end of month t for the target month t+1":
    value         the series in month t (already known)
    target        the series in month t+1 (what the model must predict; NaN for the last month)
Feature names count back from the target month: lag_1 = month t, lag_12 = same month last year.

Step 6.1: series + targets. Step 6.2: lag and rolling features. (6.3-6.5 add calendar, macro, profile.)
"""

import pandas as pd

from src.preprocessing.aggregate_monthly import TOTAL

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
