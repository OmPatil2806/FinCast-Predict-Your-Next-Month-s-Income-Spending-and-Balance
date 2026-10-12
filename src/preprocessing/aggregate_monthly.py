"""Turn transactions into monthly series.

First version (step 3.7): totals per month and spending per category for the Overview page.
Step 5.7 extends this to one row per user, month and category for the forecasting models.
"""

import pandas as pd

MONTHLY_COLUMNS = ["month", "income", "expenses", "savings", "savings_rate"]
UNCATEGORISED = "Uncategorised"


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per calendar month with income, expenses, savings and savings_rate.

    Expenses are returned as positive numbers. Months without transactions between the
    first and last month are filled with zeros so charts and models see no gaps.
    month is a pandas Period ('2026-07'); savings_rate is a fraction (0.55 = 55%) and
    is NaN for months without income.
    """
    if df.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    data = df[["date", "amount"]].copy()
    data["month"] = pd.to_datetime(data["date"]).dt.to_period("M")
    data["income"] = data["amount"].clip(lower=0)
    data["expenses"] = (-data["amount"]).clip(lower=0)

    monthly = data.groupby("month")[["income", "expenses"]].sum()
    all_months = pd.period_range(monthly.index.min(), monthly.index.max(), freq="M")
    monthly = monthly.reindex(all_months, fill_value=0.0).rename_axis("month").reset_index()

    monthly["savings"] = monthly["income"] - monthly["expenses"]
    income = monthly["income"].where(monthly["income"] > 0)
    monthly["savings_rate"] = monthly["savings"] / income
    monthly[["income", "expenses", "savings"]] = monthly[["income", "expenses", "savings"]].round(2)
    return monthly[MONTHLY_COLUMNS]


def aggregate_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Total spending per category (positive numbers), largest first.

    Only expenses are counted. Rows without a category are grouped as 'Uncategorised'.
    """
    columns = ["category", "expenses"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    spending = df[df["amount"] < 0].copy()
    if spending.empty:
        return pd.DataFrame(columns=columns)
    if "category" in spending:
        category = spending["category"].fillna("").astype(str).str.strip()
    else:
        category = pd.Series("", index=spending.index)
    spending["category"] = category.replace("", UNCATEGORISED)
    spending["expenses"] = -spending["amount"]
    result = spending.groupby("category", as_index=False)["expenses"].sum()
    result["expenses"] = result["expenses"].round(2)
    return result.sort_values("expenses", ascending=False, ignore_index=True)
