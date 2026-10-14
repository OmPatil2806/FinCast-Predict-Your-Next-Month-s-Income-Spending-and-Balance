"""Turn transactions into monthly series.

Step 3.7: totals per month and spending per category for one user (Overview page).
Step 5.7: aggregate_user_month_category() - one row per user, month and category for the
forecasting models, including month totals (category "ALL") and the running end balance.
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


# ---------- step 5.7: user x month x category table for the models ----------

TOTAL = "ALL"
LONG_COLUMNS = ["user_id", "month", "category", "income", "expenses", "savings",
                "n_transactions", "savings_rate", "balance_end"]


def aggregate_user_month_category(transactions: pd.DataFrame,
                                  opening_balances: dict | float = 0.0) -> pd.DataFrame:
    """One row per user, month and category, plus a category "ALL" row with the month totals.

    - income / expenses are positive sums of money in / out; savings = income - expenses
    - every user gets every month from their first to their last transaction and every category
      seen in the data; months or categories without transactions are filled with 0
    - "ALL" rows also carry savings_rate and balance_end = opening balance + running savings
    - missing categories are grouped as "Uncategorised"; without user_id everything is one user
    """
    columns = ["date", "amount"]
    data = transactions[columns].copy()
    data["user_id"] = transactions["user_id"] if "user_id" in transactions else "user"
    category = transactions["category"] if "category" in transactions else pd.Series(index=transactions.index, dtype=object)
    data["category"] = category.fillna("").astype(str).str.strip().replace("", UNCATEGORISED)
    data["month"] = pd.to_datetime(data["date"]).dt.to_period("M")
    data["income"] = data["amount"].clip(lower=0)
    data["expenses"] = (-data["amount"]).clip(lower=0)
    if data.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    sums = (data.groupby(["user_id", "month", "category"])
            .agg(income=("income", "sum"), expenses=("expenses", "sum"), n_transactions=("amount", "size"))
            .reset_index())

    # full grid: each user's month range x all categories, so gaps become explicit zeros
    ranges = data.groupby("user_id")["month"].agg(["min", "max"])
    user_months = pd.DataFrame(
        [(user, month) for user, row in ranges.iterrows()
         for month in pd.period_range(row["min"], row["max"], freq="M")],
        columns=["user_id", "month"])
    categories = pd.DataFrame({"category": sorted(data["category"].unique())})
    grid = user_months.merge(categories, how="cross")
    by_category = grid.merge(sums, on=["user_id", "month", "category"], how="left")
    by_category[["income", "expenses"]] = by_category[["income", "expenses"]].fillna(0.0)
    by_category["n_transactions"] = by_category["n_transactions"].fillna(0).astype(int)

    totals = (by_category.groupby(["user_id", "month"], as_index=False)[["income", "expenses", "n_transactions"]]
              .sum().assign(category=TOTAL))

    table = pd.concat([totals, by_category], ignore_index=True)
    table[["income", "expenses"]] = table[["income", "expenses"]].round(2)
    table["savings"] = (table["income"] - table["expenses"]).round(2)

    table = table.sort_values(["user_id", "month"], kind="stable")
    is_total = table["category"] == TOTAL
    totals_income = table.loc[is_total, "income"]
    table["savings_rate"] = float("nan")
    table.loc[is_total, "savings_rate"] = table.loc[is_total, "savings"] / totals_income.where(totals_income > 0)

    if isinstance(opening_balances, dict):
        opening = table["user_id"].map(opening_balances).fillna(0.0)
    else:
        opening = pd.Series(float(opening_balances), index=table.index)
    running = table.loc[is_total].groupby("user_id")["savings"].cumsum()
    table["balance_end"] = float("nan")
    table.loc[is_total, "balance_end"] = (running + opening[is_total]).round(2)

    order = {TOTAL: 0}
    table = table.sort_values(["user_id", "month", "category"],
                              key=lambda col: col.map(order).fillna(1) if col.name == "category" else col,
                              kind="stable").reset_index(drop=True)
    return table[LONG_COLUMNS]
