"""Step 9.1: compare stored forecasts with what really happened.

update_actuals(user_id)   for every stored forecast (step 8.4) whose month is over AND covered by the
                          user's transactions, fill forecasts.actual with the real value, computed with
                          the same monthly aggregation as training:
                              income, expenses, expenses:<Category>   money in / out that month
                              balance                                 balance at the end of that month
                          A month counts as covered when the user has transactions in it or in a later
                          month (a month without any transaction then really was €0). Months that are
                          still running, or not imported yet, stay open and are tried again next time.
accuracy(user_id)         the scored forecasts with error, % error and whether the real value fell
                          inside the 10-90 % range.

The Forecast page runs update_actuals every time it opens, so new months are scored automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import EXPENSES, INCOME
from src.preprocessing.aggregate_monthly import TOTAL, aggregate_user_month_category

BALANCE = "balance"


def actual_values(transactions: pd.DataFrame, opening_balance: float = 0.0) -> pd.DataFrame:
    """Long table month, metric, actual for every month the transactions cover."""
    if transactions.empty:
        return pd.DataFrame(columns=["month", "metric", "actual"])
    monthly = aggregate_user_month_category(transactions.assign(user_id="user"), opening_balances=float(opening_balance))
    totals = monthly[monthly["category"] == TOTAL]
    categories = monthly[(monthly["category"] != TOTAL) & (monthly["category"] != "Income")]
    parts = [
        totals.assign(metric=INCOME, actual=totals["income"]),
        totals.assign(metric=EXPENSES, actual=totals["expenses"]),
        totals.assign(metric=BALANCE, actual=totals["balance_end"]),
        categories.assign(metric=EXPENSES + ":" + categories["category"], actual=categories["expenses"]),
    ]
    out = pd.concat(parts, ignore_index=True)[["month", "metric", "actual"]]
    out["month"] = out["month"].astype(str)
    return out


def due_forecasts(forecasts: pd.DataFrame, transactions: pd.DataFrame, today=None) -> pd.DataFrame:
    """Open forecasts whose month is over and covered by the data."""
    if forecasts.empty or transactions.empty:
        return forecasts.iloc[0:0]
    current = pd.Timestamp(today if today is not None else pd.Timestamp.today()).to_period("M")
    last_covered = pd.to_datetime(transactions["date"]).max().to_period("M")
    months = pd.PeriodIndex(forecasts["target_month"], freq="M")
    return forecasts[forecasts["actual"].isna() & (months < current) & (months <= last_covered)]


def update_actuals(user_id: int, db_path: str | Path | None = None, today=None) -> int:
    """Fill in the real values of finished months; returns the number of forecast values scored."""
    from app.database.crud import get_forecasts, get_profile, get_transactions, set_forecast_actuals
    due = due_forecasts(get_forecasts(user_id, db_path=db_path), get_transactions(user_id, db_path=db_path), today)
    if due.empty:
        return 0
    profile = get_profile(user_id, db_path=db_path) or {}
    actual = actual_values(get_transactions(user_id, db_path=db_path), profile.get("opening_balance") or 0.0)
    merged = due.merge(actual, left_on=["target_month", "metric"], right_on=["month", "metric"],
                       how="left", suffixes=("_stored", ""))
    # a category the user had no transaction in at all that month: it really was 0
    merged["actual"] = merged["actual"].where(merged["actual"].notna() | ~merged["metric"].str.startswith(EXPENSES + ":"), 0.0)
    merged = merged.dropna(subset=["actual"])
    return set_forecast_actuals(user_id, list(merged[["target_month", "metric", "actual"]].itertuples(index=False)),
                                db_path=db_path)


def score(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Scored rows with error (actual - forecast), abs % error and whether the range held the real value."""
    done = forecasts.dropna(subset=["actual"]).copy()
    done["error"] = done["actual"] - done["predicted"]
    done["abs_pct_error"] = (done["error"].abs() / done["actual"].abs().where(done["actual"].abs() > 0)).astype(float)
    done["inside"] = (done["actual"] >= done["lower"]) & (done["actual"] <= done["upper"])
    return done.reset_index(drop=True)


def accuracy(user_id: int, db_path: str | Path | None = None) -> pd.DataFrame:
    from app.database.crud import get_forecasts
    return score(get_forecasts(user_id, db_path=db_path))


def summary(scored: pd.DataFrame, metrics=(INCOME, EXPENSES, BALANCE)) -> pd.DataFrame:
    """Per metric: months checked, months inside the range, median absolute error in euro and %."""
    rows = []
    for metric in metrics:
        part = scored[scored["metric"] == metric]
        if part.empty:
            continue
        rows.append({"metric": metric, "months": len(part), "inside": int(part["inside"].sum()),
                     "median_abs_error": float(part["error"].abs().median()),
                     "median_abs_pct_error": float(np.nanmedian(part["abs_pct_error"]))
                     if part["abs_pct_error"].notna().any() else np.nan})
    return pd.DataFrame(rows)
