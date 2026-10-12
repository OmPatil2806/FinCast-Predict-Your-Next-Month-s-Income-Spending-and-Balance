"""Data quality functions: turn parsed transactions into clean rows ready to save.

Input:  DataFrame with date, description, amount (the standard schema from the parsers).
Output: the same columns plus is_income and occurrence, and a report of removed rows.

Identical rows inside one file are kept on purpose (two 3.50 coffees on the same day are
two real purchases). Each copy gets an occurrence number 1, 2, ... so that the duplicate
check when saving (date + description + amount + occurrence) ignores a re-uploaded file
but still stores both coffees.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "description", "amount"]


def clean_transactions(df: pd.DataFrame,
                       today: pd.Timestamp | None = None) -> tuple[pd.DataFrame, dict]:
    """Clean parsed transactions and report how many rows were removed and why.

    Rules:
      - descriptions: trimmed, repeated spaces collapsed
      - removed: missing date, empty description, missing or zero amount, date in the future
      - amounts rounded to cents; expenses stay negative, income positive
      - adds is_income (amount > 0) and occurrence (1 for the first copy of a row, 2 for the next, ...)
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")

    today = (today or pd.Timestamp.today()).normalize()
    out = df[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["description"] = (
        out["description"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce").round(2)

    # each row is removed for the first rule it breaks, so the counts add up
    checks = {
        "missing_date": out["date"].isna(),
        "empty_description": out["description"] == "",
        "missing_amount": out["amount"].isna(),
        "zero_amount": out["amount"] == 0,
        "future_date": out["date"] > today,
    }
    removed = pd.Series(False, index=out.index)
    report = {"input_rows": len(out)}
    for reason, mask in checks.items():
        newly_removed = mask & ~removed
        report[reason] = int(newly_removed.sum())
        removed |= newly_removed
    out = out[~removed]

    out = out.sort_values("date", kind="stable").reset_index(drop=True)
    out["is_income"] = out["amount"] > 0
    out["occurrence"] = out.groupby(["date", "description", "amount"]).cumcount() + 1

    report["removed_rows"] = int(removed.sum())
    report["kept_rows"] = len(out)
    if report["removed_rows"]:
        logger.info("Cleaning removed %d of %d rows: %s", report["removed_rows"], report["input_rows"],
                    {k: v for k, v in report.items() if k in checks and v})
    return out, report
