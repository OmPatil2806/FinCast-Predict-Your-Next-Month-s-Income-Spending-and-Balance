"""Data quality functions: turn parsed transactions into clean rows ready to save,
and check a whole dataset before it is used for training (validate_transactions).

Input:  DataFrame with date, description, amount (the standard schema from the parsers).
Output: the same columns plus is_income and occurrence, and a report of removed rows.

Identical rows inside one file are kept on purpose (two 3.50 coffees on the same day are
two real purchases). Each copy gets an occurrence number 1, 2, ... so that the duplicate
check when saving (date + description + amount + occurrence) ignores a re-uploaded file
but still stores both coffees.
"""


import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

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


# ---------- data checks (step 4.9) ----------

MAX_ABS_AMOUNT = 50_000      # a single private transaction above this is almost certainly an error
MAX_GAP_MONTHS = 2           # a user may skip at most this many months in a row


class DataValidationError(ValueError):
    """Raised with a list of every problem found, so they can all be fixed at once."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Data checks failed:\n- " + "\n- ".join(problems))


def validate_transactions(df: pd.DataFrame, today: pd.Timestamp | None = None,
                          max_abs_amount: float = MAX_ABS_AMOUNT,
                          max_gap_months: int = MAX_GAP_MONTHS) -> dict:
    """Check a transactions table and raise DataValidationError listing every problem.

    Checks: no future dates, no empty descriptions, amounts within +/- max_abs_amount,
    and no user with a gap of more than max_gap_months months without any transaction.
    Without a user_id column the whole table is treated as one user.
    Returns a short report when everything passes.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError([f"missing columns: {', '.join(missing)}"])

    today = (today or pd.Timestamp.today()).normalize()
    dates = pd.to_datetime(df["date"], errors="coerce")
    descriptions = df["description"].fillna("").astype(str).str.strip()
    amounts = pd.to_numeric(df["amount"], errors="coerce")
    problems = []

    def report(mask: pd.Series, message: str) -> None:
        count = int(mask.sum())
        if count:
            examples = ", ".join(str(i) for i in df.index[mask][:3])
            problems.append(f"{count:,} {message} (e.g. rows {examples})")

    report(dates.isna(), "rows with a missing or invalid date")
    report(dates > today, f"rows dated after {today:%Y-%m-%d}")
    report(descriptions == "", "rows with an empty description")
    report(amounts.isna(), "rows with a missing or non-numeric amount")
    report(amounts.abs() > max_abs_amount, f"rows with an amount beyond ±€{max_abs_amount:,.0f}")

    users = df["user_id"] if "user_id" in df.columns else pd.Series("all", index=df.index)
    months = pd.DataFrame({"user": users, "month": dates.dt.to_period("M")}).dropna().drop_duplicates()
    gaps = {}
    for user, group in months.groupby("user"):
        ordered = group["month"].sort_values()
        steps = (ordered.dt.year.diff() * 12 + ordered.dt.month.diff()).dropna() - 1  # months skipped
        if len(steps) and steps.max() > max_gap_months:
            gaps[user] = int(steps.max())
    if gaps:
        examples = ", ".join(f"{u} ({g} months)" for u, g in list(gaps.items())[:3])
        problems.append(f"{len(gaps)} users with a gap longer than {max_gap_months} months: {examples}")

    if problems:
        raise DataValidationError(problems)
    return {
        "rows": len(df),
        "users": int(months["user"].nunique()),
        "first_date": dates.min(),
        "last_date": dates.max(),
    }
