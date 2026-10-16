"""All database queries for the FinCast app.

Every function opens its own connection and closes it when done.
Every query on user data filters by user_id, so users only ever see their own rows.
The optional db_path argument lets tests use a temporary database.
"""

import hashlib
import sqlite3
from pathlib import Path

import pandas as pd

from app.database.db import get_connection


def _row_to_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


# ---------- users ----------

def create_user(username: str, email: str, password_hash: str,
                db_path: str | Path | None = None) -> int:
    """Insert a user and an empty profile with default settings; return the new user id.

    Raises sqlite3.IntegrityError if the username or email is already taken.
    """
    conn = get_connection(db_path)
    try:
        with conn:  # one transaction: both rows are saved, or neither
            cur = conn.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                (username.strip(), email.strip().lower(), password_hash),
            )
            user_id = cur.lastrowid
            conn.execute("INSERT INTO profiles (user_id) VALUES (?)", (user_id,))
        return user_id
    finally:
        conn.close()


def get_user_by_username(username: str, db_path: str | Path | None = None) -> dict | None:
    """Return the user as a dict, or None if no such username (case-insensitive)."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username.strip(),)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_profile(user_id: int, db_path: str | Path | None = None) -> dict | None:
    """Return the user's profile (income_type, currency, opening_balance, ...) or None."""
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


def get_user_by_email(email: str, db_path: str | Path | None = None) -> dict | None:
    """Return the user as a dict, or None if no such email (case-insensitive)."""
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email.strip().lower(),)
        ).fetchone()
        return _row_to_dict(row)
    finally:
        conn.close()


# ---------- transactions ----------

TRANSACTION_COLUMNS = [
    "id", "date", "description", "amount", "category", "category_source",
    "is_income", "source", "anomaly_status", "created_at",
]


def make_dedupe_hash(date, description: str, amount: float, occurrence: int = 1) -> str:
    """Fingerprint of one transaction. The occurrence number keeps identical rows apart
    (two 3.50 coffees on one day), while re-uploading the same file gives the same hashes."""
    key = f"{pd.Timestamp(date):%Y-%m-%d}|{description}|{float(amount):.2f}|{int(occurrence)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def add_transactions(user_id: int, df: pd.DataFrame, source: str = "csv",
                     db_path: str | Path | None = None) -> dict:
    """Save cleaned transactions for a user and skip any that are already stored.

    df needs date, description and amount (the output of clean_transactions);
    is_income, occurrence and category are used when present.
    Returns {"added": n, "skipped": m}.
    """
    if df.empty:
        return {"added": 0, "skipped": 0}
    rows = []
    for rec in df.to_dict("records"):
        amount = round(float(rec["amount"]), 2)
        rows.append((
            user_id,
            f"{pd.Timestamp(rec['date']):%Y-%m-%d}",
            rec["description"],
            amount,
            rec.get("category"),
            rec.get("category_source"),
            int(rec.get("is_income", amount > 0)),
            source,
            make_dedupe_hash(rec["date"], rec["description"], amount, rec.get("occurrence", 1)),
        ))
    conn = get_connection(db_path)
    try:
        with conn:
            before = conn.total_changes
            conn.executemany(
                """INSERT OR IGNORE INTO transactions
                   (user_id, date, description, amount, category, category_source,
                    is_income, source, dedupe_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            added = conn.total_changes - before
        return {"added": added, "skipped": len(rows) - added}
    finally:
        conn.close()


def next_occurrence(user_id: int, date, description: str, amount: float,
                    db_path: str | Path | None = None) -> int:
    """Occurrence number for a new manual entry: 1 + how many identical rows the user already has.

    Lets a user add the same coffee twice on one day without it being treated as a duplicate.
    """
    conn = get_connection(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE user_id = ? AND date = ? AND description = ? AND amount = ?",
            (user_id, f"{pd.Timestamp(date):%Y-%m-%d}", description, round(float(amount), 2)),
        ).fetchone()[0]
        return count + 1
    finally:
        conn.close()


def get_transactions(user_id: int, start=None, end=None,
                     db_path: str | Path | None = None) -> pd.DataFrame:
    """Return one user's transactions, newest first, optionally between start and end (inclusive)."""
    query = f"SELECT {', '.join(TRANSACTION_COLUMNS)} FROM transactions WHERE user_id = ?"
    params: list = [user_id]
    if start is not None:
        query += " AND date >= ?"
        params.append(f"{pd.Timestamp(start):%Y-%m-%d}")
    if end is not None:
        query += " AND date <= ?"
        params.append(f"{pd.Timestamp(end):%Y-%m-%d}")
    query += " ORDER BY date DESC, id DESC"

    conn = get_connection(db_path)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df["is_income"] = df["is_income"].astype(bool)
    return df


def get_uncategorized_transactions(user_id: int, db_path: str | Path | None = None) -> pd.DataFrame:
    """The user's transactions that have no category yet (id, description, amount)."""
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(
            "SELECT id, description, amount FROM transactions WHERE user_id = ? AND category IS NULL",
            conn, params=[user_id],
        )
    finally:
        conn.close()


def update_categories(user_id: int, updates: pd.DataFrame, db_path: str | Path | None = None) -> int:
    """Set category and category_source for existing rows (columns: id, category, category_source).

    Only rows that belong to user_id are changed. Returns the number of rows updated.
    """
    if updates.empty:
        return 0
    rows = [(r["category"], r["category_source"], int(r["id"]), user_id)
            for r in updates.to_dict("records")]
    conn = get_connection(db_path)
    try:
        with conn:
            before = conn.total_changes
            conn.executemany(
                "UPDATE transactions SET category = ?, category_source = ? WHERE id = ? AND user_id = ?", rows)
            return conn.total_changes - before
    finally:
        conn.close()


# ---------- forecasts (step 8.4) ----------

def save_forecast(user_id: int, forecast, db_path: str | Path | None = None) -> dict:
    """Store a predictor Forecast: one row per metric (income, expenses, expenses:<Category>, balance).

    Only the FIRST forecast per user, target month and metric is kept (INSERT OR IGNORE on the
    UNIQUE key): that is the honest one, made before the month was known, and it is what Phase 9
    compares with the actual values. Returns {"saved": n, "already_saved": m}.
    """
    target = str(forecast.target_month)
    version = None if forecast.model_version is None else str(forecast.model_version)
    rows = [(user_id, target, r["series"], round(float(r["predicted"]), 2), round(float(r["lower"]), 2),
             round(float(r["upper"]), 2), version) for r in forecast.series.to_dict("records")]
    b = forecast.balance
    rows.append((user_id, target, "balance", round(float(b["predicted"]), 2), round(float(b["lower"]), 2),
                 round(float(b["upper"]), 2), version))
    conn = get_connection(db_path)
    try:
        with conn:
            before = conn.total_changes
            conn.executemany(
                """INSERT OR IGNORE INTO forecasts (user_id, target_month, metric, predicted, lower, upper, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
            saved = conn.total_changes - before
    finally:
        conn.close()
    return {"saved": saved, "already_saved": len(rows) - saved}


def get_forecasts(user_id: int, target_month=None, db_path: str | Path | None = None) -> pd.DataFrame:
    """The user's stored forecasts (optionally one target month 'YYYY-MM'), oldest month first."""
    query = ("SELECT target_month, metric, predicted, lower, upper, model_version, created_at, actual "
             "FROM forecasts WHERE user_id = ?")
    params: list = [user_id]
    if target_month is not None:
        query += " AND target_month = ?"
        params.append(str(target_month))
    query += " ORDER BY target_month, metric"
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
