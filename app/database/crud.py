"""All database queries for the FinCast app.

Every function opens its own connection and closes it when done.
Every query on user data filters by user_id, so users only ever see their own rows.
The optional db_path argument lets tests use a temporary database.
"""

import sqlite3
from pathlib import Path

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
