"""SQLite connection and database setup for the FinCast app."""

import sqlite3
from pathlib import Path

from app.database.models import ADDED_COLUMNS, ALL_TABLES
from src.utils.helpers import load_config, resolve_path


def get_db_path() -> Path:
    """Full path of the app database, taken from config.yaml (database.path)."""
    return resolve_path(load_config()["database"]["path"])


def get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows readable by column name.

    Open a new connection for each unit of work and close it afterwards;
    Streamlit runs pages in different threads, so connections are not shared.
    """
    path = Path(db_path) if db_path else get_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row             # row["username"] instead of row[1]
    conn.execute("PRAGMA foreign_keys = ON")   # needed for ON DELETE CASCADE
    return conn


def init_db(db_path: str | Path | None = None) -> Path:
    """Create the database file and all tables if they do not exist yet. Safe to call on every start."""
    path = Path(db_path) if db_path else get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(path)
    try:
        for statement in ALL_TABLES:
            conn.executescript(statement)
        for table, column, definition in ADDED_COLUMNS:
            existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        conn.commit()
    finally:
        conn.close()
    return path
