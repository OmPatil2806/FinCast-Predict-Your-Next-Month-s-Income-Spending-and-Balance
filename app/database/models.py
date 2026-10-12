"""SQLite table definitions for the FinCast app database.

Every table except users carries a user_id that points to users.id with
ON DELETE CASCADE, so deleting a user removes all of their data.
Run by init_db() in app/database/db.py; IF NOT EXISTS makes it safe to run on every start.
"""

CREATE_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,  -- "Om" and "om" are the same user
    email         TEXT    NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT    NOT NULL,              -- bcrypt hash, never the plain password
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_PROFILES = """
CREATE TABLE IF NOT EXISTS profiles (
    user_id         INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    income_type     TEXT    NOT NULL DEFAULT 'salary'
                    CHECK (income_type IN ('salary', 'freelance')),
    currency        TEXT    NOT NULL DEFAULT 'EUR',
    opening_balance REAL    NOT NULL DEFAULT 0,  -- needed for the end-of-month balance forecast
    savings_goal    REAL,
    goal_date       TEXT
);
"""

CREATE_TRANSACTIONS = """
CREATE TABLE IF NOT EXISTS transactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date            TEXT    NOT NULL,            -- ISO format YYYY-MM-DD
    description     TEXT    NOT NULL,
    amount          REAL    NOT NULL,            -- expenses negative, income positive
    category        TEXT,
    category_source TEXT    CHECK (category_source IN ('rule', 'model', 'user')),
    is_income       INTEGER NOT NULL DEFAULT 0 CHECK (is_income IN (0, 1)),
    source          TEXT    NOT NULL DEFAULT 'csv' CHECK (source IN ('csv', 'manual', 'demo')),
    dedupe_hash     TEXT    NOT NULL,            -- date + description + amount + occurrence number
    anomaly_status  TEXT    CHECK (anomaly_status IN ('flagged', 'confirmed', 'dismissed')),
    created_at      TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, dedupe_hash)                -- uploading the same file twice adds nothing
);
"""

CREATE_MONTHLY_SUMMARY = """
CREATE TABLE IF NOT EXISTS monthly_summary (
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    month       TEXT    NOT NULL,                -- YYYY-MM
    category    TEXT    NOT NULL DEFAULT 'ALL',  -- 'ALL' row holds the month totals
    income      REAL    NOT NULL DEFAULT 0,
    expenses    REAL    NOT NULL DEFAULT 0,
    savings     REAL    NOT NULL DEFAULT 0,
    balance_end REAL,
    PRIMARY KEY (user_id, month, category)
);
"""

CREATE_FORECASTS = """
CREATE TABLE IF NOT EXISTS forecasts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    target_month  TEXT    NOT NULL,              -- YYYY-MM being predicted
    metric        TEXT    NOT NULL,              -- e.g. income, expenses, balance, expenses:Groceries
    predicted     REAL    NOT NULL,
    lower         REAL,
    upper         REAL,
    model_version TEXT,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    actual        REAL,                          -- filled after the month ends
    UNIQUE (user_id, target_month, metric)       -- one forecast per user, month and metric
);
"""

CREATE_GOALS = """
CREATE TABLE IF NOT EXISTS goals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name          TEXT    NOT NULL,
    target_amount REAL    NOT NULL CHECK (target_amount > 0),
    target_date   TEXT,
    created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions (user_id, date);
CREATE INDEX IF NOT EXISTS idx_forecasts_user_month   ON forecasts (user_id, target_month);
CREATE INDEX IF NOT EXISTS idx_goals_user             ON goals (user_id);
"""

# Order matters: users first, because every other table points to it.
ALL_TABLES = [
    CREATE_USERS,
    CREATE_PROFILES,
    CREATE_TRANSACTIONS,
    CREATE_MONTHLY_SUMMARY,
    CREATE_FORECASTS,
    CREATE_GOALS,
    CREATE_INDEXES,
]
