"""Tests for the Anomalies review list, status updates and page (step 8.6)."""

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.auth.auth_service import register
from app.database.crud import add_transactions, get_transactions, set_anomaly_status
from app.database.db import init_db
from src.forecasting.anomalies import history_note, load_detector, review_list
from src.ingestion.synthetic_generator import generate_dataset
from src.utils.helpers import get_project_root


@pytest.fixture(scope="module")
def detector():
    return load_detector()


@pytest.fixture(scope="module")
def mine():
    transactions, users = generate_dataset(n_users=1, months=12, start="2025-01-01", seed=3)
    tx = transactions.drop(columns=["user_id", "archetype"])
    duplicate = tx[(tx["category"] == "Groceries") & (tx["date"] >= "2025-12-01")].iloc[[0]].copy()
    duplicate["date"] = pd.to_datetime(duplicate["date"]) + pd.Timedelta(days=1)
    return pd.concat([tx, duplicate], ignore_index=True).assign(id=lambda d: range(1, len(d) + 1))


def test_review_list_finds_the_duplicate_and_sorts_by_score(mine, detector):
    flags = review_list(mine, detector)
    assert not flags.empty and flags["anomaly_score"].is_monotonic_decreasing
    assert flags["reason"].str.contains("possible duplicate").any()
    recent = review_list(mine, detector, since="2025-12-01")
    assert (recent["date"] >= "2025-12-01").all() and len(recent) <= len(flags)


def test_dismissed_flags_are_hidden_unless_asked(mine, detector):
    flags = review_list(mine, detector)
    first = flags.loc[0, "id"]
    marked = mine.assign(anomaly_status=mine["id"].map({first: "dismissed"}))
    assert first not in review_list(marked, detector)["id"].tolist()
    assert first in review_list(marked, detector, include_dismissed=True)["id"].tolist()


def test_history_note():
    two_months = pd.DataFrame({"date": ["2026-08-03", "2026-09-10"], "amount": [-5.0, -7.0]})
    assert "2 months" in history_note(two_months)
    assert history_note(pd.DataFrame({"date": ["2026-07-01", "2026-08-01", "2026-09-01"], "amount": [-1.0] * 3})) is None
    assert review_list(two_months.head(0), None).empty


def test_status_only_changes_the_users_own_rows(tmp_path):
    db = init_db(tmp_path / "app.db")
    a = register("user_a", "a@example.com", "secret123", db_path=db)
    b = register("user_b", "b@example.com", "secret123", db_path=db)
    add_transactions(a, pd.DataFrame({"date": ["2026-09-01"], "description": ["Shop"], "amount": [-10.0]}), db_path=db)
    tid = int(get_transactions(a, db_path=db)["id"].iloc[0])
    assert not set_anomaly_status(b, tid, "dismissed", db_path=db)                 # someone else's row
    assert set_anomaly_status(a, tid, "confirmed", db_path=db)
    assert get_transactions(a, db_path=db)["anomaly_status"].iloc[0] == "confirmed"
    assert set_anomaly_status(a, tid, None, db_path=db)
    assert get_transactions(a, db_path=db)["anomaly_status"].isna().all()
    with pytest.raises(ValueError):
        set_anomaly_status(a, tid, "ignored", db_path=db)


def test_page_asks_for_login():
    page = str(get_project_root() / "dashboard" / "pages" / "5_Anomalies.py")
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception and [w.value for w in at.warning] == ["Please log in first."]
