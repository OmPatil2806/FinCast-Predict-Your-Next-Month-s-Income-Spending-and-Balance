"""Tests for the Profile page pieces (step 8.8): settings, password, export, account deletion."""

import io
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.auth.auth_service import AuthError, change_password, login, register
from app.database.crud import (
    add_transactions, count_user_rows, create_goal, delete_user, get_profile, save_forecast, set_anomaly_status,
    get_transactions, update_profile,
)
from app.database.db import get_connection, init_db
from app.database.export import user_data_zip
from src.utils.helpers import get_project_root


@pytest.fixture
def db(tmp_path):
    return init_db(tmp_path / "app.db")


def _fill(user_id, db):
    tx = pd.DataFrame({"date": ["2026-08-01", "2026-08-02", "2026-09-01"], "description": ["Salary", "Shop", "Shop"],
                       "amount": [3000.0, -50.0, -60.0], "category": ["Income", "Shopping", "Shopping"]})
    add_transactions(user_id, tx, db_path=db)
    set_anomaly_status(user_id, int(get_transactions(user_id, db_path=db)["id"].iloc[0]), "confirmed", db_path=db)
    series = pd.DataFrame({"series": ["income", "expenses"], "lower": [1.0, 1.0], "predicted": [2.0, 2.0],
                           "upper": [3.0, 3.0]})
    save_forecast(user_id, SimpleNamespace(series=series, target_month=pd.Period("2026-10", "M"), model_version=1,
                                           balance={"predicted": 5.0, "lower": 1.0, "upper": 9.0}), db_path=db)
    create_goal(user_id, "Holiday", 1000, db_path=db)
    conn = get_connection(db)
    with conn:
        conn.execute("INSERT INTO monthly_summary (user_id, month) VALUES (?, '2026-08')", (user_id,))
    conn.close()


def test_update_profile(db):
    user = register("om_patil", "om@example.com", "secret123", db_path=db)
    assert update_profile(user, income_type="freelance", opening_balance=1234.567, db_path=db)
    profile = get_profile(user, db_path=db)
    assert profile["income_type"] == "freelance" and profile["opening_balance"] == 1234.57 and profile["currency"] == "EUR"
    with pytest.raises(ValueError):
        update_profile(user, income_type="lottery", db_path=db)
    assert not update_profile(user, db_path=db)


def test_change_password(db):
    user = register("om_patil", "om@example.com", "secret123", db_path=db)
    for current, new in (("wrong-pass", "newsecret1"), ("secret123", "short"), ("secret123", "secret123")):
        with pytest.raises(AuthError):
            change_password(user, current, new, db_path=db)
    change_password(user, "secret123", "newsecret1", db_path=db)
    assert login("om_patil", "newsecret1", db_path=db) is not None
    assert login("om_patil", "secret123", db_path=db) is None


def test_export_has_every_table_and_no_password(db):
    user = register("om_patil", "om@example.com", "secret123", db_path=db)
    _fill(user, db)
    archive = zipfile.ZipFile(io.BytesIO(user_data_zip(user, db_path=db)))
    assert set(archive.namelist()) == {"README.txt", "account.csv", "profiles.csv", "transactions.csv",
                                       "monthly_summary.csv", "forecasts.csv", "goals.csv"}
    assert len(pd.read_csv(archive.open("transactions.csv"))) == 3
    assert len(pd.read_csv(archive.open("forecasts.csv"))) == 3
    account = archive.read("account.csv").decode()
    assert "om_patil" in account and "password" not in account and "$2b$" not in account


def test_delete_removes_every_row_and_only_that_users(db):
    a = register("user_a", "a@example.com", "secret123", db_path=db)
    b = register("user_b", "b@example.com", "secret123", db_path=db)
    _fill(a, db)
    _fill(b, db)
    before = delete_user(a, db_path=db)
    assert all(before[t] > 0 for t in before)                       # every table had rows for user a
    assert set(count_user_rows(a, db_path=db).values()) == {0}       # and none are left
    assert count_user_rows(b, db_path=db) == before                  # user b untouched
    assert login("user_a", "secret123", db_path=db) is None


def test_page_asks_for_login():
    page = str(get_project_root() / "dashboard" / "pages" / "7_Profile.py")
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception and [w.value for w in at.warning] == ["Please log in first."]
