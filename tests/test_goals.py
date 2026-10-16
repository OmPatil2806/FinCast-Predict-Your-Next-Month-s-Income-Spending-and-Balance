"""Tests for savings goals: projection maths, storage, migration and the page (step 8.7)."""

import sqlite3

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.auth.auth_service import register
from app.components.goal_view import goal_summary, status_label
from app.database.crud import create_goal, delete_goal, get_goals, update_goal
from app.database.db import init_db
from src.forecasting.goals import project_goal, savings_pace
from src.utils.helpers import get_project_root

TODAY = "2026-10-16"


def test_savings_pace_averages_history_and_forecast():
    monthly = pd.DataFrame({"savings": [100.0] * 4 + [400.0] * 6})
    assert savings_pace(monthly) == 400.0                                    # last 6 months only
    assert savings_pace(monthly, forecast_savings=1100.0) == 500.0           # (6 * 400 + 1100) / 7
    assert savings_pace(pd.DataFrame(), forecast_savings=None) == 0.0


@pytest.mark.parametrize("target, saved, pace, date, status", [
    (5000, 1000, 500, "2027-06-30", "on_track"),
    (5000, 1000, 500, "2027-01-31", "behind"),
    (5000, 6000, 500, "2027-01-31", "reached"),
    (5000, 0, 0, "2027-01-01", "no_progress"),
    (5000, 0, -200, None, "no_progress"),
    (5000, 100, 300, "2026-05-01", "overdue"),
    (5000, 1000, 400, None, "no_deadline"),
])
def test_status(target, saved, pace, date, status):
    assert project_goal(target, saved, pace, date, today=TODAY).status == status


def test_projection_numbers():
    p = project_goal(5000, 1000, 500, "2027-06-30", today=TODAY)
    assert p.remaining == 4000 and p.months_needed == 8.0 and p.progress == 0.2
    assert p.projected_date == pd.Timestamp("2027-06-30")                     # 8 months on, end of month
    assert p.required_per_month == pytest.approx(4000 / ((pd.Timestamp("2027-06-30") - pd.Timestamp(TODAY)).days / 30.44), abs=0.01)
    due_now = project_goal(1000, 0, 100, "2026-10-31", today=TODAY)           # less than a month left
    assert due_now.required_per_month == 1000.0
    assert project_goal(1000, 5000, 100, today=TODAY).progress == 1.0


def test_summary_text():
    goal = {"name": "Car", "target_amount": 15000.0, "saved_amount": 2000.0, "target_date": "2027-06-30"}
    lines = goal_summary(goal, project_goal(15000, 2000, 1000, "2027-06-30", today=TODAY))
    assert lines[0] == "€2,000 of €15,000 saved (13%)"
    assert "30 June 2027" in lines[-1] and "more than now" in lines[-1]
    no_date = goal | {"target_date": None}
    assert len(goal_summary(no_date, project_goal(15000, 2000, 1000, None, today=TODAY))) == 2
    assert status_label("behind") == "⚠️ Behind"


@pytest.fixture
def db(tmp_path):
    return init_db(tmp_path / "app.db")


def test_goal_storage_is_per_user_and_validated(db):
    a = register("user_a", "a@example.com", "secret123", db_path=db)
    b = register("user_b", "b@example.com", "secret123", db_path=db)
    first = create_goal(a, "  Holiday ", 2400, 300, "2027-03-31", db_path=db)
    create_goal(a, "Rainy day", 1000, db_path=db)
    goals = get_goals(a, db_path=db)
    assert goals["name"].tolist() == ["Holiday", "Rainy day"]                 # dated goals first
    assert goals["target_date"].tolist() == ["2027-03-31", None]
    assert get_goals(b, db_path=db).empty

    assert not update_goal(b, first, db_path=db, saved_amount=999)            # not b's goal
    assert update_goal(a, first, db_path=db, saved_amount=800, target_date=None)
    assert get_goals(a, db_path=db).set_index("id").loc[first, ["saved_amount", "target_date"]].tolist() == [800.0, None]
    for bad in ({"target_amount": 0}, {"saved_amount": -5}, {"name": " "}, {"colour": "red"}):
        with pytest.raises(ValueError):
            update_goal(a, first, db_path=db, **bad)
    with pytest.raises(ValueError):
        create_goal(a, "", 100, db_path=db)
    assert not delete_goal(b, first, db_path=db) and delete_goal(a, first, db_path=db)
    assert len(get_goals(a, db_path=db)) == 1


def test_old_database_gets_the_new_column(tmp_path):
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE goals (id INTEGER PRIMARY KEY, user_id INTEGER, name TEXT, "
                 "target_amount REAL, target_date TEXT, created_at TEXT)")
    conn.close()
    init_db(path)
    init_db(path)                                                              # safe to run twice
    conn = sqlite3.connect(path)
    columns = [row[1] for row in conn.execute("PRAGMA table_info(goals)")]
    conn.close()
    assert "saved_amount" in columns


def test_page_asks_for_login():
    page = str(get_project_root() / "dashboard" / "pages" / "6_Goals.py")
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception and [w.value for w in at.warning] == ["Please log in first."]
