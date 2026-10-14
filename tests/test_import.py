"""Tests for the import path after parsing: cleaning, saving without duplicates, monthly totals."""

import pandas as pd
import pytest

from app.auth.auth_service import register
from app.database.crud import (
    add_transactions, get_transactions, get_uncategorized_transactions, next_occurrence, update_categories,
)
from app.database.db import init_db
from src.ingestion.bank_csv_parser import parse_bank_csv
from src.preprocessing.aggregate_monthly import aggregate_by_category, aggregate_monthly
from src.preprocessing.cleaning import clean_transactions
from src.utils.helpers import get_project_root

SAMPLE = get_project_root() / "data" / "sample" / "sample_bank_export.csv"
TODAY = pd.Timestamp("2026-10-12")


@pytest.fixture
def db(tmp_path):
    return init_db(tmp_path / "test.db")


@pytest.fixture
def user_id(db):
    return register("om_patil", "om@example.com", "secret123", db_path=db)


@pytest.fixture
def sample():
    clean, _ = clean_transactions(parse_bank_csv(SAMPLE), today=TODAY)
    return clean


def test_cleaning_removes_invalid_rows_and_reports_why():
    raw = pd.DataFrame({
        "date": ["2026-07-01", None, "2026-07-02", "2026-07-03", "2030-01-01", "2026-07-05"],
        "description": ["  Coffee   Shop ", "REWE", "   ", "LIDL", "Future", "Bad"],
        "amount": [-3.5, -10, -5, 0, -20, None],
    })
    clean, report = clean_transactions(raw, today=TODAY)
    assert clean["description"].tolist() == ["Coffee Shop"]
    assert report["removed_rows"] == 5
    for reason in ("missing_date", "empty_description", "zero_amount", "future_date", "missing_amount"):
        assert report[reason] == 1


def test_identical_rows_are_kept_with_occurrence_numbers():
    raw = pd.DataFrame({"date": ["2026-07-01"] * 3, "description": ["Coffee"] * 3, "amount": [-3.5] * 3})
    clean, _ = clean_transactions(raw, today=TODAY)
    assert clean["occurrence"].tolist() == [1, 2, 3]


def test_second_upload_adds_nothing(db, user_id, sample):
    assert add_transactions(user_id, sample, db_path=db) == {"added": 60, "skipped": 0}
    assert add_transactions(user_id, sample, db_path=db) == {"added": 0, "skipped": 60}
    assert len(get_transactions(user_id, db_path=db)) == 60


def test_users_only_see_their_own_transactions(db, user_id, sample):
    other = register("other_user", "other@example.com", "secret123", db_path=db)
    add_transactions(user_id, sample, db_path=db)
    assert get_transactions(other, db_path=db).empty


def test_date_filter(db, user_id, sample):
    add_transactions(user_id, sample, db_path=db)
    august = get_transactions(user_id, "2026-08-01", "2026-08-31", db_path=db)
    assert len(august) == 20
    assert august["amount"].sum() == pytest.approx(1111.31)


def test_manual_duplicate_gets_next_occurrence(db, user_id):
    entry, _ = clean_transactions(
        pd.DataFrame({"date": ["2026-10-01"], "description": ["Coffee"], "amount": [-3.5]}), today=TODAY)
    for expected in (1, 2):
        entry["occurrence"] = next_occurrence(user_id, "2026-10-01", "Coffee", -3.5, db_path=db)
        assert entry["occurrence"].iloc[0] == expected
        assert add_transactions(user_id, entry, source="manual", db_path=db)["added"] == 1


def test_monthly_totals_match_transactions(sample):
    monthly = aggregate_monthly(sample)
    assert monthly["month"].astype(str).tolist() == ["2026-07", "2026-08", "2026-09"]
    assert monthly["income"].tolist() == [3200.0, 3234.99, 3200.0]
    assert monthly["savings"].sum() == pytest.approx(sample["amount"].sum())


def test_missing_months_are_filled_with_zero():
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-15", "2026-04-02"]), "amount": [-100.0, 2000.0]})
    monthly = aggregate_monthly(df)
    assert len(monthly) == 4
    assert monthly.loc[1:2, ["income", "expenses"]].sum().sum() == 0


def test_spending_by_category_groups_missing_as_uncategorised(sample):
    by_category = aggregate_by_category(sample)
    assert by_category.to_dict("records") == [{"category": "Uncategorised", "expenses": 5121.18}]


def test_categorize_older_transactions_only_for_own_user(db, user_id, sample):
    other = register("other_user", "other@example.com", "secret123", db_path=db)
    add_transactions(user_id, sample.head(3), db_path=db)
    add_transactions(other, sample.head(3), db_path=db)
    pending = get_uncategorized_transactions(user_id, db_path=db)
    assert len(pending) == 3

    updates = pending[["id"]].assign(category="Rent", category_source="rule")
    assert update_categories(user_id, updates, db_path=db) == 3
    assert get_uncategorized_transactions(user_id, db_path=db).empty
    # the other user's identical rows are untouched, even if their ids were passed
    other_ids = get_uncategorized_transactions(other, db_path=db)[["id"]]
    assert update_categories(user_id, other_ids.assign(category="Rent", category_source="rule"), db_path=db) == 0
    assert len(get_uncategorized_transactions(other, db_path=db)) == 3
