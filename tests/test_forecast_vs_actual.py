"""Forecast vs actual (step 9.1): finished months get their real values, open months stay open."""

import pandas as pd
import pytest

from app.auth.auth_service import register
from app.database.crud import add_transactions, get_forecasts, save_forecast, update_profile
from app.database.db import init_db
from src.forecasting.predictor import forecast_from_transactions, load_forecasters
from src.ingestion.synthetic_generator import generate_dataset
from src.monitoring.forecast_vs_actual import (
    actual_values, due_forecasts, score, summary, update_actuals,
)


@pytest.fixture(scope="module")
def forecasters():
    return load_forecasters()


@pytest.fixture
def user_with_history(tmp_path, forecasters):
    """A synthetic user with forecasts made (without peeking) for July, August, September and October 2025."""
    db = init_db(tmp_path / "app.db")
    user = register("om_patil", "om@example.com", "secret123", db_path=db)
    update_profile(user, opening_balance=1000, db_path=db)
    transactions, _ = generate_dataset(n_users=1, months=21, start="2024-01-01", seed=4)   # Jan 2024 - Sep 2025
    mine = transactions.drop(columns=["user_id", "archetype"])
    mine["occurrence"] = mine.groupby(["date", "description", "amount"]).cumcount() + 1
    add_transactions(user, mine, db_path=db)
    for month in ("2025-07", "2025-08", "2025-09", "2025-10"):
        forecast = forecast_from_transactions(mine, forecasters, opening_balance=1000, as_of=f"{month}-05")
        save_forecast(user, forecast, db_path=db)
    return db, user, mine


def test_actual_values_match_the_transactions():
    tx = pd.DataFrame({"date": ["2025-01-03", "2025-01-10", "2025-02-01"], "amount": [2000.0, -300.0, -50.0],
                       "category": ["Income", "Rent", "Dining"]})
    a = actual_values(tx, opening_balance=100).set_index(["month", "metric"])["actual"]
    assert a[("2025-01", "income")] == 2000 and a[("2025-01", "expenses")] == 300
    assert a[("2025-01", "expenses:Rent")] == 300 and a[("2025-02", "expenses:Rent")] == 0
    assert a[("2025-01", "balance")] == 1800 and a[("2025-02", "balance")] == 1750


def test_only_finished_and_covered_months_are_scored(user_with_history):
    db, user, mine = user_with_history
    scored = update_actuals(user, db_path=db, today="2025-10-20")
    stored = get_forecasts(user, db_path=db)
    done = stored.dropna(subset=["actual"])
    assert sorted(done["target_month"].unique()) == ["2025-07", "2025-08", "2025-09"]
    assert stored.loc[stored["target_month"] == "2025-10", "actual"].isna().all()       # October is still running
    assert scored == len(done) and update_actuals(user, db_path=db, today="2025-10-20") == 0

    real = actual_values(mine, 1000).set_index(["month", "metric"])["actual"]
    for row in done.itertuples():
        assert row.actual == pytest.approx(real.get((row.target_month, row.metric), 0.0), abs=0.01)


def test_month_not_imported_yet_stays_open(user_with_history):
    db, user, _ = user_with_history
    assert update_actuals(user, db_path=db, today="2025-09-15") > 0                      # July and August
    assert get_forecasts(user, "2025-09", db_path=db)["actual"].isna().all()             # September not over


def test_score_and_summary():
    f = pd.DataFrame({"target_month": ["2025-07", "2025-08", "2025-08"], "metric": ["income", "income", "expenses"],
                      "predicted": [100.0, 100.0, 50.0], "lower": [90.0, 90.0, 40.0], "upper": [110.0, 105.0, 60.0],
                      "actual": [95.0, 120.0, None]})
    scored = score(f)
    assert len(scored) == 2 and scored["inside"].tolist() == [True, False]
    assert scored["error"].tolist() == [-5.0, 20.0]
    s = summary(scored).set_index("metric")
    assert s.loc["income", "months"] == 2 and s.loc["income", "inside"] == 1
    assert due_forecasts(f.iloc[0:0], pd.DataFrame({"date": []})).empty


def test_accuracy_panel_pieces():
    from app.components.charts import accuracy_chart
    from app.components.forecast_view import accuracy_headline, accuracy_table
    f = pd.DataFrame({"target_month": ["2025-07", "2025-08"], "metric": ["expenses", "expenses"],
                      "predicted": [2000.0, 2100.0], "lower": [1800.0, 1900.0], "upper": [2300.0, 2400.0],
                      "actual": [2200.0, 2600.0]})
    scored = score(f)
    table = accuracy_table(scored, "expenses")
    assert table["month"].tolist() == ["August 2025", "July 2025"]                # newest first
    assert table["in_range"].tolist() == ["❌ no", "✅ yes"] and table["error"].tolist() == [500.0, 200.0]
    value, detail = accuracy_headline(summary(scored).set_index("metric").loc["expenses"])
    assert value == "1 of 2" and detail.startswith("months in range · typical miss €350")
    fig = accuracy_chart(scored, "expenses", "Spending", "#eb6834")
    assert [t.name for t in fig.data] == ["Forecast range (10–90 %)", "Forecast", "What happened"]
    assert list(fig.data[0].base) == [1800.0, 1900.0] and accuracy_chart(scored, "income", "Income", "#2a78d6") is None
