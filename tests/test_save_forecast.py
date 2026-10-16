"""Saving forecasts to the forecasts table (step 8.4)."""

from types import SimpleNamespace

import pandas as pd
import pytest

from app.auth.auth_service import register
from app.database.crud import get_forecasts, save_forecast
from app.database.db import get_connection, init_db


def _forecast(scale=1.0, month="2026-10"):
    series = pd.DataFrame({"series": ["income", "expenses", "expenses:Rent"],
                           "lower": [2800.0, 1800.0, 890.0], "predicted": [3000.0, 2100.0, 900.0],
                           "upper": [3300.0, 2500.0, 950.0]})
    series[["lower", "predicted", "upper"]] *= scale
    return SimpleNamespace(series=series, target_month=pd.Period(month, "M"), model_version=1,
                           balance={"current": 100.0, "predicted": 1000.0 * scale, "lower": 300.0, "upper": 1600.0})


@pytest.fixture
def db(tmp_path):
    return init_db(tmp_path / "app.db")


def test_one_row_per_metric_and_only_the_first_forecast_is_kept(db):
    user = register("om_patil", "om@example.com", "secret123", db_path=db)
    assert save_forecast(user, _forecast(), db_path=db) == {"saved": 4, "already_saved": 0}
    assert save_forecast(user, _forecast(scale=2.0), db_path=db) == {"saved": 0, "already_saved": 4}
    stored = get_forecasts(user, "2026-10", db_path=db).set_index("metric")
    assert sorted(stored.index) == ["balance", "expenses", "expenses:Rent", "income"]
    assert stored.loc["income", "predicted"] == 3000.0                  # the first one, not the re-run
    assert stored.loc["balance", ["lower", "upper"]].tolist() == [300.0, 1600.0]
    assert stored["model_version"].eq("1").all() and stored["actual"].isna().all()

    save_forecast(user, _forecast(month="2026-11"), db_path=db)          # next month adds new rows
    assert get_forecasts(user, db_path=db)["target_month"].tolist()[0] == "2026-10"
    assert len(get_forecasts(user, db_path=db)) == 8


def test_users_only_see_their_own_forecasts_and_delete_cascades(db):
    a = register("user_a", "a@example.com", "secret123", db_path=db)
    b = register("user_b", "b@example.com", "secret123", db_path=db)
    save_forecast(a, _forecast(), db_path=db)
    assert get_forecasts(b, db_path=db).empty
    conn = get_connection(db)
    with conn:
        conn.execute("DELETE FROM users WHERE id = ?", (a,))
    conn.close()
    assert get_forecasts(a, db_path=db).empty
