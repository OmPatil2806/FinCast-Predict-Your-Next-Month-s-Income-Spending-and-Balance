"""Tests for the live predictor (step 8.1): same path as training, from a DataFrame or the app database."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from app.auth.auth_service import register
from app.database.crud import add_transactions, get_profile
from app.database.db import get_connection, init_db
from src.features.build_features import build_features
from src.forecasting.predictor import complete_history, forecast_from_transactions, forecast_user, load_forecasters
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category

AS_OF = "2025-12-15"          # history: Jan 2024 - Nov 2025 complete, December unfinished


@pytest.fixture(scope="module")
def forecasters():
    return load_forecasters()


@pytest.fixture(scope="module")
def data():
    transactions, users = generate_dataset(n_users=3, months=24, start="2024-01-01", seed=8)
    return transactions, users


def test_same_forecast_as_the_exported_models(data, forecasters):
    transactions, users = data
    user = users["user_id"].iloc[0]
    mine = transactions[transactions["user_id"] == user]
    forecast = forecast_from_transactions(mine, forecasters, as_of=AS_OF, user_id=user)

    complete = complete_history(mine, AS_OF)
    table, _ = build_features(aggregate_user_month_category(complete))
    rows = table[table["month"] == table["month"].max()]
    expected = pd.concat([forecasters[f].predict(rows[(rows["series"] == "income") == (f == "income")])
                          .assign(series=rows["series"]) for f in forecasters])
    merged = forecast.series.merge(expected, on="series", suffixes=("", "_expected"))
    assert len(merged) == len(forecast.series) == 12
    for column in ("lower", "predicted", "upper"):
        np.testing.assert_allclose(merged[column], merged[f"{column}_expected"])
    assert str(forecast.origin_month) == "2025-11" and str(forecast.target_month) == "2025-12"
    assert forecast.series["series"].iloc[:2].tolist() == ["income", "expenses"]
    assert ((forecast.series["lower"] <= forecast.series["predicted"])
            & (forecast.series["predicted"] <= forecast.series["upper"])).all()


def test_unfinished_month_is_ignored(data, forecasters):
    transactions, users = data
    mine = transactions[transactions["user_id"] == users["user_id"].iloc[1]]
    early = forecast_from_transactions(mine, forecasters, as_of="2025-12-01")
    late = forecast_from_transactions(mine, forecasters, as_of="2025-12-31")
    pd.testing.assert_frame_equal(early.series, late.series)          # December rows never count
    with pytest.raises(ValueError):
        forecast_from_transactions(mine, forecasters, as_of="2024-01-20")


def test_balance_adds_up(data, forecasters):
    transactions, users = data
    mine = transactions[transactions["user_id"] == users["user_id"].iloc[2]]
    forecast = forecast_from_transactions(mine, forecasters, opening_balance=1000, as_of=AS_OF)
    complete = complete_history(mine, AS_OF)
    assert forecast.balance["current"] == pytest.approx(1000 + complete["amount"].sum(), abs=0.05)
    income, spend = forecast.get("income"), forecast.get("expenses")
    assert forecast.balance["predicted"] == pytest.approx(forecast.balance["current"] + income["predicted"] - spend["predicted"], abs=0.01)
    assert forecast.balance["lower"] <= forecast.balance["predicted"] <= forecast.balance["upper"]


def test_unknown_category_falls_back_to_its_average(data, forecasters):
    transactions, users = data
    mine = transactions[transactions["user_id"] == users["user_id"].iloc[0]].copy()
    pets = mine.sample(30, random_state=1).index
    mine.loc[pets, "category"] = "Pets"
    forecast = forecast_from_transactions(mine, forecasters, as_of=AS_OF)
    row = forecast.get("expenses:Pets")
    assert row["champion"].startswith("mean_3") and row["lower"] == row["predicted"] == row["upper"]
    assert any("Pets" in note for note in forecast.notes)


def test_forecast_from_the_app_database(data, forecasters, tmp_path):
    transactions, users = data
    db = init_db(tmp_path / "app.db")
    user_id = register("demo_user", "demo@example.com", "secret123", db_path=db)
    conn = get_connection(db)
    with conn:
        conn.execute("UPDATE profiles SET opening_balance = 2500 WHERE user_id = ?", (user_id,))
    conn.close()
    assert get_profile(user_id, db_path=db)["opening_balance"] == 2500

    mine = transactions[transactions["user_id"] == users["user_id"].iloc[0]].copy()
    mine["occurrence"] = mine.groupby(["date", "description", "amount"]).cumcount() + 1
    assert add_transactions(user_id, mine, db_path=db)["added"] == len(mine)

    from_db = forecast_user(user_id, db_path=db, as_of=AS_OF, forecasters=forecasters)
    direct = forecast_from_transactions(mine, forecasters, opening_balance=2500, as_of=AS_OF)
    pd.testing.assert_frame_equal(from_db.series, direct.series)
    assert from_db.balance == direct.balance and from_db.model_version == forecasters["income"].version

    empty_user = register("new_user", "new@example.com", "secret123", db_path=db)
    with pytest.raises(ValueError, match="import"):
        forecast_user(empty_user, db_path=db, forecasters=forecasters)


def test_predictor_loads_without_mlflow():
    code = "import sys, src.forecasting.predictor; print('mlflow' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() == "False"
