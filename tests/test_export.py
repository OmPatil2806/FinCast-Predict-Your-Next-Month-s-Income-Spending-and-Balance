"""Tests for the champion export and registry (step 7.9), with tiny models and a throw-away MLflow store."""

import json
import sys
import subprocess

import joblib
import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.features.build_features import build_features
from src.forecasting.model import ChampionForecaster
from src.ingestion.synthetic_generator import generate_dataset
from src.preprocessing.aggregate_monthly import aggregate_user_month_category
from src.training.export_champion import export

FAST = {"n_estimators": 20, "num_leaves": 8, "min_child_samples": 5}


@pytest.fixture(scope="module")
def data():
    transactions, _ = generate_dataset(n_users=5, months=18, start="2024-01-01", seed=21)
    table, info = build_features(aggregate_user_month_category(transactions))
    spend_series = sorted(set(table["series"]) - {"income"})
    champions = {"rule": "test", "families": {
        "income": {"mode": "relative", "width_factor": 1.0, "test_months": "x", "system": {"mae": 1.0},
                   "series": {"income": {"champion": "lightgbm"}}},
        "spend": {"mode": "raw", "width_factor": 1.2, "test_months": "x", "system": {"mae": 2.0},
                  "series": {s: {"champion": "lightgbm" if i % 2 else "last_month"} for i, s in enumerate(spend_series)}},
    }}
    return table, info["feature_columns"], champions


def test_export_without_registry(data, tmp_path):
    table, features, champions = data
    meta = export(table, features, champions, {"income": FAST, "spend": FAST}, out_dir=tmp_path, register_models=False)
    assert meta["version"] == 1 and meta["feature_columns"] == features
    saved = json.loads((tmp_path / "model_meta.json").read_text(encoding="utf-8"))
    assert set(saved["families"]) == {"income", "spend"}

    model = joblib.load(tmp_path / "forecaster_spend.joblib")
    assert isinstance(model, ChampionForecaster) and model.width_factor == 1.2
    rows = table[(table["month"] == table["month"].max()) & (table["series"] != "income")]
    out = model.predict(rows)
    assert list(out.columns) == ["champion", "lower", "predicted", "upper"] and out.index.equals(rows.index)
    assert ((out["lower"] <= out["predicted"]) & (out["predicted"] <= out["upper"])).all()
    rent = rows["series"] == "expenses:Rent"
    if out.loc[rent, "champion"].eq("last_month").all():
        assert (out.loc[rent, "predicted"] == rows.loc[rent, "lag_1"].clip(lower=0)).all()

    assert export(table, features, champions, {"income": FAST, "spend": FAST}, out_dir=tmp_path,
                  register_models=False)["version"] == 2          # every export bumps the version


def test_export_registers_and_sets_the_champion_alias(data, tmp_path):
    table, features, champions = data
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    meta = export(table, features, champions, {"income": FAST, "spend": FAST}, out_dir=tmp_path / "models",
                  tracking_uri=uri, artifact_root=tmp_path / "mlruns")
    registry = meta["families"]["income"]["registry"]
    mlflow.set_tracking_uri(uri)
    assert MlflowClient().get_model_version_by_alias(registry["name"], "champion").version == registry["version"]
    rows = table[(table["month"] == table["month"].max()) & (table["series"] == "income")]
    served = mlflow.pyfunc.load_model(f"models:/{registry['name']}@champion").predict(rows)
    pd.testing.assert_frame_equal(served, joblib.load(tmp_path / "models" / "forecaster_income.joblib").predict(rows))


def test_the_app_can_load_the_model_without_mlflow():
    code = "import sys, src.forecasting.model; print('mlflow' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() == "False"
