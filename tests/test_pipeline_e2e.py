"""End to end (step 9.4): ingestion + training on a tiny synthetic dataset in a temporary folder.

Runs every default pipeline step (data, baselines, champion rule, export + registry, short-history
calibration, drift reference, SHAP, anomaly model) through FINCAST_CONFIG, and checks the outputs exist
and that nothing in the real project folders was touched.
"""

import copy
import hashlib
import json
import shutil

import joblib
import pytest
import yaml

from src.utils.helpers import CONFIG_ENV, get_project_root, load_categories, load_config

ROOT = get_project_root()
WATCHED = ["models/champions.json", "models/model_meta.json", "models/anomaly_meta.json",
           "models/drift_reference.json", "models/cold_start.json", "reports/shap_importance_spend.csv"]


def _fingerprint():
    return {p: hashlib.md5((ROOT / p).read_bytes()).hexdigest() for p in WATCHED if (ROOT / p).exists()}


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    config = copy.deepcopy(load_config())
    for key in ("data_raw", "data_interim", "data_processed", "models", "logs", "reports"):
        config["paths"][key] = str(tmp_path / key)
    config["synthetic"].update(n_users=8, months=26, start="2024-01-01")
    config["mlflow"]["tracking_uri"] = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    config["mlflow"]["artifact_root"] = str(tmp_path / "mlruns")
    config["forecast"]["cold_start_origins"] = ["2025-06", "2025-09"]
    config["anomaly"].update(planted_duplicates=20, planted_price_rises=20)
    models = tmp_path / "models"
    models.mkdir()
    shutil.copy(ROOT / "models" / "categorizer.joblib", models)               # the app's categoriser
    small = {"n_estimators": 30, "learning_rate": 0.1, "num_leaves": 8, "min_child_samples": 5,
             "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}
    (models / "forecast_params.json").write_text(json.dumps(
        {"families": {f: {"params": small} for f in ("income", "spend")}}))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setenv(CONFIG_ENV, str(path))
    load_config.cache_clear()
    load_categories.cache_clear()
    yield tmp_path
    monkeypatch.delenv(CONFIG_ENV)
    load_config.cache_clear()
    load_categories.cache_clear()


def test_full_pipeline_on_tiny_data(tiny_config, monkeypatch):
    from pipelines import run_ingestion, run_training
    before = _fingerprint()
    monkeypatch.setattr(run_ingestion, "run_macro", lambda: (None, "skipped in tests (no network)"))

    assert run_ingestion.main() == 0
    assert (tiny_config / "data_raw" / "synthetic_transactions.csv").exists()
    assert run_training.main(["--train-categorizer"]) == 0

    models = tiny_config / "models"
    for name in ("categorizer.joblib", "categorizer_meta.json", "baseline_scores.json", "champions.json", "forecaster_income.joblib", "forecaster_spend.joblib",
                 "model_meta.json", "cold_start.json", "drift_reference.json", "anomaly.joblib", "anomaly_meta.json"):
        assert (models / name).exists(), name
    assert (tiny_config / "reports" / "figures" / "shap_summary_spend.png").exists()
    meta = json.loads((models / "model_meta.json").read_text())
    assert meta["training_data"]["users"] == 8 and meta["families"]["spend"]["registry"]["alias"] == "champion"
    assert joblib.load(models / "forecaster_spend.joblib").width_factor > 0
    assert _fingerprint() == before                                    # the real project files are untouched
