"""MLflow setup helper, tested against a throw-away tracking store."""

import mlflow

from src.utils.helpers import load_config
from src.utils.mlflow_utils import create_all_experiments, setup_experiment


def test_create_all_experiments(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    created = create_all_experiments(tracking_uri=uri, artifact_root=tmp_path / "mlruns")
    assert set(created) == set(load_config()["mlflow"]["experiments"].values())
    assert {"forecast_income", "forecast_spend", "categorizer", "anomaly", "baselines"} <= set(created)
    # calling again finds the same experiments instead of creating new ones
    assert create_all_experiments(tracking_uri=uri, artifact_root=tmp_path / "mlruns") == created


def test_setup_experiment_activates_it_and_runs_land_there(tmp_path):
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    experiment_id = setup_experiment("forecast_spend", tracking_uri=uri, artifact_root=tmp_path / "mlruns")
    with mlflow.start_run() as run:
        mlflow.log_metric("mae", 1.0)
    assert run.info.experiment_id == experiment_id
    assert mlflow.get_experiment(experiment_id).artifact_location.endswith("mlruns/forecast_spend")
