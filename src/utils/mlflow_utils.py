"""One place to set up MLflow, so every training script logs to the same store the same way.

    setup_experiment("forecast_spend")   -> tracking URI from config.yaml (sqlite:///mlflow.db at the
                                            project root), experiment created on first use with its
                                            artifacts under mlruns/<experiment>/, then made active
    create_all_experiments()             -> the five FinCast experiments listed in config.yaml

Run once to create them all:  python -m src.utils.mlflow_utils
"""

from pathlib import Path

import mlflow

from src.utils.helpers import load_config, mlflow_tracking_uri, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)


def setup_experiment(key: str, tracking_uri: str | None = None, artifact_root: Path | None = None) -> str:
    """Point MLflow at the project's tracking store and activate the experiment `key` (a key of
    mlflow.experiments in config.yaml, or a plain name). Returns the experiment id."""
    experiments = load_config()["mlflow"]["experiments"]
    name = experiments.get(key, key)
    mlflow.set_tracking_uri(tracking_uri or mlflow_tracking_uri())
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        default_root = resolve_path(load_config()["mlflow"].get("artifact_root", "mlruns"))
        location = Path(artifact_root or default_root) / name
        experiment_id = mlflow.create_experiment(name, artifact_location=location.as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(name)
    return experiment_id


def create_all_experiments(tracking_uri: str | None = None, artifact_root: Path | None = None) -> dict:
    """Create (or find) every experiment listed in config.yaml; returns {name: experiment id}."""
    names = load_config()["mlflow"]["experiments"]
    return {name: setup_experiment(key, tracking_uri, artifact_root) for key, name in names.items()}


if __name__ == "__main__":
    for name, experiment_id in create_all_experiments().items():
        log.info(f"{name:<16} id {experiment_id}")
    log.info(f"Tracking store: {mlflow_tracking_uri()}\nOpen the UI with:  mlflow ui --backend-store-uri sqlite:///mlflow.db")
