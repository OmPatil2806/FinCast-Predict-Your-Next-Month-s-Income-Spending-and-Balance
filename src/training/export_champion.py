"""Step 7.9: train the final champion on all data, register it in MLflow and export it to models/.

Per family (income, spend):
    1. fit the 10th / 50th / 90th percentile LightGBM models on every row with a known target
       (no hold-out now - the backtests in 7.3-7.8 already measured how well this recipe does),
       with the settings from step 7.6 and the champions + range width from step 7.8
    2. wrap them in a ChampionForecaster and save it with joblib to models/forecaster_<family>.joblib
    3. reload the file and check it forecasts exactly like the in-memory model
    4. log it as an MLflow pyfunc model, register it as fincast_forecaster_<family> and point the
       alias "champion" at the new version

models/model_meta.json records the version, date, git commit, feature list, training data, settings,
champion per series and the backtest metrics. The version goes up by one on every export.

Run:  python -m src.training.export_champion
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm
import mlflow
import numpy as np
import pandas as pd
import sklearn
from mlflow.tracking import MlflowClient

from src.features.build_features import build_features
from src.forecasting.model import ChampionForecaster
from src.training.train_forecaster import FAMILIES, QUANTILES, ForecastModel, family_rows, saved_params
from src.utils.helpers import load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment
from src.utils.logger import get_logger

log = get_logger(__name__)


class ForecasterPyfunc(mlflow.pyfunc.PythonModel):
    """MLflow wrapper so the registry can serve the exported joblib bundle."""

    def load_context(self, context):
        self.model = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        return self.model.predict(model_input)


def load_champions(path: Path | None = None) -> dict:
    config = load_config()
    path = path or resolve_path(config["paths"]["models"]) / config["forecast"]["champions_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run step 7.8 first: python -m src.training.select_champion")
    return json.loads(path.read_text())


def next_version(meta_path: Path) -> int:
    if meta_path.exists():
        return int(json.loads(meta_path.read_text()).get("version", 0)) + 1
    return 1


def train_family(table: pd.DataFrame, family: str, feature_columns: list[str], champions: dict,
                 params: dict | None = None, version: int | None = None) -> ChampionForecaster:
    """Final models on every row with a target."""
    entry = champions["families"][family]
    rows = family_rows(table, family).dropna(subset=["target"])
    levels = sorted(rows["series"].unique())
    models = {q: ForecastModel(family, entry["mode"], q, feature_columns, levels, params or {}).fit(rows)
              for q in QUANTILES}
    return ChampionForecaster(family=family, mode=entry["mode"], models=models,
                              champions={s: v["champion"] for s, v in entry["series"].items()},
                              width_factor=entry["width_factor"], version=version)


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=resolve_path("."), check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def register(model_path: Path, family: str, metrics: dict, version: int, tracking_uri: str | None = None,
             artifact_root: Path | None = None) -> dict:
    """Log the bundle as a pyfunc model, register it and move the 'champion' alias to it."""
    config = load_config()
    name = config["forecast"]["registry_name"].format(family=family)
    setup_experiment(FAMILIES[family], tracking_uri, artifact_root)
    with mlflow.start_run(run_name=f"champion_export_v{version}") as run:
        mlflow.log_params({"family": family, "export_version": version})
        mlflow.log_metrics(metrics)
        info = mlflow.pyfunc.log_model(
            name="model", python_model=ForecasterPyfunc(), artifacts={"bundle": str(model_path)},
            code_paths=[str(resolve_path("src"))], registered_model_name=name,
            pip_requirements=[f"lightgbm=={lightgbm.__version__}", f"pandas=={pd.__version__}",
                              f"numpy=={np.__version__}", f"mlflow=={mlflow.__version__}"])
    client = MlflowClient()
    registry_version = int(info.registered_model_version)
    client.set_registered_model_alias(name, "champion", str(registry_version))
    return {"name": name, "version": registry_version, "alias": "champion", "run_id": run.info.run_id}


def export(table: pd.DataFrame, feature_columns: list[str], champions: dict, params: dict | None = None,
           out_dir: Path | None = None, register_models: bool = True, tracking_uri: str | None = None,
           artifact_root: Path | None = None) -> dict:
    config = load_config()
    out_dir = out_dir or resolve_path(config["paths"]["models"])
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / config["forecast"]["meta_file"]
    version = next_version(meta_path)
    params = params or {}
    with_target = table.dropna(subset=["target"])
    latest = table[table["month"] == table["month"].max()]            # rows the app would forecast from

    meta = {
        "version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "feature_columns": list(feature_columns),
        "quantiles": list(QUANTILES),
        "training_data": {"users": int(table["user_id"].nunique()), "rows": int(len(with_target)),
                          "target_months": f"{with_target['target_month'].min()} to {with_target['target_month'].max()}"},
        "champion_rule": champions["rule"],
        "libraries": {"python": platform.python_version(), "lightgbm": lightgbm.__version__,
                      "scikit-learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
        "families": {},
    }
    for family in FAMILIES:
        model = train_family(table, family, feature_columns, champions, params.get(family), version)
        path = out_dir / config["forecast"]["model_file"].format(family=family)
        joblib.dump(model, path, compress=3)

        rows = family_rows(latest, family)
        reloaded = joblib.load(path).predict(rows)
        pd.testing.assert_frame_equal(reloaded, model.predict(rows))     # the file forecasts like the model

        entry = champions["families"][family]
        family_meta = {
            "file": path.name, "size_kb": round(path.stat().st_size / 1024, 1), "mode": model.mode,
            "params": params.get(family, "defaults"), "width_factor": model.width_factor,
            "series_levels": model.models[0.5].series_levels, "champions": model.champions,
            "backtest": {"test_months": entry["test_months"], **entry["system"], "check": entry.get("check")},
        }
        if register_models:
            metrics = {k: v for k, v in entry["system"].items() if isinstance(v, (int, float))}
            family_meta["registry"] = register(path, family, metrics, version, tracking_uri, artifact_root)
        meta["families"][family] = family_meta
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    return meta


def main() -> None:
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    table, info = build_features(monthly)
    meta = export(table, info["feature_columns"], load_champions(), saved_params())
    log.info(f"Exported champion version {meta['version']} (trained on {meta['training_data']['rows']:,} rows, "
             f"{meta['training_data']['target_months']})")
    for family, m in meta["families"].items():
        lightgbm_series = [s for s, c in m["champions"].items() if c == "lightgbm"]
        registry = m.get("registry", {})
        log.info(f"  {family}: models/{m['file']} ({m['size_kb']} KB) · LightGBM for {len(lightgbm_series)}/"
                 f"{len(m['champions'])} series · width x{m['width_factor']:.2f} · registry "
                 f"{registry.get('name')} v{registry.get('version')} @{registry.get('alias')}")
    log.info(f"Saved models/{config['forecast']['meta_file']}")


if __name__ == "__main__":
    main()
