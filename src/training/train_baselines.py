"""The bar to beat: three simple forecasts, scored with the same backtest the models will use.

    last_month            next month = this month (lag_1)
    same_month_last_year  next month = the same month a year earlier (lag_12; lag_1 while there is no year yet)
    mean_3                next month = average of the last 3 months (roll_mean_3; lag_1 while shorter)

Scored on every test month of the rolling backtest (step 6.7) and on unseen users with only
3 / 6 / 12 months of history. Results are logged to the MLflow experiment "baselines", and the best
baseline MAE per series is written to models/baseline_scores.json for the champion rule in Phase 7.

Run:  python -m src.training.train_baselines
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.backtest import backtest_folds, holdout_user_folds
from src.evaluation.metrics import point_metrics, summarize
from src.features.build_features import build_features
from src.utils.helpers import load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment

BASELINES = {
    "last_month": lambda t: t["lag_1"],
    "same_month_last_year": lambda t: t["lag_12"].fillna(t["lag_1"]),
    "mean_3": lambda t: t["roll_mean_3"].fillna(t["lag_1"]),
}


def series_group(series: pd.Series) -> pd.Series:
    """income / expenses (total) / category (any expenses:<Category>)."""
    return series.where(series.isin(["income", "expenses"]), "category")


def predictions_on_folds(table: pd.DataFrame, folds) -> pd.DataFrame:
    """Every baseline's forecast for every test row of the rolling backtest (one row per forecast)."""
    test_rows = np.concatenate([fold.test_idx for fold in folds])
    test = table.iloc[test_rows]
    frames = [test[["user_id", "series", "month", "target_month"]].assign(
                 baseline=name, actual=test["target"].to_numpy(), predicted=forecast(test).to_numpy())
              for name, forecast in BASELINES.items()]
    out = pd.concat(frames, ignore_index=True)
    out["series_group"] = series_group(out["series"])
    return out


def evaluate(table: pd.DataFrame, users: pd.DataFrame | None = None, monthly: pd.DataFrame | None = None,
             seed: int = 42) -> dict:
    """All result tables: overall, per series group, per series, per user type, new users, best per series."""
    predictions = predictions_on_folds(table, backtest_folds(table))
    if users is not None:
        predictions = predictions.merge(users[["user_id", "archetype"]], on="user_id", how="left")

    results = {
        "test_months": int(predictions["target_month"].nunique()),
        "predictions": predictions,
        "overall": summarize(predictions, "baseline"),
        "by_group": summarize(predictions, ["series_group", "baseline"]),
        "by_series": summarize(predictions, ["series", "baseline"]),
    }
    if "archetype" in predictions:
        results["by_archetype"] = summarize(predictions, ["series_group", "archetype", "baseline"])

    best = results["by_series"].loc[results["by_series"].groupby("series")["mae"].idxmin()]
    results["best_by_series"] = best[["series", "baseline", "mae", "smape"]].reset_index(drop=True)

    if monthly is not None:
        rows = []
        for fold in holdout_user_folds(monthly, build_features, seed=seed):
            for name, forecast in BASELINES.items():
                rows.append({"history_months": fold.history_months, "baseline": name,
                             **point_metrics(fold.test["target"], forecast(fold.test))})
        results["new_users"] = pd.DataFrame(rows)
    return results


def save_scores(results: dict, path: Path | None = None) -> Path:
    """Best baseline per series (the champion rule in Phase 7 must beat these)."""
    config = load_config()
    path = path or resolve_path(config["paths"]["models"]) / config["baselines"]["scores_file"]
    best = results["best_by_series"]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_months": results["test_months"],
        "metric": "mae",
        "best_by_series": {row.series: {"baseline": row.baseline, "mae": round(row.mae, 2), "smape": round(row.smape, 2)}
                           for row in best.itertuples()},
        "overall_by_baseline": {row.baseline: round(row.mae, 2) for row in results["overall"].itertuples()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def log_to_mlflow(results: dict) -> list[str]:
    """One run per baseline in the 'baselines' experiment, with the full tables as CSV artifacts."""
    setup_experiment("baselines")

    run_ids = []
    for baseline in BASELINES:
        with mlflow.start_run(run_name=baseline) as run:
            mlflow.log_params({"baseline": baseline, "test_months": results["test_months"]})
            overall = results["overall"].set_index("baseline").loc[baseline]
            mlflow.log_metrics({"mae": overall["mae"], "rmse": overall["rmse"], "smape": overall["smape"]})
            for row in results["by_group"][results["by_group"]["baseline"] == baseline].itertuples():
                mlflow.log_metrics({f"mae_{row.series_group}": row.mae, f"smape_{row.series_group}": row.smape})
            if "new_users" in results:
                for row in results["new_users"][results["new_users"]["baseline"] == baseline].itertuples():
                    mlflow.log_metric(f"mae_new_user_{row.history_months}m", row.mae)
            with tempfile.TemporaryDirectory() as tmp:
                for key in ("by_group", "by_series", "by_archetype", "new_users", "best_by_series"):
                    if key in results:
                        file = Path(tmp) / f"{key}.csv"
                        results[key].to_csv(file, index=False)
                        mlflow.log_artifact(str(file), "tables")
            run_ids.append(run.info.run_id)
    return run_ids


def _print(results: dict) -> None:
    fmt = {"mae": "{:,.2f}".format, "rmse": "{:,.2f}".format, "smape": "{:.1f}".format}
    print(f"Baselines over {results['test_months']} test months\n")
    print(results["by_group"].to_string(index=False, formatters=fmt), "\n")
    if "by_archetype" in results:
        pivot = results["by_archetype"].pivot_table(index=["series_group", "archetype"], columns="baseline", values="mae")
        print("MAE (€) by user type:\n" + pivot.round(2).to_string(), "\n")
    if "new_users" in results:
        pivot = results["new_users"].pivot_table(index="history_months", columns="baseline", values="mae")
        print("MAE (€) for unseen users by months of history:\n" + pivot.round(2).to_string(), "\n")
    print("Best baseline per series (the bar to beat):\n"
          + results["best_by_series"].to_string(index=False, formatters=fmt))


def main() -> None:
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    users = pd.read_csv(resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["users_file"])
    table, _ = build_features(monthly)
    results = evaluate(table, users=users, monthly=monthly, seed=config["project"]["seed"])
    _print(results)
    path = save_scores(results)
    run_ids = log_to_mlflow(results)
    print(f"\nSaved {path.relative_to(resolve_path('.'))} · logged {len(run_ids)} runs to MLflow experiment "
          f"'{config['mlflow']['experiments']['baselines']}'")


if __name__ == "__main__":
    main()
