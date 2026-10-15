"""Main forecasting models: LightGBM, one global model per family, scored with the backtest.

Families
    income   the series "income"
    spend    total expenses and every expenses:<Category> ("series" is a categorical input)

Target modes (step 7.3 compares them)
    raw        predict next month's value in euro directly
    relative   predict log(1 + next) - log(1 + scale), scale = the series' recent level
               (6-month mean, else 3-month mean, else last month). Rent at €1,200 and Health at €40
               then look alike to one global model. The L1 / quantile loss is applied to the
               transformed target; since log is monotonic, the predicted median / quantile maps back.

Step 7.3: median models (L1 loss) per family and mode on all 23 backtest months, compared with the
best baseline on exactly the same rows, logged to MLflow (one parent run, one nested run per fold).

Run:  python -m src.training.train_forecaster
"""

from dataclasses import dataclass, field

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from src.evaluation.backtest import backtest_folds
from src.evaluation.metrics import mae, point_metrics, summarize
from src.features.build_features import build_features
from src.training.train_baselines import BASELINES, series_group
from src.utils.helpers import load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment

FAMILIES = {"income": "forecast_income", "spend": "forecast_spend"}
MODES = ("raw", "relative")
DEFAULT_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 40,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "verbose": -1, "n_jobs": 4,
}


def family_rows(table: pd.DataFrame, family: str) -> pd.DataFrame:
    is_income = table["series"] == "income"
    rows = table[is_income] if family == "income" else table[~is_income]
    return rows.reset_index(drop=True)


def scale_of(rows: pd.DataFrame) -> pd.Series:
    """The series' recent level, used by the relative mode."""
    return rows["roll_mean_6"].fillna(rows["roll_mean_3"]).fillna(rows["lag_1"]).fillna(0.0).clip(lower=0.0)


def to_model_target(rows: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "raw":
        return rows["target"]
    return np.log1p(rows["target"].clip(lower=0)) - np.log1p(scale_of(rows))


def from_model_output(rows: pd.DataFrame, output: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return np.clip(output, 0, None)
    return np.clip(np.expm1(output + np.log1p(scale_of(rows).to_numpy())), 0, None)


def model_inputs(rows: pd.DataFrame, feature_columns: list[str], series_levels: list[str]) -> pd.DataFrame:
    X = rows[feature_columns].copy()
    if len(series_levels) > 1:
        X["series"] = pd.Categorical(rows["series"], categories=series_levels)
    return X


@dataclass
class ForecastModel:
    """A fitted LightGBM model plus everything needed to use it on new rows."""
    family: str
    mode: str
    quantile: float
    feature_columns: list[str]
    series_levels: list[str]
    params: dict = field(default_factory=dict)
    booster: lgb.LGBMRegressor | None = None

    def fit(self, rows: pd.DataFrame) -> "ForecastModel":
        objective = {"objective": "l1"} if self.quantile == 0.5 else {"objective": "quantile", "alpha": self.quantile}
        self.booster = lgb.LGBMRegressor(**{**DEFAULT_PARAMS, **self.params, **objective})
        self.booster.fit(model_inputs(rows, self.feature_columns, self.series_levels), to_model_target(rows, self.mode))
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        output = self.booster.predict(model_inputs(rows, self.feature_columns, self.series_levels))
        return from_model_output(rows, output, self.mode)


def backtest_family(table: pd.DataFrame, family: str, mode: str, feature_columns: list[str],
                    quantile: float = 0.5, params: dict | None = None, last_n_folds: int | None = None,
                    on_fold=None) -> pd.DataFrame:
    """Train on the past and predict the next month for every backtest fold of one family.

    Returns one row per forecast: user_id, series, month, target_month, actual, predicted.
    on_fold(fold_number, test_month, fold_predictions) is called after each fold (for nested MLflow runs).
    """
    rows = family_rows(table, family)
    levels = sorted(rows["series"].unique())
    folds = backtest_folds(rows)
    if last_n_folds:
        folds = folds[-last_n_folds:]
    out = []
    for number, fold in enumerate(folds):
        train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        model = ForecastModel(family, mode, quantile, feature_columns, levels, params or {}).fit(train)
        predicted = test[["user_id", "series", "month", "target_month"]].assign(
            actual=test["target"].to_numpy(), predicted=model.predict(test))
        out.append(predicted)
        if on_fold:
            on_fold(number, fold.test_month, predicted)
    return pd.concat(out, ignore_index=True)


def compare_with_baselines(predictions: pd.DataFrame, table: pd.DataFrame,
                           users: pd.DataFrame | None = None) -> dict:
    """MAE of the model next to the best baseline per series, on exactly the same forecasts."""
    keys = ["user_id", "series", "month"]
    rows = table.set_index(keys).loc[pd.MultiIndex.from_frame(predictions[keys])].reset_index()
    joined = predictions.copy()
    for name, forecast in BASELINES.items():
        joined[name] = forecast(rows).to_numpy()
    joined["series_group"] = series_group(joined["series"])
    if users is not None:
        joined = joined.merge(users[["user_id", "archetype"]], on="user_id", how="left")

    per_series = []
    for series, part in joined.groupby("series"):
        baseline_mae = {name: mae(part["actual"], part[name]) for name in BASELINES}
        best = min(baseline_mae, key=baseline_mae.get)
        model_mae = mae(part["actual"], part["predicted"])
        per_series.append({"series": series, "model_mae": model_mae, "best_baseline": best,
                           "baseline_mae": baseline_mae[best],
                           "improvement": 1 - model_mae / baseline_mae[best] if baseline_mae[best] > 0 else 0.0})
    result = {"per_series": pd.DataFrame(per_series), "joined": joined,
              "by_group": summarize(joined, "series_group")}
    if "archetype" in joined:
        rows_ = []
        for (group, archetype), part in joined.groupby(["series_group", "archetype"]):
            best = min(mae(part["actual"], part[name]) for name in BASELINES)
            rows_.append({"series_group": group, "archetype": archetype,
                          "model_mae": mae(part["actual"], part["predicted"]), "best_baseline_mae": best})
        result["by_archetype"] = pd.DataFrame(rows_)
    return result


def run_median_comparison(table: pd.DataFrame, feature_columns: list[str], users: pd.DataFrame | None = None,
                          log: bool = True) -> dict:
    """Step 7.3: both families x both target modes, logged to MLflow with nested fold runs."""
    results = {}
    for family, experiment in FAMILIES.items():
        for mode in MODES:
            if log:
                setup_experiment(experiment)
                parent = mlflow.start_run(run_name=f"lgbm_median_{mode}")
                mlflow.log_params({**DEFAULT_PARAMS, "family": family, "mode": mode, "quantile": 0.5,
                                   "features": len(feature_columns)})

            def on_fold(number, test_month, fold_predictions):
                if log:
                    with mlflow.start_run(run_name=f"fold_{number:02d}_{test_month}", nested=True):
                        mlflow.log_metric("mae", mae(fold_predictions["actual"], fold_predictions["predicted"]))

            predictions = backtest_family(table, family, mode, feature_columns, on_fold=on_fold)
            comparison = compare_with_baselines(predictions, table, users)
            overall = point_metrics(predictions["actual"], predictions["predicted"])
            results[(family, mode)] = {"overall": overall, **comparison, "predictions": predictions}
            if log:
                mlflow.log_metrics({k: v for k, v in overall.items() if k != "n"})
                for row in comparison["per_series"].itertuples():
                    tag = row.series.replace(":", "_")
                    mlflow.log_metrics({f"mae_{tag}": row.model_mae, f"improvement_{tag}": row.improvement})
                mlflow.end_run()
    return results


def _print(results: dict) -> None:
    for (family, mode), r in results.items():
        o = r["overall"]
        print(f"\n=== {family} · {mode}: MAE €{o['mae']:,.2f} · RMSE €{o['rmse']:,.2f} · sMAPE {o['smape']:.1f}% · n={o['n']:,}")
        table = r["per_series"].assign(improvement=lambda d: (d["improvement"] * 100).round(1).astype(str) + " %")
        print(table.to_string(index=False, formatters={"model_mae": "€{:,.2f}".format, "baseline_mae": "€{:,.2f}".format}))
        if "by_archetype" in r:
            print(r["by_archetype"].round(2).to_string(index=False))


def main() -> None:
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    users = pd.read_csv(resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["users_file"])
    table, info = build_features(monthly)
    results = run_median_comparison(table, info["feature_columns"], users)
    _print(results)
    for family in FAMILIES:
        best = min(MODES, key=lambda m: results[(family, m)]["overall"]["mae"])
        print(f"\n{family}: better target mode = {best}")


if __name__ == "__main__":
    main()
