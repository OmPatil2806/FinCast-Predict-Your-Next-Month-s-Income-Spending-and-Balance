"""Step 7.5: classic statistical models as a second comparison point.

    ets      Holt-Winters exponential smoothing: additive damped trend, additive yearly season
             (season only once a series has 24 months, otherwise trend only)
    sarimax  SARIMAX(1,0,0)(1,0,0,12) with a constant

Unlike LightGBM these are local models: one fit per user and series, on that series' own history
only. They forecast income and total expenses one month ahead for the last 6 backtest months, and
are scored next to the best baseline and the LightGBM median model on exactly the same forecasts.
A fit that fails, or a series that is flat (no information to fit), falls back to last month's value;
the number of fallbacks is reported. Results go to the forecast_income / forecast_spend experiments.

Run:  python -m src.training.train_statistical
"""

import warnings

import mlflow
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.evaluation.backtest import backtest_folds
from src.evaluation.metrics import summarize
from src.features.build_features import build_features
from src.training.train_baselines import BASELINES
from src.training.train_forecaster import FAMILY_MODE, backtest_family
from src.utils.helpers import load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment
from src.utils.logger import get_logger

log = get_logger(__name__)

SERIES = {"income": "income", "expenses": "spend"}          # series -> LightGBM family
EXPERIMENT = {"income": "forecast_income", "expenses": "forecast_spend"}
LAST_N_FOLDS = 6
SEASON = 12


def _fit_ets(history: np.ndarray) -> float:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = "add" if len(history) >= 2 * SEASON else None
    model = ExponentialSmoothing(history, trend="add", damped_trend=True, seasonal=seasonal,
                                 seasonal_periods=SEASON if seasonal else None)
    return float(model.fit().forecast(1)[0])


def _fit_sarimax(history: np.ndarray) -> float:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(history, order=(1, 0, 0), seasonal_order=(1, 0, 0, SEASON), trend="c")
    return float(model.fit(disp=False).forecast(1)[0])


MODELS = {"ets": _fit_ets, "sarimax": _fit_sarimax}


def forecast_one(history: np.ndarray, model: str) -> tuple[float, bool]:
    """One-step forecast from a series' own history. Returns (forecast, used_fallback)."""
    last = float(history[-1])
    if len(history) < 6 or np.ptp(history) == 0:
        return last, True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value = MODELS[model](history.astype(float))
        except Exception:
            return last, True
    if not np.isfinite(value):
        return last, True
    return max(value, 0.0), False                     # amounts are never negative


def histories(table: pd.DataFrame, series: str) -> dict:
    """user_id -> that user's monthly values of one series, as a month-indexed Series (value = lag_1 at month)."""
    rows = table[table["series"] == series].sort_values("month")
    return {user: part.set_index("month")["lag_1"] for user, part in rows.groupby("user_id")}


def test_rows(table: pd.DataFrame, series: str, last_n_folds: int = LAST_N_FOLDS) -> pd.DataFrame:
    """The forecasts to make: the same test rows (and months) the rolling backtest uses."""
    rows = table[table["series"] == series].reset_index(drop=True)
    folds = backtest_folds(rows)[-last_n_folds:]
    return pd.concat([rows.iloc[fold.test_idx] for fold in folds], ignore_index=True)


def _forecast_user(history: pd.Series, origins: list, model: str) -> list[tuple]:
    out = []
    for origin in origins:
        value, fallback = forecast_one(history[history.index <= origin].to_numpy(), model)
        out.append((origin, value, fallback))
    return out


def backtest_statistical(table: pd.DataFrame, series: str, model: str, last_n_folds: int = LAST_N_FOLDS,
                         n_jobs: int = -1) -> pd.DataFrame:
    """Refit per user and test month, using only data up to the forecast origin (month t)."""
    tests = test_rows(table, series, last_n_folds)
    history = histories(table, series)
    users = sorted(tests["user_id"].unique())
    origins = {user: sorted(part["month"]) for user, part in tests.groupby("user_id")}
    results = Parallel(n_jobs=n_jobs)(delayed(_forecast_user)(history[user], origins[user], model) for user in users)
    forecasts = pd.DataFrame([(user, origin, value, fallback)
                              for user, rows in zip(users, results) for origin, value, fallback in rows],
                             columns=["user_id", "month", "predicted", "fallback"])
    out = tests[["user_id", "series", "month", "target_month", "target"]].rename(columns={"target": "actual"})
    return out.merge(forecasts, on=["user_id", "month"], how="left")


def compare(table: pd.DataFrame, feature_columns: list[str], users: pd.DataFrame | None = None,
            last_n_folds: int = LAST_N_FOLDS, n_jobs: int = -1, lgbm_params: dict | None = None) -> dict:
    """ETS, SARIMAX, LightGBM (median) and the three baselines on the same income / expenses forecasts."""
    keys = ["user_id", "series", "month"]
    results = {}
    for series, family in SERIES.items():
        joined = test_rows(table, series, last_n_folds)[keys + ["target_month", "target"]].rename(
            columns={"target": "actual"})
        rows = table.set_index(keys).loc[pd.MultiIndex.from_frame(joined[keys])].reset_index()
        for name, forecast in BASELINES.items():
            joined[name] = forecast(rows).to_numpy()
        fallbacks = {}
        for model in MODELS:
            predicted = backtest_statistical(table, series, model, last_n_folds, n_jobs)
            joined = joined.merge(predicted[keys + ["predicted", "fallback"]].rename(
                columns={"predicted": model, "fallback": f"{model}_fallback"}), on=keys, how="left")
            fallbacks[model] = int(joined[f"{model}_fallback"].sum())
        lgbm = backtest_family(table, family, FAMILY_MODE[family], feature_columns,
                               params=lgbm_params, last_n_folds=last_n_folds)
        joined = joined.merge(lgbm.loc[lgbm["series"] == series, keys + ["predicted"]].rename(
            columns={"predicted": "lightgbm"}), on=keys, how="left")
        if users is not None:
            joined = joined.merge(users[["user_id", "archetype"]], on="user_id", how="left")

        forecasters = list(BASELINES) + list(MODELS) + ["lightgbm"]
        long = pd.concat([joined.assign(model=name, predicted=joined[name]) for name in forecasters],
                         ignore_index=True)
        overall = summarize(long, "model").set_index("model").loc[forecasters].reset_index()
        best_baseline = overall[overall["model"].isin(list(BASELINES))].sort_values("mae").iloc[0]
        overall["vs_best_baseline"] = 1 - overall["mae"] / best_baseline["mae"]
        result = {"joined": joined, "overall": overall, "best_baseline": best_baseline["model"],
                  "fallbacks": fallbacks, "n": len(joined),
                  "test_months": f"{joined['target_month'].min()} to {joined['target_month'].max()}"}
        if "archetype" in joined:
            result["by_archetype"] = summarize(long, ["archetype", "model"]).pivot_table(
                index="archetype", columns="model", values="mae")[forecasters]
        results[series] = result
    return results


def log_to_mlflow(results: dict) -> None:
    for series, r in results.items():
        setup_experiment(EXPERIMENT[series])
        for model in MODELS:
            row = r["overall"].set_index("model").loc[model]
            with mlflow.start_run(run_name=f"{model}_{series}"):
                mlflow.log_params({"model": model, "series": series, "test_months": r["test_months"],
                                   "local_fit": "per user and test month"})
                mlflow.log_metrics({"mae": row["mae"], "rmse": row["rmse"], "smape": row["smape"],
                                    "improvement_vs_best_baseline": row["vs_best_baseline"],
                                    "fallbacks": r["fallbacks"][model], "n": r["n"]})
                if "by_archetype" in r:
                    for archetype, value in r["by_archetype"][model].items():
                        mlflow.log_metric(f"mae_{archetype}", value)


def _print(results: dict) -> None:
    for series, r in results.items():
        log.info(f"\n=== {series}: {r['n']} forecasts, test months {r['test_months']} "
                 f"(fallbacks: {', '.join(f'{m} {n}' for m, n in r['fallbacks'].items())})")
        log.info(r["overall"][["model", "mae", "rmse", "smape", "vs_best_baseline"]].to_string(index=False, formatters={
               "mae": "€{:,.2f}".format, "rmse": "€{:,.2f}".format, "smape": "{:.1f}".format,
               "vs_best_baseline": "{:+.1%}".format}))
        if "by_archetype" in r:
            log.info("MAE (€) by user type:\n" + r["by_archetype"].round(2).to_string())


def main() -> None:
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    users = pd.read_csv(resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["users_file"])
    table, info = build_features(monthly)
    results = compare(table, info["feature_columns"], users)
    _print(results)
    log_to_mlflow(results)


if __name__ == "__main__":
    main()
