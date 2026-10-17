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
Step 7.4: 10th / 50th / 90th percentile models -> a forecast range. Its coverage is checked, and a
width factor is calibrated on the earlier test months and verified on the later ones.

Run:  python -m src.training.train_forecaster            (7.3 median comparison)
      python -m src.training.train_forecaster --ranges   (7.4 ranges)
"""

import argparse
import json

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.backtest import backtest_folds
from src.evaluation.metrics import coverage, mae, pinball_loss, point_metrics, summarize
from src.features.build_features import build_features
from src.forecasting.model import (  # noqa: F401  (re-exported: training code and tests import them from here)
    DEFAULT_PARAMS, ForecastModel, from_model_output, model_inputs, scale_of, to_model_target, widen,
)
from src.training.train_baselines import BASELINES, series_group
from src.utils.helpers import load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment
from src.utils.logger import get_logger

log = get_logger(__name__)

FAMILIES = {"income": "forecast_income", "spend": "forecast_spend"}
MODES = ("raw", "relative")
FAMILY_MODE = {"income": "relative", "spend": "raw"}   # lower overall MAE in the 7.3 comparison
QUANTILES = (0.1, 0.5, 0.9)
TARGET_COVERAGE = 0.80                                  # a 10-90 % range should hold ~80 % of real values


def saved_params() -> dict:
    """Per-family settings chosen by step 7.6 (models/forecast_params.json); empty dict -> the defaults."""
    config = load_config()
    path = resolve_path(config["paths"]["models"]) / config["forecast"]["params_file"]
    if not path.exists():
        return {}
    return {family: entry["params"] for family, entry in json.loads(path.read_text(encoding="utf-8"))["families"].items()}


def family_rows(table: pd.DataFrame, family: str) -> pd.DataFrame:
    is_income = table["series"] == "income"
    rows = table[is_income] if family == "income" else table[~is_income]
    return rows.reset_index(drop=True)


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


def backtest_family_quantiles(table: pd.DataFrame, family: str, mode: str, feature_columns: list[str],
                              quantiles=QUANTILES, params: dict | None = None,
                              last_n_folds: int | None = None) -> pd.DataFrame:
    """Like backtest_family, but fits one model per quantile in every fold.

    Returns actual, lower (q10), predicted (median) and upper (q90). Quantile models are trained
    separately, so they can cross; each row's three values are sorted to keep lower <= median <= upper.
    """
    rows = family_rows(table, family)
    levels = sorted(rows["series"].unique())
    folds = backtest_folds(rows)
    if last_n_folds:
        folds = folds[-last_n_folds:]
    out = []
    for fold in folds:
        train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx]
        values = np.column_stack([
            ForecastModel(family, mode, q, feature_columns, levels, params or {}).fit(train).predict(test)
            for q in quantiles])
        values.sort(axis=1)                                       # no crossing quantiles
        out.append(test[["user_id", "series", "month", "target_month"]].assign(
            actual=test["target"].to_numpy(), lower=values[:, 0], predicted=values[:, 1], upper=values[:, 2]))
    return pd.concat(out, ignore_index=True)


def calibrate_width(predictions: pd.DataFrame, target: float = TARGET_COVERAGE,
                    factors=np.round(np.arange(0.5, 3.01, 0.05), 2)) -> tuple[float, dict]:
    """Choose the width factor on the EARLIER half of the test months, check it on the LATER half.

    Fitting the factor on the same months it is judged on would flatter the result.
    """
    months = sorted(predictions["target_month"].unique())
    split = months[len(months) // 2]
    early, late = predictions[predictions["target_month"] < split], predictions[predictions["target_month"] >= split]
    gaps = {f: abs(coverage(early["actual"], *widen(early, f)[["lower", "upper"]].T.to_numpy()) - target) for f in factors}
    factor = float(min(gaps, key=gaps.get))
    report = {
        "factor": factor,
        "calibration_months": f"{months[0]} to {split - 1}",
        "check_months": f"{split} to {months[-1]}",
        "coverage_raw_check": coverage(late["actual"], late["lower"], late["upper"]),
        "coverage_calibrated_check": coverage(late["actual"], *widen(late, factor)[["lower", "upper"]].T.to_numpy()),
    }
    return factor, report


def run_ranges(table: pd.DataFrame, feature_columns: list[str], users: pd.DataFrame | None = None,
               log: bool = True) -> dict:
    """Step 7.4: quantile models per family, coverage before/after calibration, pinball loss."""
    results = {}
    for family, experiment in FAMILIES.items():
        mode = FAMILY_MODE[family]
        predictions = backtest_family_quantiles(table, family, mode, feature_columns)
        factor, calibration = calibrate_width(predictions)
        calibrated = widen(predictions, factor)
        calibrated["series_group"] = series_group(calibrated["series"])
        if users is not None:
            calibrated = calibrated.merge(users[["user_id", "archetype"]], on="user_id", how="left")
        metrics = {
            "coverage_raw": coverage(predictions["actual"], predictions["lower"], predictions["upper"]),
            "coverage_calibrated": coverage(calibrated["actual"], calibrated["lower"], calibrated["upper"]),
            "pinball_q10": pinball_loss(predictions["actual"], predictions["lower"], 0.1),
            "pinball_q50": pinball_loss(predictions["actual"], predictions["predicted"], 0.5),
            "pinball_q90": pinball_loss(predictions["actual"], predictions["upper"], 0.9),
            "mae": mae(predictions["actual"], predictions["predicted"]),
            "mean_width_calibrated": float((calibrated["upper"] - calibrated["lower"]).mean()),
        }
        by = ["series_group", "archetype"] if "archetype" in calibrated else "series_group"
        results[family] = {"mode": mode, "factor": factor, "calibration": calibration, "metrics": metrics,
                           "predictions": calibrated,
                           "by_group": summarize(calibrated, by, lower="lower", upper="upper")}
        if log:
            setup_experiment(experiment)
            with mlflow.start_run(run_name=f"lgbm_quantiles_{mode}"):
                mlflow.log_params({**DEFAULT_PARAMS, "family": family, "mode": mode,
                                   "quantiles": "0.1,0.5,0.9", "width_factor": factor})
                mlflow.log_metrics({**metrics, "coverage_raw_check": calibration["coverage_raw_check"],
                                    "coverage_calibrated_check": calibration["coverage_calibrated_check"]})
    return results


def _print_ranges(results: dict) -> None:
    for family, r in results.items():
        m, c = r["metrics"], r["calibration"]
        log.info(f"\n=== {family} ({r['mode']}): coverage raw {m['coverage_raw']:.1%} -> calibrated "
                 f"{m['coverage_calibrated']:.1%} (width factor {r['factor']:.2f})")
        log.info(f"    factor chosen on {c['calibration_months']}, checked on {c['check_months']}: "
                 f"{c['coverage_raw_check']:.1%} -> {c['coverage_calibrated_check']:.1%}")
        log.info(f"    pinball q10 {m['pinball_q10']:.2f} · q50 {m['pinball_q50']:.2f} · q90 {m['pinball_q90']:.2f} · "
                 f"mean range width €{m['mean_width_calibrated']:,.0f}")
        table = r["by_group"][[c for c in r["by_group"].columns if c in ("series_group", "archetype", "mae", "coverage", "n")]]
        log.info(table.to_string(index=False, formatters={"mae": "€{:,.2f}".format, "coverage": "{:.1%}".format}))


def _print(results: dict) -> None:
    for (family, mode), r in results.items():
        o = r["overall"]
        log.info(f"\n=== {family} · {mode}: MAE €{o['mae']:,.2f} · RMSE €{o['rmse']:,.2f} · sMAPE {o['smape']:.1f}% · n={o['n']:,}")
        table = r["per_series"].assign(improvement=lambda d: (d["improvement"] * 100).round(1).astype(str) + " %")
        log.info(table.to_string(index=False, formatters={"model_mae": "€{:,.2f}".format, "baseline_mae": "€{:,.2f}".format}))
        if "by_archetype" in r:
            log.info(r["by_archetype"].round(2).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranges", action="store_true", help="step 7.4: quantile models and coverage")
    args = parser.parse_args(argv)
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    users = pd.read_csv(resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["users_file"])
    table, info = build_features(monthly)
    if args.ranges:
        _print_ranges(run_ranges(table, info["feature_columns"], users))
        return
    results = run_median_comparison(table, info["feature_columns"], users)
    _print(results)
    for family in FAMILIES:
        best = min(MODES, key=lambda m: results[(family, m)]["overall"]["mae"])
        log.info(f"\n{family}: better target mode = {best}")


if __name__ == "__main__":
    main()
