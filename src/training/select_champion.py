"""Step 7.8: the champion rule - pick the forecaster for each series, honestly.

Per series (income, total expenses and the 10 categories), over all 23 rolling-backtest months:

    LightGBM (median, settings from step 7.6) is champion only if its MAE is at least
    `forecast.champion_threshold` (10 %) lower than the best of the three baselines on exactly the
    same forecasts. Otherwise that best baseline stays champion.

ETS / SARIMAX (step 7.5) are not candidates: they were scored only on income and total expenses over
the last 6 months, and neither beat LightGBM or the best baseline there.

The forecast range always comes from the LightGBM 10th / 90th percentile models. The width factor
from step 7.4 is used only if it brought coverage closer to 80 % on the later (check) months;
otherwise the raw width (factor 1.0) is kept. When a baseline is champion, the range is stretched
to include its point forecast. The whole system (champion point + range) is then scored on the same
backtest, next to "best baseline everywhere" and "LightGBM everywhere".

Picking champions and scoring them on the same months flatters the result a little, so there is also
a check: champions picked on the earlier half of the test months only, scored on the later half.

Writes models/champions.json and logs a "champion_selection" run per family.

Run:  python -m src.training.select_champion
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.evaluation.metrics import coverage, mae, point_metrics
from src.forecasting.model import LIGHTGBM, apply_champions
from src.features.build_features import build_features
from src.training.train_baselines import series_group
from src.training.train_forecaster import (
    FAMILIES, FAMILY_MODE, TARGET_COVERAGE, backtest_family_quantiles, calibrate_width, compare_with_baselines,
    saved_params,
)
from src.utils.helpers import display_path, load_config, resolve_path
from src.utils.mlflow_utils import setup_experiment
from src.utils.logger import get_logger

log = get_logger(__name__)

def choose_champions(per_series: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """per_series from compare_with_baselines -> one champion per series, with the reason."""
    out = per_series.copy()
    wins = out["improvement"] >= threshold
    out["champion"] = np.where(wins, LIGHTGBM, out["best_baseline"])
    out["reason"] = [
        f"LightGBM MAE {m:,.2f} is {imp:.1%} below the best baseline ({b} {bm:,.2f}); threshold {threshold:.0%}"
        if win else
        f"LightGBM MAE {m:,.2f} is only {imp:+.1%} vs {b} {bm:,.2f}; below the {threshold:.0%} threshold, baseline kept"
        for win, m, imp, b, bm in zip(wins, out["model_mae"], out["improvement"], out["best_baseline"], out["baseline_mae"])]
    return out


def choose_width(factor: float, calibration: dict, target: float = TARGET_COVERAGE) -> tuple[float, str]:
    """Keep the calibrated width only if it moved coverage closer to the target on the check months."""
    raw, calibrated = calibration["coverage_raw_check"], calibration["coverage_calibrated_check"]
    if factor == 1.0:
        return 1.0, f"calibration chose factor 1.00 (no change): check-month coverage {raw:.1%}"
    if abs(calibrated - target) < abs(raw - target):
        return factor, f"calibrated factor {factor:.2f}: check-month coverage {raw:.1%} -> {calibrated:.1%}"
    return 1.0, (f"raw width kept: factor {factor:.2f} would move check-month coverage {raw:.1%} -> "
                 f"{calibrated:.1%}, further from {target:.0%}")


def select_family(table: pd.DataFrame, family: str, feature_columns: list[str], threshold: float,
                  params: dict | None = None, users: pd.DataFrame | None = None) -> dict:
    mode = FAMILY_MODE[family]
    predictions = backtest_family_quantiles(table, family, mode, feature_columns, params=params)
    comparison = compare_with_baselines(predictions, table, users)
    champions = choose_champions(comparison["per_series"], threshold)
    raw_factor, calibration = calibrate_width(predictions)
    factor, width_reason = choose_width(raw_factor, calibration)

    keys = ["user_id", "series", "month"]
    rows = table.set_index(keys).loc[pd.MultiIndex.from_frame(predictions[keys])].reset_index()
    final = apply_champions(rows, predictions["lower"].to_numpy(), predictions["predicted"].to_numpy(),
                            predictions["upper"].to_numpy(), dict(zip(champions["series"], champions["champion"])),
                            factor)
    final = pd.concat([predictions[keys + ["target_month", "actual"]], final.reset_index(drop=True)], axis=1)
    best_baseline = dict(zip(champions["series"], champions["best_baseline"]))
    baseline_only = apply_champions(rows, predictions["lower"].to_numpy(), predictions["predicted"].to_numpy(),
                                    predictions["upper"].to_numpy(), best_baseline, factor)["predicted"]

    final["series_group"] = series_group(final["series"])
    by_group = []
    for group, part in final.groupby("series_group"):
        by_group.append({"series_group": group, "n": len(part),
                         "champion_mae": mae(part["actual"], part["predicted"]),
                         "baseline_only_mae": mae(part["actual"], baseline_only[part.index]),
                         "lightgbm_only_mae": mae(part["actual"], predictions.loc[part.index, "predicted"]),
                         "coverage": coverage(part["actual"], part["lower"], part["upper"])})
    system = {
        **point_metrics(final["actual"], final["predicted"]),
        "coverage": coverage(final["actual"], final["lower"], final["upper"]),
        "mean_width": float((final["upper"] - final["lower"]).mean()),
        "baseline_only_mae": mae(final["actual"], baseline_only),
        "lightgbm_only_mae": mae(predictions["actual"], predictions["predicted"]),
    }
    check = holdout_check(predictions, rows, champions, threshold, factor, users)
    return {"family": family, "mode": mode, "champions": champions, "width_factor": factor, "check": check,
            "width_reason": width_reason, "calibration": calibration, "system": system,
            "by_group": pd.DataFrame(by_group), "predictions": final,
            "test_months": f"{final['target_month'].min()} to {final['target_month'].max()}"}


def holdout_check(predictions: pd.DataFrame, rows: pd.DataFrame, champions: pd.DataFrame, threshold: float,
                  factor: float, users: pd.DataFrame | None = None) -> dict:
    """Pick champions on the earlier test months only, score that choice on the later ones."""
    months = sorted(predictions["target_month"].unique())
    split = months[len(months) // 2]
    early = (predictions["target_month"] < split).to_numpy()
    picked = choose_champions(compare_with_baselines(predictions[early], rows, users)["per_series"], threshold)
    late = ~early

    def system(choice: dict) -> float:
        out = apply_champions(rows[late], predictions.loc[late, "lower"].to_numpy(),
                              predictions.loc[late, "predicted"].to_numpy(), predictions.loc[late, "upper"].to_numpy(),
                              choice, factor)
        return mae(predictions.loc[late, "actual"], out["predicted"])

    full = dict(zip(champions["series"], champions["champion"]))
    early_choice = dict(zip(picked["series"], picked["champion"]))
    return {"pick_months": f"{months[0]} to {split - 1}", "check_months": f"{split} to {months[-1]}",
            "agree": sum(early_choice.get(k) == v for k, v in full.items()), "series": len(full),
            "check_mae_champions": system(early_choice),
            "check_mae_best_baseline": system(dict(zip(picked["series"], picked["best_baseline"]))),
            "check_mae_lightgbm": mae(predictions.loc[late, "actual"], predictions.loc[late, "predicted"])}


def save_champions(results: dict, threshold: float, path: Path | None = None) -> Path:
    config = load_config()
    path = path or resolve_path(config["paths"]["models"]) / config["forecast"]["champions_file"]
    payload = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "rule": f"LightGBM is champion only if its MAE is >= {threshold:.0%} below the best baseline",
               "threshold": threshold, "metric": "mae", "families": {}}
    for family, r in results.items():
        payload["families"][family] = {
            "mode": r["mode"], "test_months": r["test_months"],
            "width_factor": r["width_factor"], "width_reason": r["width_reason"],
            "system": {k: round(float(v), 4) for k, v in r["system"].items()},
            "check": {k: round(v, 4) if isinstance(v, float) else v for k, v in r["check"].items()},
            "series": {row.series: {"champion": row.champion, "lightgbm_mae": round(row.model_mae, 2),
                                    "best_baseline": row.best_baseline, "baseline_mae": round(row.baseline_mae, 2),
                                    "improvement": round(row.improvement, 4), "reason": row.reason}
                       for row in r["champions"].itertuples()},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def log_to_mlflow(results: dict, threshold: float, champions_path: Path) -> None:
    for family, r in results.items():
        setup_experiment(FAMILIES[family])
        with mlflow.start_run(run_name="champion_selection"):
            mlflow.log_params({"family": family, "mode": r["mode"], "threshold": threshold,
                               "width_factor": r["width_factor"], "test_months": r["test_months"]})
            mlflow.log_metrics({f"system_{k}": v for k, v in r["system"].items()})
            mlflow.log_metrics({k: v for k, v in r["check"].items() if isinstance(v, (int, float))})
            for row in r["champions"].itertuples():
                tag = row.series.replace(":", "_")
                mlflow.set_tag(f"champion_{tag}", row.champion)
                mlflow.log_metric(f"improvement_{tag}", row.improvement)
            mlflow.log_artifact(str(champions_path))
            with tempfile.TemporaryDirectory() as tmp:
                for key in ("champions", "by_group"):
                    file = Path(tmp) / f"{key}.csv"
                    r[key].to_csv(file, index=False)
                    mlflow.log_artifact(str(file), "tables")


def _print(results: dict) -> None:
    for family, r in results.items():
        s = r["system"]
        log.info(f"\n=== {family} ({r['mode']}), {r['test_months']}")
        table = r["champions"][["series", "champion", "model_mae", "best_baseline", "baseline_mae", "improvement"]]
        log.info(table.to_string(index=False, formatters={"model_mae": "€{:,.2f}".format, "baseline_mae": "€{:,.2f}".format,
                                                          "improvement": "{:+.1%}".format}))
        log.info(f"range: {r['width_reason']}")
        log.info(f"system MAE €{s['mae']:,.2f} (best baseline everywhere €{s['baseline_only_mae']:,.2f}, LightGBM "
                 f"everywhere €{s['lightgbm_only_mae']:,.2f}) · coverage {s['coverage']:.1%} · mean width €{s['mean_width']:,.0f}")
        c = r["check"]
        log.info(f"check: picked on {c['pick_months']}, scored on {c['check_months']}: MAE €{c['check_mae_champions']:,.2f} "
                 f"(best baseline €{c['check_mae_best_baseline']:,.2f}, LightGBM €{c['check_mae_lightgbm']:,.2f}); "
                 f"{c['agree']}/{c['series']} series pick the same champion as the full period")
        log.info(r["by_group"].to_string(index=False, formatters={
               "champion_mae": "€{:,.2f}".format, "baseline_only_mae": "€{:,.2f}".format,
               "lightgbm_only_mae": "€{:,.2f}".format, "coverage": "{:.1%}".format}))


def main() -> None:
    config = load_config()
    threshold = config["forecast"]["champion_threshold"]
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    users = pd.read_csv(resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["users_file"])
    table, info = build_features(monthly)
    params = saved_params()
    results = {family: select_family(table, family, info["feature_columns"], threshold, params.get(family), users)
               for family in FAMILIES}
    _print(results)
    path = save_champions(results, threshold)
    log_to_mlflow(results, threshold, path)
    log.info(f"\nSaved {display_path(path)}")


if __name__ == "__main__":
    main()
