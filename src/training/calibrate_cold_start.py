"""Step 8.2: how much worse are forecasts for users with only a few months of history?

Users the models never saw (20 % held out per origin, see holdout_user_folds) get their history cut to
3, 4, 5, 6 or 12 months; their features are rebuilt from only that history, and models trained on the
other users forecast their next month. Per history length this measures the champion system's MAE and
how often the 10-90 % range holds the real value.

Users with fewer than `forecast.cold_start_months` (6) months get their range widened by one factor:
the smallest factor that brings coverage back to the 80 % target, chosen on the first half of the
origins and checked on the second half. Written to models/cold_start.json for the predictor.

Run:  python -m src.training.calibrate_cold_start
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.backtest import holdout_user_folds
from src.evaluation.metrics import coverage, mae
from src.features.build_features import build_features
from src.forecasting.model import apply_champions
from src.training.export_champion import load_champions
from src.training.train_forecaster import (
    FAMILY_MODE, QUANTILES, TARGET_COVERAGE, ForecastModel, family_rows, saved_params,
)
from src.utils.helpers import display_path, load_config, resolve_path
from src.utils.logger import get_logger

log = get_logger(__name__)

HISTORIES = (3, 4, 5, 6, 12)
FACTORS = np.round(np.arange(1.0, 4.01, 0.05), 2)


def holdout_predictions(monthly: pd.DataFrame, feature_columns: list[str], champions: dict, origins: list,
                        params: dict | None = None, seed: int = 42) -> pd.DataFrame:
    """Champion forecasts with raw (un-widened beyond the export) LightGBM ranges for held-out users."""
    params = params or {}
    out = []
    for i, origin in enumerate(origins):
        folds = holdout_user_folds(monthly, build_features, histories=HISTORIES,
                                   origin=pd.Period(origin, freq="M"), seed=seed + i)
        train = folds[0].train
        for family in ("income", "spend"):
            entry = champions["families"][family]
            rows = family_rows(train, family)
            levels = sorted(rows["series"].unique())
            models = {q: ForecastModel(family, entry["mode"], q, feature_columns, levels, params.get(family, {}))
                      .fit(rows) for q in QUANTILES}
            for fold in folds:
                test = family_rows(fold.test, family)
                test = test[test["series"].isin(levels)].reset_index(drop=True)
                values = np.sort(np.column_stack([models[q].predict(test) for q in QUANTILES]), axis=1)
                final = apply_champions(test, values[:, 0], values[:, 1], values[:, 2],
                                        {s: v["champion"] for s, v in entry["series"].items()}, entry["width_factor"])
                out.append(test[["user_id", "series"]].assign(
                    origin=str(origin), history=fold.history_months, family=family,
                    actual=test["target"].to_numpy(), median=values[:, 1], **{c: final[c].to_numpy()
                                                                              for c in ("lower", "predicted", "upper")}))
    return pd.concat(out, ignore_index=True)


def widen_around(predictions: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Stretch each range's distance to the point forecast by `factor` (lower bound never below 0)."""
    return predictions.assign(
        lower=np.clip(predictions["predicted"] - factor * (predictions["predicted"] - predictions["lower"]), 0, None),
        upper=predictions["predicted"] + factor * (predictions["upper"] - predictions["predicted"]))


def choose_factor(short: pd.DataFrame, target: float = TARGET_COVERAGE) -> float:
    """Smallest factor whose coverage on these rows reaches the target (1.0 if already there)."""
    for factor in FACTORS:
        w = widen_around(short, factor)
        if coverage(w["actual"], w["lower"], w["upper"]) >= target:
            return float(factor)
    return float(FACTORS[-1])


def calibrate(predictions: pd.DataFrame, cold_start_months: int, target: float = TARGET_COVERAGE) -> dict:
    origins = sorted(predictions["origin"].unique())
    pick, check = origins[: len(origins) // 2], origins[len(origins) // 2:]
    is_short = predictions["history"] < cold_start_months
    factors = {}
    for family in ("income", "spend"):
        fam = predictions["family"] == family
        factors[family] = choose_factor(predictions[fam & is_short & predictions["origin"].isin(pick)], target)

    by_history = []
    for (family, history), part in predictions.groupby(["family", "history"]):
        late = part[part["origin"].isin(check)]
        factor = factors[family] if history < cold_start_months else 1.0
        widened = widen_around(late, factor)
        by_history.append({"family": family, "history": int(history), "n": len(part),
                           "mae": mae(part["actual"], part["predicted"]),
                           "coverage": coverage(part["actual"], part["lower"], part["upper"]),
                           "check_coverage_raw": coverage(late["actual"], late["lower"], late["upper"]),
                           "check_coverage_widened": coverage(widened["actual"], widened["lower"], widened["upper"]),
                           "factor": factor})
    by_history = pd.DataFrame(by_history)
    full = by_history[by_history["history"] == max(HISTORIES)].set_index("family")["mae"]
    by_history["mae_vs_12_months"] = by_history["mae"] / by_history["family"].map(full)
    return {"factors": factors, "pick_origins": pick, "check_origins": check, "by_history": by_history}


def save(result: dict, cold_start_months: int, path: Path | None = None) -> Path:
    config = load_config()
    path = path or resolve_path(config["paths"]["models"]) / config["forecast"]["cold_start_file"]
    table = result["by_history"]
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "min_history_months": config["forecast"]["min_history_months"],
        "cold_start_months": cold_start_months,
        "width_factor": result["factors"],
        "method": "held-out users, history cut to N months; factor = smallest widening that reaches 80 % "
                  "coverage on the pick origins, checked on the others",
        "pick_origins": result["pick_origins"], "check_origins": result["check_origins"],
        "by_history": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}
                       for row in table.to_dict("records")],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def main() -> None:
    config = load_config()
    forecast = config["forecast"]
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    _, info = build_features(monthly[monthly["user_id"] == monthly["user_id"].iloc[0]])   # just the feature list
    predictions = holdout_predictions(monthly, info["feature_columns"], load_champions(),
                                      forecast["cold_start_origins"], saved_params(), config["project"]["seed"])
    result = calibrate(predictions, forecast["cold_start_months"])
    log.info(f"Users never seen by the models, history cut to N months; factor picked on {result['pick_origins']}, "
             f"checked on {result['check_origins']}\n")
    log.info(result["by_history"].to_string(index=False, formatters={
           "mae": "€{:,.2f}".format, "coverage": "{:.1%}".format, "check_coverage_raw": "{:.1%}".format,
           "check_coverage_widened": "{:.1%}".format, "mae_vs_12_months": "x{:.2f}".format, "factor": "{:.2f}".format}))
    path = save(result, forecast["cold_start_months"])
    log.info(f"\nWidening for fewer than {forecast['cold_start_months']} months: {result['factors']} · "
             f"saved {display_path(path)}")


if __name__ == "__main__":
    main()
