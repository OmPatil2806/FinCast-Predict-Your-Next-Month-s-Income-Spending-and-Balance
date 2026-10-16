"""Step 7.7: explain the LightGBM forecasts with SHAP and save the plots as MLflow artifacts.

For each family the median model (settings from models/forecast_params.json, else the defaults) is
trained on everything before the last test month and explains its forecasts for that month. SHAP
values come from LightGBM's own exact TreeSHAP (pred_contrib) and are drawn with the shap package.

They are in the model's target units: euro for spend (raw mode); for income (relative mode, a
log-ratio to the series' recent level) they are shown x100 as log-points, so +10 is roughly +10 %.

Written to reports/figures/ (and logged to the forecast_income / forecast_spend experiments):
    shap_summary_<family>.png        mean |SHAP| per input (overall importance)
    shap_beeswarm_<family>.png       how high / low input values push forecasts up / down
    shap_example_<family>.png        one forecast broken down input by input (the forecast at the
                                     90th percentile of total |SHAP|: a clearly explained case, not an outlier)
    forecast_vs_actual_<family>.png  forecast against the real value for that month
    reports/shap_importance_<family>.csv

Run:  python -m src.training.explain_forecaster
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from src.evaluation.backtest import backtest_folds  # noqa: E402
from src.evaluation.metrics import point_metrics  # noqa: E402
from src.features.build_features import build_features  # noqa: E402
from src.training.train_forecaster import (  # noqa: E402
    FAMILIES, FAMILY_MODE, ForecastModel, family_rows, model_inputs, saved_params,
)
from src.utils.helpers import display_path, load_config, resolve_path  # noqa: E402
from src.utils.mlflow_utils import setup_experiment  # noqa: E402

EXAMPLE_SERIES = {"income": "income", "spend": "expenses"}
UNITS = {"raw": "€", "relative": "log-points vs recent level, ≈ %"}
SCALE = {"raw": 1.0, "relative": 100.0}                 # display factor for the SHAP values


def explain_family(table: pd.DataFrame, family: str, feature_columns: list[str], params: dict | None = None) -> dict:
    """Fit on all months before the last test month, explain the forecasts for that month."""
    rows = family_rows(table, family)
    fold = backtest_folds(rows)[-1]
    train, test = rows.iloc[fold.train_idx], rows.iloc[fold.test_idx].reset_index(drop=True)
    mode = FAMILY_MODE[family]
    levels = sorted(rows["series"].unique())
    model = ForecastModel(family, mode, 0.5, feature_columns, levels, params or {}).fit(train)

    X = model_inputs(test, feature_columns, levels)
    contrib = model.booster.predict(X, pred_contrib=True)              # last column = expected value
    values, base = contrib[:, :-1], contrib[:, -1]
    predicted = model.predict(test)

    importance = pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(values).mean(axis=0) * SCALE[mode]}) \
        .sort_values("mean_abs_shap", ascending=False, ignore_index=True)
    importance["share"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum()

    display = X.copy()
    if "series" in display:
        display["series"] = display["series"].cat.codes                  # numbers for the beeswarm colours
    scale = SCALE[mode]
    explanation = shap.Explanation(values=values * scale, base_values=base * scale,
                                   data=display.to_numpy(dtype=float), feature_names=list(X.columns))
    candidates = np.flatnonzero((test["series"] == EXAMPLE_SERIES[family]).to_numpy())
    strength = np.abs(values[candidates]).sum(axis=1)
    example = int(candidates[np.argsort(strength)[int(0.9 * (len(candidates) - 1))]])

    return {"family": family, "mode": mode, "test_month": str(fold.test_month), "rows": test,
            "predicted": predicted, "explanation": explanation, "importance": importance,
            "example": example, "metrics": point_metrics(test["target"], predicted),
            "additivity_error": float(np.abs(values.sum(axis=1) + base
                                             - model.booster.predict(X, raw_score=True)).max())}


def save_plots(result: dict, figures: Path) -> list[Path]:
    figures.mkdir(parents=True, exist_ok=True)
    family, unit, month = result["family"], UNITS[result["mode"]], result["test_month"]
    explanation, rows = result["explanation"], result["rows"]
    paths = []

    def save(name: str, title: str) -> None:
        plt.title(title, fontsize=11)
        plt.tight_layout()
        path = figures / f"{name}_{family}.png"
        plt.savefig(path, dpi=110, bbox_inches="tight")
        plt.close("all")
        paths.append(path)

    shap.plots.bar(explanation, max_display=15, show=False)
    save("shap_summary", f"{family}: mean |SHAP| per input ({unit}), forecasts for {month}")

    np.random.seed(0)                                     # the beeswarm jitters points randomly
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    save("shap_beeswarm", f"{family}: input value vs push on the forecast ({unit})")

    i = result["example"]
    shap.plots.waterfall(explanation[i], max_display=12, show=False)
    save("shap_example", f"{family}: {rows.loc[i, 'user_id']} {rows.loc[i, 'series']} for {month} "
                         f"(forecast €{result['predicted'][i]:,.0f}, actual €{rows.loc[i, 'target']:,.0f})")

    fig, ax = plt.subplots(figsize=(6, 6))
    actual, predicted = rows["target"].to_numpy(), result["predicted"]
    ax.scatter(actual + 1, predicted + 1, s=8, alpha=0.35)
    top = max(actual.max(), predicted.max()) + 1
    ax.plot([1, top], [1, top], color="grey", linewidth=1, linestyle="--")
    ax.set(xscale="log", yscale="log", xlabel="actual € (+1, log scale)", ylabel="forecast € (+1, log scale)")
    m = result["metrics"]
    save("forecast_vs_actual", f"{family}: forecast vs actual, {month} (MAE €{m['mae']:,.2f}, n={m['n']})")
    return paths


def log_to_mlflow(result: dict, paths: list[Path], table_path: Path) -> None:
    setup_experiment(FAMILIES[result["family"]])
    with mlflow.start_run(run_name=f"shap_{result['mode']}"):
        mlflow.log_params({"family": result["family"], "mode": result["mode"], "test_month": result["test_month"]})
        mlflow.log_metrics({k: v for k, v in result["metrics"].items() if k != "n"})
        for row in result["importance"].head(10).itertuples():
            mlflow.log_metric(f"shap_share_{row.feature}", row.share)
        for path in paths + [table_path]:
            mlflow.log_artifact(str(path), "explain")


def main() -> None:
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    table, info = build_features(monthly)
    params = saved_params()
    reports = resolve_path(config["paths"].get("reports", "reports"))
    for family in FAMILIES:
        result = explain_family(table, family, info["feature_columns"], params.get(family))
        paths = save_plots(result, reports / "figures")
        table_path = reports / f"shap_importance_{family}.csv"
        result["importance"].round(5).to_csv(table_path, index=False)
        log_to_mlflow(result, paths, table_path)
        top = result["importance"].head(8)
        print(f"\n=== {family} ({result['mode']}, SHAP in {UNITS[result['mode']]}), forecasts for "
              f"{result['test_month']}: MAE €{result['metrics']['mae']:,.2f}, "
              f"SHAP additivity error {result['additivity_error']:.1e}")
        print(top.to_string(index=False, formatters={"mean_abs_shap": "{:,.4f}".format, "share": "{:.1%}".format}))
        print("saved:", ", ".join(display_path(p) for p in paths))


if __name__ == "__main__":
    main()
