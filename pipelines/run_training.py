"""Stage 2 of the pipeline: raw transactions -> model-ready tables -> trained, exported models.

    python pipelines/run_training.py                       # data + all models (~8 min)
    python pipelines/run_training.py --data-only           # only the data steps
    python pipelines/run_training.py --train-categorizer   # also retrain + log the categoriser first
    python pipelines/run_training.py --tune                # also re-run the Optuna search (+~10 min)
    python pipelines/run_training.py --experiments         # also the 7.3 / 7.4 / 7.5 comparisons (+~10 min)

Data steps:
  (optional) retrain the categoriser and log it to MLflow (src/training/train_categorizer.py)
  categorise every transaction with the app's own categorize() and compare with the true labels
  aggregate to one row per user x month x category (+ "ALL" totals and end balance)
  reconcile the table with the raw transactions, then save data/processed/monthly.parquet
  build the model features (src/features/build_features.py), save data/processed/features.parquet
Model steps (each one is also runnable on its own with python -m src.training.<module>):
  baselines (6.9) -> [tuning 7.6] -> [comparisons 7.3-7.5] -> champion rule (7.8)
  -> export + registry (7.9) -> short-history calibration (8.2) -> SHAP explanations (7.7)
  -> anomaly model (7.10)
Without --tune the settings already in models/forecast_params.json are used.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.features.build_features import build_features  # noqa: E402
from src.preprocessing.aggregate_monthly import (  # noqa: E402
    ReconciliationError, aggregate_user_month_category, reconcile,
)
from src.preprocessing.categorizer import categorize, load_categorizer  # noqa: E402
from src.utils.helpers import load_config, resolve_path  # noqa: E402


def _step(number: int, total: int, label: str, message: str, seconds: float) -> None:
    print(f"[{number}/{total}] {label:.<24} {message}  ({seconds:.1f}s)")


def load_transactions(config: dict) -> pd.DataFrame:
    path = resolve_path(config["paths"]["data_raw"]) / config["synthetic"]["transactions_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python pipelines/run_ingestion.py` first.")
    return pd.read_csv(path, parse_dates=["date"])


def categorize_transactions(transactions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Replace the generator's labels with what the app itself would assign, and report agreement."""
    predicted = categorize(transactions, load_categorizer())
    report = {
        "agreement": float((predicted["category"] == transactions["category"]).mean()),
        "by_source": predicted["category_source"].value_counts(dropna=False).to_dict(),
        "needs_review": int(predicted["needs_review"].sum()),
    }
    return transactions.assign(true_category=transactions["category"], category=predicted["category"]), report


def save_monthly(monthly: pd.DataFrame, config: dict) -> Path:
    out_dir = resolve_path(config["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / config["processed"]["monthly_file"]
    # store the month as a date (first day of the month): portable across parquet readers
    monthly.assign(month=monthly["month"].dt.to_timestamp()).to_parquet(path, index=False)
    return path


def save_features(features: pd.DataFrame, config: dict) -> Path:
    path = resolve_path(config["paths"]["data_processed"]) / config["processed"]["features_file"]
    periods = {c: features[c].dt.to_timestamp() for c in ("month", "target_month")}
    features.assign(**periods).to_parquet(path, index=False)
    return path


def model_steps(tune: bool = False, experiments: bool = False) -> list[tuple[str, str, object]]:
    """(label, what it produces, function) for every model step, in order. Imported lazily (heavy)."""
    from src.training import (
        calibrate_cold_start, explain_forecaster, export_champion, select_champion, train_anomaly, train_baselines,
        train_forecaster, train_statistical, tune_forecaster,
    )
    steps = [("Baselines ", "models/baseline_scores.json", train_baselines.main)]
    if tune:
        steps.append(("Tuning (Optuna) ", "models/forecast_params.json", lambda: tune_forecaster.main([])))
    if experiments:
        steps += [("LightGBM median ", "MLflow comparison runs", lambda: train_forecaster.main([])),
                  ("LightGBM ranges ", "MLflow coverage runs", lambda: train_forecaster.main(["--ranges"])),
                  ("ETS / SARIMAX ", "MLflow comparison runs", train_statistical.main)]
    steps += [("Champion rule ", "models/champions.json", select_champion.main),
              ("Export + registry ", "models/forecaster_*.joblib, model_meta.json", export_champion.main),
              ("Short histories ", "models/cold_start.json", calibrate_cold_start.main),
              ("SHAP explanations ", "reports/figures/", explain_forecaster.main),
              ("Anomaly model ", "models/anomaly.joblib, anomaly_meta.json", train_anomaly.main)]
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-categorizer", action="store_true", help="retrain and log the categoriser first")
    parser.add_argument("--data-only", action="store_true", help="stop after the features")
    parser.add_argument("--tune", action="store_true", help="re-run the Optuna search (step 7.6)")
    parser.add_argument("--experiments", action="store_true", help="also run the 7.3 / 7.4 / 7.5 comparisons")
    args = parser.parse_args(argv)

    config = load_config()
    steps = [] if args.data_only else model_steps(args.tune, args.experiments)
    total = (5 if args.train_categorizer else 4) + len(steps)
    step = 0
    began = time.perf_counter()

    if args.train_categorizer:
        from src.training.train_categorizer import main as train_categorizer_main
        step += 1
        t = time.perf_counter()
        train_categorizer_main()
        _step(step, total, "Categoriser ", "retrained, saved to models/ and logged to MLflow", time.perf_counter() - t)

    step += 1
    t = time.perf_counter()
    try:
        transactions = load_transactions(config)
    except FileNotFoundError as err:
        print(err)
        return 1
    transactions, report = categorize_transactions(transactions)
    sources = ", ".join(f"{k}: {v:,}" for k, v in report["by_source"].items())
    _step(step, total, "Categorise ", f"{len(transactions):,} rows ({sources}) · "
          f"{report['agreement']:.2%} match the true labels · {report['needs_review']:,} need review",
          time.perf_counter() - t)

    step += 1
    t = time.perf_counter()
    monthly = aggregate_user_month_category(transactions)
    users, months = monthly["user_id"].nunique(), monthly["month"].nunique()
    _step(step, total, "Aggregate ", f"{len(monthly):,} rows · {users} users × {months} months × "
          f"{monthly['category'].nunique()} (categories + ALL)", time.perf_counter() - t)

    step += 1
    t = time.perf_counter()
    try:
        check = reconcile(monthly, transactions)
    except ReconciliationError as err:
        print(f"[{step}/{total}] Reconcile FAILED - {err}")
        return 1
    path = save_monthly(monthly, config)
    _step(step, total, "Reconcile + save ", f"€{check['monthly_total']:,.2f} = raw €{check['raw_total']:,.2f} ✓ "
          f"→ {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)", time.perf_counter() - t)

    step += 1
    t = time.perf_counter()
    features, info = build_features(monthly)
    path = save_features(features, config)
    usable = int(features["target"].notna().sum())
    macro = "with" if info["macro"] else "WITHOUT (no ECB cache yet)"
    _step(step, total, "Features ", f"{len(features):,} rows ({usable:,} with a target) × "
          f"{len(info['feature_columns'])} features, {macro} macro → {path.relative_to(ROOT)} "
          f"({path.stat().st_size / 1024:.0f} KB)", time.perf_counter() - t)

    for label, produces, run in steps:
        step += 1
        print(f"\n[{step}/{total}] {label}{'-' * (24 - len(label))}")
        t = time.perf_counter()
        run()
        _step(step, total, label, f"done → {produces}", time.perf_counter() - t)

    print(f"\nDone in {time.perf_counter() - began:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
