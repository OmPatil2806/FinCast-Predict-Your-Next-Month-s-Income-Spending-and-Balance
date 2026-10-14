"""Stage 2 of the pipeline: turn the raw transactions into model-ready tables (and, later, models).

    python pipelines/run_training.py                       # categorise, aggregate, check, save
    python pipelines/run_training.py --train-categorizer   # also retrain + log the categoriser first

Current steps (Phase 5):
1. (optional) retrain the categoriser and log it to MLflow (src/training/train_categorizer.py)
2. categorise every transaction with the app's own categorize() and compare with the true labels
3. aggregate to one row per user x month x category (+ "ALL" totals and end balance)
4. reconcile the table with the raw transactions, then save data/processed/monthly.parquet
Phases 6-7 add features, baselines and the forecasting models here.
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-categorizer", action="store_true", help="retrain and log the categoriser first")
    args = parser.parse_args(argv)

    config = load_config()
    total = 4 if args.train_categorizer else 3
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

    print(f"Done in {time.perf_counter() - began:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
