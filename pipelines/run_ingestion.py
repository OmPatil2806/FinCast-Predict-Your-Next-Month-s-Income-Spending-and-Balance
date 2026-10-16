"""Stage 1 of the pipeline: collect all raw data into data/raw/.

    python pipelines/run_ingestion.py

1. synthetic users and transactions (src/ingestion/synthetic_generator.py), then data checks
2. macro data from the ECB, with the cached copy as fallback (src/ingestion/macro_api.py)
3. monthly calendar features: holidays and working days (src/utils/helpers.py)

A missing ECB connection is not fatal: the step reports it and the pipeline continues.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.ingestion.macro_api import MacroDataUnavailable, load_macro  # noqa: E402
from src.ingestion.synthetic_generator import generate_dataset, save_dataset  # noqa: E402
from src.preprocessing.cleaning import DataValidationError, validate_transactions  # noqa: E402
from src.utils.helpers import display_path, holidays_per_month, load_config, resolve_path, set_seed  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def _step(number: int, total: int, label: str, message: str, seconds: float) -> None:
    logger.info(f"[{number}/{total}] {label:.<24} {message}  ({seconds:.1f}s)")


def run_synthetic(config: dict, raw_dir: Path) -> pd.DataFrame:
    settings = config["synthetic"]
    transactions, users = generate_dataset(
        n_users=settings["n_users"], months=settings["months"],
        start=settings["start"], seed=config["project"]["seed"],
    )
    validate_transactions(transactions)   # stop before saving if the generator produced bad data
    save_dataset(transactions, users, raw_dir, settings["transactions_file"], settings["users_file"])
    return transactions


def run_macro() -> tuple[pd.DataFrame | None, str]:
    try:
        macro = load_macro(refresh=True)
    except MacroDataUnavailable as err:
        return None, f"SKIPPED - {err}"
    return macro, f"{len(macro)} months · {macro['month'].min()} → {macro['month'].max()}"


def run_calendar(config: dict, raw_dir: Path, transactions: pd.DataFrame) -> pd.DataFrame:
    calendar = holidays_per_month(transactions["date"].min(), transactions["date"].max())
    calendar.to_csv(raw_dir / config["calendar"]["file"], index=False)
    return calendar


def main() -> int:
    config = load_config()
    set_seed(config["project"]["seed"])
    raw_dir = resolve_path(config["paths"]["data_raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()

    t = time.perf_counter()
    try:
        transactions = run_synthetic(config, raw_dir)
    except DataValidationError as err:
        logger.error(f"[1/3] Synthetic users FAILED\n{err}")
        return 1
    _step(1, 3, "Synthetic users ",
          f"{len(transactions):,} transactions · {transactions['user_id'].nunique()} users · "
          f"{transactions['date'].min():%Y-%m} → {transactions['date'].max():%Y-%m} · checks passed",
          time.perf_counter() - t)

    t = time.perf_counter()
    macro, message = run_macro()
    _step(2, 3, "Macro data (ECB) ", message, time.perf_counter() - t)

    t = time.perf_counter()
    calendar = run_calendar(config, raw_dir, transactions)
    _step(3, 3, "Calendar ", f"{len(calendar)} months · {int((calendar['holidays'] > 0).sum())} with holidays",
          time.perf_counter() - t)

    files = ", ".join(sorted(p.name for p in raw_dir.glob("*.csv")))
    logger.info(f"Done in {time.perf_counter() - began:.1f}s. Files in {display_path(raw_dir)}/: {files}")
    if macro is None:
        logger.warning("Note: macro data is missing. Run `python -m src.ingestion.macro_api` with internet "
                       "access before Phase 6 (features).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
