"""Monthly macro data from the ECB Data Portal: German inflation (HICP) and the ECB key rate.

fetch_macro()  downloads both series (free, no API key) and returns one row per month.
load_macro()   is what the rest of the project uses: it tries a fresh download when asked,
               otherwise (or when offline) it reads the local cache, then the committed copy
               in data/sample/, so the app and pipelines also work without internet.

Run on its own with:  python -m src.ingestion.macro_api
"""

import io
from pathlib import Path

import pandas as pd
import requests

from src.utils.helpers import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

MACRO_COLUMNS = ["month", "inflation_de", "ecb_rate"]


class MacroDataUnavailable(RuntimeError):
    """Raised when there is neither internet access nor any cached macro file."""


def _settings() -> dict:
    return load_config()["macro"]


def _cache_paths() -> tuple[Path, Path]:
    config = load_config()
    raw = resolve_path(config["paths"]["data_raw"]) / config["macro"]["cache_file"]
    committed = resolve_path(config["paths"]["data_sample"]) / config["macro"]["committed_cache_file"]
    return raw, committed


def parse_ecb_csv(text: str) -> pd.Series:
    """Turn an ECB 'csvdata' response into a monthly series (last observation of each month).

    Monthly series have one value per month; daily series (like the key rate) keep the
    rate in force at the end of the month.
    """
    df = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    if df.empty:
        raise ValueError("The ECB response contains no observations.")
    df["month"] = pd.PeriodIndex(pd.to_datetime(df["TIME_PERIOD"]), freq="M")
    df = df.dropna(subset=["OBS_VALUE"]).sort_values("TIME_PERIOD")
    return df.groupby("month")["OBS_VALUE"].last().astype(float)


def fetch_series(series_key: str, start: str, base_url: str, timeout: float = 20) -> pd.Series:
    """Download one ECB series from `start` (YYYY-MM) until the latest available month."""
    response = requests.get(
        f"{base_url}/{series_key}",
        params={"format": "csvdata", "startPeriod": start, "detail": "dataonly"},
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_ecb_csv(response.text)


def fetch_macro(start: str | None = None) -> pd.DataFrame:
    """Download all configured series and join them into one monthly table."""
    settings = _settings()
    start = start or settings["start"]
    columns = {
        name: fetch_series(key, start, settings["base_url"], settings["timeout_seconds"])
        for name, key in settings["series"].items()
    }
    macro = pd.DataFrame(columns)
    # the key rate only changes on decision days: carry the last known value into later months
    macro["ecb_rate"] = macro["ecb_rate"].ffill()
    macro = macro.rename_axis("month").reset_index()
    return macro[MACRO_COLUMNS]


def save_macro(macro: pd.DataFrame, paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        macro.to_csv(path, index=False)


def read_macro_csv(path: Path) -> pd.DataFrame:
    macro = pd.read_csv(path)
    macro["month"] = pd.PeriodIndex(macro["month"], freq="M")
    return macro[MACRO_COLUMNS]


def load_macro(refresh: bool = False, cache_paths: tuple[Path, Path] | None = None) -> pd.DataFrame:
    """Return the monthly macro table.

    With refresh=True a fresh download is tried first and written to both caches.
    Without internet (or refresh=False) the local cache is used, then the committed copy.
    """
    raw_cache, committed_cache = cache_paths or _cache_paths()
    if refresh:
        try:
            macro = fetch_macro()
            save_macro(macro, [raw_cache, committed_cache])
            logger.info("Downloaded %d months of macro data from the ECB.", len(macro))
            return macro
        except (requests.RequestException, ValueError) as err:
            logger.warning("ECB download failed (%s); using the cached macro data instead.", err)

    for path in (raw_cache, committed_cache):
        if path.exists():
            return read_macro_csv(path)
    raise MacroDataUnavailable(
        "No macro data available: the ECB could not be reached and no cache exists. "
        "Run `python -m src.ingestion.macro_api` once with internet access."
    )


def main() -> None:
    try:
        macro = load_macro(refresh=True)
    except MacroDataUnavailable as err:
        logger.info(err)
        raise SystemExit(1)
    logger.info(f"{len(macro)} months of macro data ({macro['month'].min()} to {macro['month'].max()})")
    logger.info(macro.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
