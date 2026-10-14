"""Small shared helpers: project root, config loading, paths, random seeds and the holiday calendar."""

import random
from functools import lru_cache
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import yaml


def get_project_root() -> Path:
    """Return the repository root (the folder that holds config/ and src/)."""
    # this file lives at <root>/src/utils/helpers.py
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=None)
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Read the YAML settings once and return them as a dictionary."""
    with open(get_project_root() / config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Turn a path from config.yaml (relative to the project root) into a full path."""
    return get_project_root() / relative_path


def mlflow_tracking_uri() -> str:
    """The MLflow tracking URI from config.yaml, with a relative sqlite path anchored at the project root
    (so runs land in the same mlflow.db no matter which folder a script is started from)."""
    uri = load_config()["mlflow"]["tracking_uri"]
    prefix = "sqlite:///"
    if uri.startswith(prefix) and not Path(uri[len(prefix):]).is_absolute():
        return prefix + (get_project_root() / uri[len(prefix):]).as_posix()
    return uri


@lru_cache(maxsize=None)
def load_categories() -> dict:
    """Category rules from config/categories.yaml: {'fallback': 'Other', 'categories': [...]}."""
    with open(resolve_path(load_config()["categories_file"]), encoding="utf-8") as f:
        return yaml.safe_load(f)


def category_names() -> list[str]:
    """All category names in the order they are listed in categories.yaml."""
    return [c["name"] for c in load_categories()["categories"]]


def set_seed(seed: int | None = None) -> int:
    """Fix the random generators so results can be reproduced. Defaults to the config seed."""
    if seed is None:
        seed = load_config()["project"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    return seed


# ---------- calendar ----------

def _country(country: str | None) -> str:
    return country or load_config()["locale"]["country"]


def get_holidays(years, country: str | None = None) -> pd.DataFrame:
    """National public holidays as a table with date and name (country from config.yaml if not given)."""
    years = [years] if isinstance(years, int) else list(years)
    calendar = holidays.country_holidays(_country(country), years=years)
    table = pd.DataFrame(sorted(calendar.items()), columns=["date", "name"])
    table["date"] = pd.to_datetime(table["date"])
    return table


def holidays_per_month(start, end, country: str | None = None) -> pd.DataFrame:
    """Calendar features for every month from start to end (inclusive).

    Columns: month (Period), days_in_month, holidays (public holidays in the month),
    working_days (Mon-Fri that are not a holiday). Used as model features in Phase 6.
    """
    months = pd.period_range(pd.Period(start, freq="M"), pd.Period(end, freq="M"), freq="M")
    days = pd.date_range(months[0].start_time, months[-1].end_time.normalize(), freq="D")
    holiday_dates = set(get_holidays(range(months[0].year, months[-1].year + 1), country)["date"])

    calendar = pd.DataFrame({"date": days})
    calendar["month"] = calendar["date"].dt.to_period("M")
    calendar["is_holiday"] = calendar["date"].isin(holiday_dates)
    calendar["is_working_day"] = (calendar["date"].dt.weekday < 5) & ~calendar["is_holiday"]

    return (calendar.groupby("month")
            .agg(days_in_month=("date", "size"), holidays=("is_holiday", "sum"),
                 working_days=("is_working_day", "sum"))
            .reset_index())
