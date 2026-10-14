import random

import numpy as np

import pandas as pd

from src.utils.helpers import (
    category_names, get_holidays, get_project_root, holidays_per_month, load_categories, load_config,
    resolve_path, set_seed,
)


def test_project_root_contains_config():
    root = get_project_root()
    assert (root / "config" / "config.yaml").is_file()
    assert (root / "src").is_dir()


def test_load_config_returns_settings():
    config = load_config()
    assert config["project"]["seed"] == 42
    assert config["locale"] == {"country": "DE", "currency": "EUR"}
    assert config["forecast"]["quantiles"] == [0.1, 0.5, 0.9]


def test_resolve_path_is_inside_project():
    path = resolve_path(load_config()["paths"]["data_raw"])
    assert path.is_absolute()
    assert path == get_project_root() / "data" / "raw"


def test_set_seed_makes_results_repeatable():
    set_seed(42)
    first = (random.random(), np.random.rand())
    set_seed(42)
    second = (random.random(), np.random.rand())
    assert first == second


def test_german_holidays_2025():
    table = get_holidays(2025, country="DE")
    assert len(table) == 9                      # nationwide holidays in Germany
    assert pd.Timestamp("2025-12-25") in set(table["date"])
    assert pd.Timestamp("2025-10-03") in set(table["date"])   # Tag der Deutschen Einheit


def test_holidays_per_month():
    calendar = holidays_per_month("2025-01", "2025-12", country="DE").set_index("month")
    assert len(calendar) == 12
    assert calendar.loc[pd.Period("2025-12"), "holidays"] == 2
    assert calendar.loc[pd.Period("2025-04"), "holidays"] == 2   # Good Friday, Easter Monday
    assert calendar.loc[pd.Period("2025-08"), "holidays"] == 0
    assert calendar.loc[pd.Period("2025-02"), "days_in_month"] == 28
    # December 2025: 23 weekdays minus Christmas Day and Boxing Day (Thu, Fri) = 21
    assert calendar.loc[pd.Period("2025-12"), "working_days"] == 21
    assert calendar["holidays"].sum() == 9


def test_holiday_country_comes_from_config():
    assert len(get_holidays(2025)) == len(get_holidays(2025, country=load_config()["locale"]["country"]))


def test_categories_file():
    rules = load_categories()
    names = category_names()
    assert len(names) == 12 and len(set(names)) == 12
    assert set(names) == {"Income", "Rent", "Savings", "Subscriptions", "Utilities", "Transport", "Groceries",
                          "Dining", "Health", "Leisure", "Shopping", "Other"}
    assert rules["fallback"] in names
    keywords = [k for c in rules["categories"] for k in c["keywords"]]
    assert all(k == k.lower() for k in keywords)          # rules match on lower-cased text
    assert len(keywords) == len(set(keywords))            # no keyword in two categories
    assert all(c.get("sign") in (None, "income", "expense") for c in rules["categories"])
