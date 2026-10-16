"""Step 8.1: next month's forecast for one user, from their own transactions.

    forecast_user(user_id)                         reads the app database (SQLite)
    forecast_from_transactions(transactions, ...)  the same from a DataFrame (tests, demo data)

Exactly the training path, so the app and the models cannot drift apart:
    transactions -> aggregate_user_month_category (5.7) -> build_features (6.x)
    -> the exported champion bundles (7.9): models/forecaster_income.joblib, forecaster_spend.joblib

Only complete months are used: with as_of in October, history ends in September at the latest and the
forecast is for the month after the user's last complete month. Each series gets the champion of step
7.8 (LightGBM or a baseline) with the LightGBM 10-90 % range. A spending category the models never saw
(for example "Uncategorised") falls back to its 3-month average with no range.

The end balance is this month's balance plus forecast income minus forecast spending; its range combines
the low income with the high spending end (and the other way round), so it is deliberately wide.
Total spending and the categories have their own champions, so the categories need not add up to the total.

Short histories (step 8.2, settings in config.yaml -> forecast):
    fewer than min_history_months (3) complete months -> NotEnoughHistory, with a friendly message
    fewer than cold_start_months (6)                   -> forecast with a note; ranges widened by the
                                                          factor measured in models/cold_start.json
    fewer than 12                                       -> a note: no "same month last year" yet, so
                                                          spending forecasts are clearly less accurate

Loads no MLflow and no training code.
"""

import json

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import EXPENSES, INCOME, build_features
from src.forecasting.model import BASELINES, ChampionForecaster
from src.preprocessing.aggregate_monthly import TOTAL, aggregate_user_month_category
from src.utils.helpers import load_config, resolve_path

FALLBACK = "mean_3"
FAMILIES = ("income", "spend")
FULL_YEAR = 12


class NotEnoughHistory(ValueError):
    """Too few complete months for a forecast worth showing."""

    def __init__(self, months: int, needed: int):
        self.months, self.needed = months, needed
        more = needed - months
        super().__init__(f"FinCast needs at least {needed} complete months of transactions to forecast; "
                         f"you have {months}. Import {more} more month{'s' if more != 1 else ''} "
                         f"(or wait until the current month is over).")


def load_cold_start(models_dir: str | Path | None = None) -> dict:
    """Range widening for short histories, measured in step 8.2 (empty -> no widening)."""
    config = load_config()
    models_dir = Path(models_dir) if models_dir else resolve_path(config["paths"]["models"])
    path = models_dir / config["forecast"]["cold_start_file"]
    return json.loads(path.read_text()) if path.exists() else {}


@dataclass
class Forecast:
    user_id: object
    origin_month: pd.Period            # last complete month used
    target_month: pd.Period            # the month being forecast
    history_months: int
    series: pd.DataFrame               # series, last_value, lower, predicted, upper, champion
    balance: dict                      # current, lower, predicted, upper
    model_version: int | None = None
    notes: list = field(default_factory=list)
    cold_start: bool = False           # fewer than cold_start_months: ranges widened, shown with a note

    def get(self, series: str) -> dict:
        """One series as a dict, e.g. forecast.get("income")["predicted"]."""
        return self.series.set_index("series").loc[series].to_dict()


def load_forecasters(models_dir: str | Path | None = None) -> dict[str, ChampionForecaster]:
    """The exported champion per family (cache this in the app: st.cache_resource)."""
    config = load_config()
    models_dir = Path(models_dir) if models_dir else resolve_path(config["paths"]["models"])
    pattern = config["forecast"]["model_file"]
    missing = [f for f in FAMILIES if not (models_dir / pattern.format(family=f)).exists()]
    if missing:
        raise FileNotFoundError(f"no exported forecaster for {missing} in {models_dir} - "
                                "run `python pipelines/run_training.py`")
    return {family: joblib.load(models_dir / pattern.format(family=family)) for family in FAMILIES}


def complete_history(transactions: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Drop the current (unfinished) month: keep transactions before the month of as_of (default today)."""
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()
    dates = pd.to_datetime(transactions["date"])
    return transactions[dates < as_of.to_period("M").to_timestamp()]


def _origin_rows(table: pd.DataFrame, feature_columns: list[str], notes: list) -> pd.DataFrame:
    rows = table[table["month"] == table["month"].max()].reset_index(drop=True)
    absent = [c for c in feature_columns if c not in rows]
    if absent:                                   # e.g. a model trained with macro data, no ECB cache here
        notes.append(f"inputs not available, treated as unknown: {', '.join(absent)}")
        rows = rows.assign(**{c: np.nan for c in absent})
    return rows


def forecast_from_transactions(transactions: pd.DataFrame, forecasters: dict | None = None,
                               opening_balance: float = 0.0, as_of=None, user_id="user",
                               cold_start: dict | None = None) -> Forecast:
    """Forecast for the month after the last complete month in `transactions` (date, amount, category).

    Raises NotEnoughHistory when there are fewer than forecast.min_history_months complete months.
    """
    forecasters = forecasters or load_forecasters()
    settings = load_config()["forecast"]
    notes: list[str] = []
    history = complete_history(transactions, as_of)
    months = int(pd.to_datetime(history["date"]).dt.to_period("M").nunique()) if not history.empty else 0
    if months == 0:
        raise NotEnoughHistory(0, settings["min_history_months"])
    first, last = pd.to_datetime(history["date"]).dt.to_period("M").agg(["min", "max"])
    months = (last - first).n + 1                         # months in between without transactions count too
    if months < settings["min_history_months"]:
        raise NotEnoughHistory(months, settings["min_history_months"])
    history = history.assign(user_id=user_id)

    monthly = aggregate_user_month_category(history, opening_balances=float(opening_balance))
    table, info = build_features(monthly)
    rows = _origin_rows(table, forecasters["income"].feature_columns, notes)
    origin = rows["month"].iloc[0]

    parts = []
    for family, model in forecasters.items():
        part = rows[(rows["series"] == INCOME) == (family == "income")]
        known = part["series"].isin(model.models[0.5].series_levels)
        if known.any():
            parts.append(pd.concat([part.loc[known, ["series", "value"]].reset_index(drop=True),
                                    model.predict(part[known]).reset_index(drop=True)], axis=1))
        if (~known).any():
            unknown = part[~known]
            point = BASELINES[FALLBACK](unknown).clip(lower=0).to_numpy()
            parts.append(pd.DataFrame({"series": unknown["series"].to_numpy(), "value": unknown["value"].to_numpy(),
                                       "champion": f"{FALLBACK} (new series)", "lower": point,
                                       "predicted": point, "upper": point}))
            notes.append(f"no model for {', '.join(unknown['series'])}: 3-month average, no range")
    series = (pd.concat(parts, ignore_index=True).rename(columns={"value": "last_value"})
              [["series", "last_value", "lower", "predicted", "upper", "champion"]])
    is_cold = months < settings["cold_start_months"]
    if is_cold:
        cold_start = load_cold_start() if cold_start is None else cold_start
        factors = cold_start.get("width_factor", {})
        is_income = series["series"] == INCOME
        factor = np.where(is_income, factors.get("income", 1.0), factors.get("spend", 1.0))
        series["lower"] = np.clip(series["predicted"] - factor * (series["predicted"] - series["lower"]), 0, None)
        series["upper"] = series["predicted"] + factor * (series["upper"] - series["predicted"])
        notes.append(f"Based on only {months} months of history: forecasts are less accurate and the ranges "
                     f"are wider. They improve once you have {settings['cold_start_months']} or more months.")
    elif months < FULL_YEAR:
        notes.append(f"{months} months of history: no 'same month last year' yet, so spending forecasts are "
                     "less accurate until you have 12 months.")

    order = {INCOME: 0, EXPENSES: 1}
    series = series.sort_values("series", key=lambda s: s.map(order).fillna(2).astype(str) + s,
                                ignore_index=True)

    current = float(monthly.loc[(monthly["category"] == TOTAL) & (monthly["month"] == origin), "balance_end"].iloc[0])
    income, spend = series.set_index("series").loc[INCOME], series.set_index("series").loc[EXPENSES]
    balance = {"current": current,
               "predicted": round(float(current + income["predicted"] - spend["predicted"]), 2),
               "lower": round(float(current + income["lower"] - spend["upper"]), 2),
               "upper": round(float(current + income["upper"] - spend["lower"]), 2)}
    return Forecast(user_id=user_id, origin_month=origin, target_month=origin + 1, history_months=months,
                    series=series, balance=balance, model_version=forecasters["income"].version, notes=notes,
                    cold_start=is_cold)


def forecast_user(user_id: int, db_path: str | Path | None = None, as_of=None,
                  forecasters: dict | None = None) -> Forecast:
    """Forecast for a logged-in user, from their transactions and opening balance in the app database."""
    from app.database.crud import get_profile, get_transactions
    transactions = get_transactions(user_id, db_path=db_path)
    if transactions.empty:
        raise NotEnoughHistory(0, load_config()["forecast"]["min_history_months"])
    profile = get_profile(user_id, db_path=db_path) or {}
    return forecast_from_transactions(transactions, forecasters, profile.get("opening_balance") or 0.0,
                                      as_of, user_id)
