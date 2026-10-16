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

Loads no MLflow and no training code.
"""

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
                               opening_balance: float = 0.0, as_of=None, user_id="user") -> Forecast:
    """Forecast for the month after the last complete month in `transactions` (date, amount, category)."""
    forecasters = forecasters or load_forecasters()
    notes: list[str] = []
    history = complete_history(transactions, as_of)
    if history.empty:
        raise ValueError("no complete month of transactions yet")
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
    order = {INCOME: 0, EXPENSES: 1}
    series = series.sort_values("series", key=lambda s: s.map(order).fillna(2).astype(str) + s,
                                ignore_index=True)

    current = float(monthly.loc[(monthly["category"] == TOTAL) & (monthly["month"] == origin), "balance_end"].iloc[0])
    income, spend = series.set_index("series").loc[INCOME], series.set_index("series").loc[EXPENSES]
    balance = {"current": current,
               "predicted": round(float(current + income["predicted"] - spend["predicted"]), 2),
               "lower": round(float(current + income["lower"] - spend["upper"]), 2),
               "upper": round(float(current + income["upper"] - spend["lower"]), 2)}
    history_months = int(rows["history_months"].iloc[0]) if "history_months" in rows else int(monthly["month"].nunique())
    return Forecast(user_id=user_id, origin_month=origin, target_month=origin + 1, history_months=history_months,
                    series=series, balance=balance, model_version=forecasters["income"].version, notes=notes)


def forecast_user(user_id: int, db_path: str | Path | None = None, as_of=None,
                  forecasters: dict | None = None) -> Forecast:
    """Forecast for a logged-in user, from their transactions and opening balance in the app database."""
    from app.database.crud import get_profile, get_transactions
    transactions = get_transactions(user_id, db_path=db_path)
    if transactions.empty:
        raise ValueError("no transactions yet - import a bank CSV first")
    profile = get_profile(user_id, db_path=db_path) or {}
    return forecast_from_transactions(transactions, forecasters, profile.get("opening_balance") or 0.0,
                                      as_of, user_id)
