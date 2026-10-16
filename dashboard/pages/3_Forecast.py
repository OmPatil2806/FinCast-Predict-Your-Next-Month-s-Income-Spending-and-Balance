"""Forecast page: next month's income, spending and end balance, with ranges, a fan chart and categories."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.auth.session import current_username, require_login  # noqa: E402
from app.components.cache import bump_cache, drift_reference, forecast as cached_forecast  # noqa: E402
from app.components.cards import format_euro  # noqa: E402
from app.components.charts import INCOME_COLOR, SPENDING_COLOR, accuracy_chart, forecast_fan_chart  # noqa: E402
from app.components.forecast_view import (  # noqa: E402
    ACCURACY_METRICS, accuracy_headline, accuracy_table, balance_risk, category_table, range_text,
)
from app.components.layout import empty_state, setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import get_transactions, save_forecast  # noqa: E402
from src.forecasting.predictor import NotEnoughHistory, complete_history  # noqa: E402
from src.monitoring.drift import check_drift  # noqa: E402
from src.monitoring.forecast_vs_actual import accuracy, summary, update_actuals  # noqa: E402
from src.preprocessing.aggregate_monthly import aggregate_monthly  # noqa: E402

TREND_MONTHS = 12

st.set_page_config(page_title="Forecast · FinCast", page_icon="💶", layout="wide")
setup_database()


def show_cards(forecast) -> None:
    income, spending, balance = forecast.get("income"), forecast.get("expenses"), forecast.balance
    left, middle, right = st.columns(3)

    def delta(value: float, versus: str = "last month") -> str:
        return f"{'-' if value < 0 else '+'}€{abs(value):,.0f} vs {versus}"

    def colour(value: float, normal: str = "normal") -> str:
        return "off" if round(value) == 0 else normal      # no green / red arrow for "no change"

    income_delta = income["predicted"] - income["last_value"]
    left.metric("Income", format_euro(income["predicted"], 0), delta(income_delta), delta_color=colour(income_delta),
                border=True, help="Most likely value; the range below holds the real value about 8 times in 10.")
    left.caption(f"Likely {range_text(income['lower'], income['upper'])}")
    spending_delta = spending["predicted"] - spending["last_value"]
    middle.metric("Spending", format_euro(spending["predicted"], 0), delta(spending_delta),
                  delta_color=colour(spending_delta, "inverse"), border=True,
                  help="Most likely value; the range below holds the real value about 8 times in 10.")
    middle.caption(f"Likely {range_text(spending['lower'], spending['upper'])}")
    right.metric("Balance at month end", format_euro(balance["predicted"], 0),
                 delta(balance["predicted"] - balance["current"], "now"), border=True,
                 help="Today's balance + forecast income − forecast spending. Its range pairs low income with "
                      "high spending (and the other way round), so it is deliberately wide.")
    right.caption(f"Likely {range_text(balance['lower'], balance['upper'])} · now {format_euro(balance['current'], 0)}")


def show_categories(forecast) -> None:
    st.subheader("Spending by category")
    st.dataframe(
        category_table(forecast), width="stretch", hide_index=True,
        column_config={
            "category": "Category",
            "last_value": st.column_config.NumberColumn("Last month (€)", format="%.0f"),
            "predicted": st.column_config.NumberColumn("Forecast (€)", format="%.0f"),
            "change": st.column_config.NumberColumn("Change (€)", format="%+.0f"),
            "range": "Likely range",
            "model": st.column_config.TextColumn("Forecast by", help="The method that was most accurate for this "
                                                 "category on past data: an ML model or a simple rule."),
        },
    )
    st.caption("Each category is forecast by whichever method was most accurate for it, separately from the "
               "total, so the categories do not add up exactly to the spending forecast above.")


def show_accuracy(user_id: int, forecast) -> None:
    """Step 9.2: how close past forecasts came to what really happened."""
    st.subheader("How accurate was I?")
    scored = accuracy(user_id)
    if scored.empty:
        st.caption(f"Nothing to check yet. Your {forecast.target_month.strftime('%B %Y')} forecast is compared with "
                   "what really happens once that month is over and its transactions are imported.")
        return
    st.caption("Each month FinCast stores its first forecast; once the month is over it is compared with your real "
               "numbers. The range is meant to hold the real value about 8 times in 10.")
    stats = summary(scored).set_index("metric")
    columns = st.columns(len(stats))
    for column, (metric, row) in zip(columns, stats.iterrows()):
        value, help_text = accuracy_headline(row)
        column.metric(ACCURACY_METRICS[metric], value, border=True)
        column.caption(help_text)

    choice = st.segmented_control("Show", list(ACCURACY_METRICS), format_func=ACCURACY_METRICS.get,
                                  default="expenses", key="accuracy_metric") or "expenses"
    color = {"income": INCOME_COLOR, "expenses": SPENDING_COLOR, "balance": INCOME_COLOR}[choice]
    figure = accuracy_chart(scored, choice, ACCURACY_METRICS[choice], color)
    if figure is not None:
        st.plotly_chart(figure, width="stretch")
    st.dataframe(accuracy_table(scored, choice), width="stretch", hide_index=True, column_config={
        "month": "Month",
        "forecast": st.column_config.NumberColumn("Forecast (€)", format="%.0f"),
        "range": "Likely range",
        "actual": st.column_config.NumberColumn("Actual (€)", format="%.0f"),
        "error": st.column_config.NumberColumn("Actual − forecast (€)", format="%+.0f"),
        "in_range": "In range"})


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("Forecast")

    transactions = get_transactions(user_id)
    if update_actuals(user_id):                   # score stored forecasts of finished months (step 9.1)
        bump_cache()
    try:
        with st.spinner("Forecasting next month…"):
            forecast = cached_forecast(user_id)
    except NotEnoughHistory as err:
        empty_state(str(err))
        return
    except FileNotFoundError:
        st.error("The forecasting models are missing. Run `python pipelines/run_training.py` once.")
        return

    saved = save_forecast(user_id, forecast)      # first forecast per month is kept (step 8.4)
    if saved["saved"]:
        bump_cache()
    st.caption(f"Hello, {current_username()} — your forecast for **{forecast.target_month.strftime("%B %Y")}**, based on "
               f"{forecast.history_months} complete months up to {forecast.origin_month.strftime("%B %Y")} "
               f"(model version {forecast.model_version}).")
    if saved["already_saved"] and not saved["saved"]:
        st.caption(f"Your first {forecast.target_month.strftime('%B %Y')} forecast is already stored; FinCast will "
                   "compare that one with what really happens once the month is over.")
    for note in forecast.notes:
        (st.info if forecast.cold_start else st.caption)(note)
    drift = check_drift(transactions, drift_reference())          # step 9.3
    if drift and drift["drifted"]:
        st.info(drift["message"], icon="🧭")

    show_cards(forecast)
    risk = balance_risk(forecast)
    if risk:
        getattr(st, risk[0])(risk[1])

    st.subheader("Income and spending: the last months and next month")
    monthly = aggregate_monthly(complete_history(transactions)).tail(TREND_MONTHS)
    st.plotly_chart(forecast_fan_chart(monthly, forecast), width="stretch")
    st.caption("Shaded fan: where next month will most likely land (10 % – 90 % range). Dotted line: the forecast.")

    show_categories(forecast)
    show_accuracy(user_id, forecast)


main()
show_footer()
