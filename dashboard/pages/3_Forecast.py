"""Forecast page: next month's income, spending and end balance, with ranges, a fan chart and categories."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from app.auth.session import current_username, require_login  # noqa: E402
from app.components.cards import format_euro  # noqa: E402
from app.components.charts import forecast_fan_chart  # noqa: E402
from app.components.forecast_view import balance_risk, category_table, range_text  # noqa: E402
from app.components.layout import setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import get_transactions, save_forecast  # noqa: E402
from src.forecasting.predictor import (  # noqa: E402
    NotEnoughHistory, complete_history, forecast_user, load_forecasters,
)
from src.preprocessing.aggregate_monthly import aggregate_monthly  # noqa: E402

TREND_MONTHS = 12

st.set_page_config(page_title="Forecast · FinCast", page_icon="💶", layout="wide")
setup_database()


@st.cache_resource
def forecasters():
    return load_forecasters()


def link_to_import() -> None:
    try:
        st.page_link("pages/2_Import_Transactions.py", label="Go to Import transactions", icon="📥")
    except StreamlitPageNotFoundError:   # page run on its own, outside the multipage app
        st.markdown("Open **Import Transactions** in the sidebar.")


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


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("Forecast")

    transactions = get_transactions(user_id)
    try:
        with st.spinner("Forecasting next month…"):
            forecast = forecast_user(user_id, forecasters=forecasters())
    except NotEnoughHistory as err:
        st.info(str(err))
        link_to_import()
        return
    except FileNotFoundError:
        st.error("The forecasting models are missing. Run `python pipelines/run_training.py` once.")
        return

    saved = save_forecast(user_id, forecast)      # first forecast per month is kept (step 8.4)
    st.caption(f"Hello, {current_username()} — your forecast for **{forecast.target_month.strftime("%B %Y")}**, based on "
               f"{forecast.history_months} complete months up to {forecast.origin_month.strftime("%B %Y")} "
               f"(model version {forecast.model_version}).")
    if saved["already_saved"] and not saved["saved"]:
        st.caption(f"Your first {forecast.target_month.strftime('%B %Y')} forecast is already stored; FinCast will "
                   "compare that one with what really happens once the month is over.")
    for note in forecast.notes:
        (st.info if forecast.cold_start else st.caption)(note)

    show_cards(forecast)
    risk = balance_risk(forecast)
    if risk:
        getattr(st, risk[0])(risk[1])

    st.subheader("Income and spending: the last months and next month")
    monthly = aggregate_monthly(complete_history(transactions)).tail(TREND_MONTHS)
    st.plotly_chart(forecast_fan_chart(monthly, forecast), width="stretch")
    st.caption("Shaded fan: where next month will most likely land (10 % – 90 % range). Dotted line: the forecast.")

    show_categories(forecast)


main()
show_footer()
