"""What-If page: change next month's income or spending and see the effect on the end balance at once."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.auth.session import require_login  # noqa: E402
from app.components.cache import forecast as cached_forecast  # noqa: E402
from app.components.cards import format_euro  # noqa: E402
from app.components.charts import what_if_chart  # noqa: E402
from app.components.forecast_view import balance_risk  # noqa: E402
from app.components.layout import empty_state, setup_database, show_footer, show_sidebar  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.forecasting.predictor import NotEnoughHistory  # noqa: E402
from src.forecasting.simulator import CATEGORY_PREFIX, Scenario, compare, simulate  # noqa: E402

MAIN_SLIDERS = 6          # the largest categories get a slider up front, the rest sit in an expander

st.set_page_config(page_title="What-If · FinCast", page_icon="💶", layout="wide")
setup_database()


def reset() -> None:
    for key in [k for k in st.session_state if str(k).startswith("whatif_")]:
        del st.session_state[key]


def controls(forecast) -> Scenario:
    categories = (forecast.series[forecast.series["series"].str.startswith(CATEGORY_PREFIX)]
                  .sort_values("predicted", ascending=False))
    names = [s.removeprefix(CATEGORY_PREFIX) for s in categories["series"]]
    predicted = dict(zip(names, categories["predicted"]))

    st.subheader("Change next month")
    income = st.slider("Income", -50, 50, 0, 5, format="%+d%%", key="whatif_income",
                       help="E.g. −20 % for fewer freelance hours, +5 % for a raise.")
    changes = {}

    def category_slider(name: str) -> None:
        changes[name] = st.slider(f"{name} (forecast {format_euro(predicted[name], 0)})", -100, 100, 0, 5,
                                  format="%+d%%", key=f"whatif_cat_{name}") / 100

    for name in names[:MAIN_SLIDERS]:
        category_slider(name)
    if len(names) > MAIN_SLIDERS:
        with st.expander(f"{len(names) - MAIN_SLIDERS} more categories"):
            for name in names[MAIN_SLIDERS:]:
                category_slider(name)
    one_off = st.number_input("One-off cost (€)", min_value=-50_000.0, max_value=50_000.0, value=0.0, step=50.0,
                              key="whatif_one_off", help="A holiday, a repair, a new laptop. Negative = extra money in.")
    st.button("Reset", on_click=reset)
    return Scenario(income_change=income / 100, category_changes=changes, one_off=one_off)


def results(forecast, scenario: Scenario) -> None:
    after = simulate(forecast, scenario)
    change = compare(forecast, after)
    st.subheader(f"{forecast.target_month.strftime('%B %Y')}: forecast vs your scenario")
    left, middle, right = st.columns(3)

    def delta(value: float):
        return None if round(value) == 0 else f"{'-' if value < 0 else '+'}€{abs(value):,.0f}"

    left.metric("Income", format_euro(after.get("income")["predicted"], 0), delta(change["income"]), border=True)
    middle.metric("Spending", format_euro(after.get("expenses")["predicted"], 0), delta(change["expenses"]),
                  delta_color="inverse", border=True)
    right.metric("Balance at month end", format_euro(after.balance["predicted"], 0), delta(change["balance"]),
                 border=True)
    right.caption(f"Likely {format_euro(after.balance['lower'], 0)} – {format_euro(after.balance['upper'], 0)}")

    risk = balance_risk(after)
    if risk:
        getattr(st, risk[0])(risk[1])
    st.plotly_chart(what_if_chart(forecast, after), width="stretch")

    changed = [name for name, c in scenario.category_changes.items() if c]
    if changed:
        keys = [CATEGORY_PREFIX + n for n in changed]
        table = pd.DataFrame({
            "category": changed,
            "forecast": forecast.series.set_index("series").loc[keys, "predicted"].to_numpy(),
            "scenario": after.series.set_index("series").loc[keys, "predicted"].to_numpy()})
        table["difference"] = table["scenario"] - table["forecast"]
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "category": "Category",
            "forecast": st.column_config.NumberColumn("Forecast (€)", format="%.0f"),
            "scenario": st.column_config.NumberColumn("Scenario (€)", format="%.0f"),
            "difference": st.column_config.NumberColumn("Difference (€)", format="%+.0f")})
    elif scenario.is_empty():
        st.caption("Move a slider to see how next month would change.")


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("What-If")
    st.caption("Try changes to next month and see your end balance move. Nothing here is saved.")
    try:
        with st.spinner("Forecasting next month…"):
            forecast = cached_forecast(user_id)        # cached: sliders only re-run the cheap simulation
    except NotEnoughHistory as err:
        empty_state(str(err))
        return
    except FileNotFoundError as err:
        get_logger("app.what_if").error("Forecast models missing: %s", err)
        st.error("The forecasting models are missing. Run `python pipelines/run_training.py` once.")
        return

    left, right = st.columns([1, 2], gap="large")
    with left:
        scenario = controls(forecast)
    with right:
        results(forecast, scenario)


main()
show_footer()
