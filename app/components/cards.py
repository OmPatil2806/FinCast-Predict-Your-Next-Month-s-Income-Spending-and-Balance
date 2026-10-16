"""KPI cards (stat tiles) and the shared euro formatter."""

import math

import pandas as pd
import streamlit as st


def format_euro(value: float, decimals: int = 2) -> str:
    """'€1,234.56', '−€12.00' (true minus sign), '–' for missing values."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    sign = "−" if value < 0 else ""
    return f"{sign}€{abs(value):,.{decimals}f}"


def _delta_euro(value: float) -> str:
    # st.metric reads the leading '-' to choose the arrow, so use an ASCII minus here
    return f"{'-' if value < 0 else '+'}€{abs(value):,.2f} vs last month"


def kpi_values(monthly: pd.DataFrame) -> dict | None:
    """Latest month's income, spending, savings and savings rate, with changes vs the month before.

    Returns None when there is no data. Deltas are None when there is only one month.
    """
    if monthly.empty:
        return None
    latest = monthly.iloc[-1]
    previous = monthly.iloc[-2] if len(monthly) > 1 else None
    values = {"month": latest["month"]}
    for key in ("income", "expenses", "savings", "savings_rate"):
        values[key] = latest[key]
        if previous is None or pd.isna(latest[key]) or pd.isna(previous[key]):
            values[f"{key}_delta"] = None
        else:
            values[f"{key}_delta"] = latest[key] - previous[key]
    return values


def kpi_row(monthly: pd.DataFrame) -> None:
    """Four stat tiles for the latest month: Income, Spending, Savings, Savings rate."""
    values = kpi_values(monthly)
    if values is None:
        return
    st.caption(f"Latest month: {values['month'].strftime('%B %Y')}")
    income, spending, savings, rate = st.columns(4)

    def delta(key: str) -> str | None:
        d = values[f"{key}_delta"]
        return None if d is None else _delta_euro(d)

    def colour(key: str, normal: str = "normal") -> str:
        d = values[f"{key}_delta"]
        return "off" if d is not None and round(d, 2) == 0 else normal   # no arrow colour for "no change"

    income.metric("Income", format_euro(values["income"]), delta("income"), delta_color=colour("income"), border=True)
    # less spending is good, so the colour of the arrow is inverted
    spending.metric("Spending", format_euro(values["expenses"]), delta("expenses"),
                    delta_color=colour("expenses", "inverse"), border=True)
    savings.metric("Savings", format_euro(values["savings"]), delta("savings"), delta_color=colour("savings"),
                   border=True)

    rate_value = "–" if pd.isna(values["savings_rate"]) else f"{values['savings_rate']:.1%}"
    rate_delta = values["savings_rate_delta"]
    rate.metric(
        "Savings rate", rate_value,
        None if rate_delta is None else f"{rate_delta * 100:+.1f} pp vs last month",
        border=True,
    )
