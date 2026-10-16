"""KPI values and chart builders (step 9.4)."""

import math

import pandas as pd

from app.components.cards import format_euro, kpi_values
from app.components.charts import (
    INCOME_COLOR, NEGATIVE_COLOR, POSITIVE_COLOR, SPENDING_COLOR, category_bar_chart, income_vs_spending_chart,
    monthly_savings_bar_chart,
)


def _monthly(n=3):
    return pd.DataFrame({"month": pd.period_range("2026-07", periods=n, freq="M"),
                         "income": [3000.0, 3000.0, 2000.0][:n], "expenses": [2500.0, 2000.0, 2600.0][:n],
                         "savings": [500.0, 1000.0, -600.0][:n], "savings_rate": [0.1667, 0.3333, -0.3][:n]})


def test_format_euro():
    assert format_euro(1234.5) == "€1,234.50" and format_euro(-12, 0) == "−€12"
    assert format_euro(float("nan")) == "–" and format_euro(None) == "–"


def test_kpi_values():
    assert kpi_values(_monthly(0)) is None
    one = kpi_values(_monthly(1))
    assert one["income"] == 3000 and one["income_delta"] is None
    three = kpi_values(_monthly(3))
    assert three["savings"] == -600 and three["savings_delta"] == -1600 and three["expenses_delta"] == 600
    assert math.isclose(three["savings_rate_delta"], -0.6333)


def test_chart_builders():
    assert income_vs_spending_chart(_monthly(0)) is None
    lines = income_vs_spending_chart(_monthly())
    assert [t.line.color for t in lines.data] == [INCOME_COLOR, SPENDING_COLOR]
    bars = monthly_savings_bar_chart(_monthly())
    assert list(bars.data[0].marker.color) == [POSITIVE_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR]
    categories = pd.DataFrame({"category": list("ABCDEFGHIJ"), "expenses": [100.0 - i for i in range(10)]})
    chart = category_bar_chart(categories, top_n=8)
    assert len(chart.data[0].y) == 8 and chart.data[0].y[0] == "Other"          # 3 smallest folded, drawn last
    assert chart.data[0].x[0] == sum(100.0 - i for i in range(7, 10))
