"""Tests for the Forecast page pieces (step 8.3): view helpers, fan chart and the page itself."""

from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.components.charts import forecast_fan_chart
from app.components.forecast_view import balance_risk, category_table, model_label
from src.utils.helpers import get_project_root


def _forecast(balance=(1000.0, 800.0, 1200.0)):
    series = pd.DataFrame({
        "series": ["income", "expenses", "expenses:Rent", "expenses:Pets", "expenses:Utilities"],
        "last_value": [3000.0, 2000.0, 900.0, 0.0, 120.4],
        "lower": [2800.0, 1800.0, 890.0, 10.0, 110.0],
        "predicted": [3000.0, 2100.0, 900.0, 10.0, 120.0],
        "upper": [3300.0, 2500.0, 950.0, 10.0, 130.0],
        "champion": ["lightgbm", "same_month_last_year", "last_month", "mean_3 (new series)", "mean_3"],
    })
    predicted, lower, upper = balance
    return SimpleNamespace(series=series, target_month=pd.Period("2026-10", "M"),
                           balance={"current": 100.0, "predicted": predicted, "lower": lower, "upper": upper},
                           get=lambda s: series.set_index("series").loc[s].to_dict())


def test_model_labels_are_plain_words():
    assert model_label("lightgbm") == "ML model"
    assert model_label("same_month_last_year") == "Same month last year"
    assert model_label("mean_3 (new series)") == "3-month average (new category)"


def test_category_table():
    table = category_table(_forecast())
    assert table["category"].tolist() == ["Rent", "Utilities", "Pets"]          # largest forecast first
    assert table.loc[table["category"] == "Utilities", "change"].item() == 0.0   # -0.4 rounds to 0, not -0
    assert str(table.loc[table["category"] == "Utilities", "change"].item()) == "0.0"
    assert table.loc[0, "range"] == "€890 – €950" and table.loc[0, "model"] == "Same as last month"


@pytest.mark.parametrize("balance, level", [((1000.0, 800.0, 1200.0), None),
                                            ((300.0, -150.0, 900.0), "warning"),
                                            ((-50.0, -600.0, 400.0), "error")])
def test_balance_risk(balance, level):
    risk = balance_risk(_forecast(balance))
    assert (risk[0] if risk else None) == level
    if risk:
        assert "October 2026" in risk[1]


def test_fan_chart_ends_at_the_forecast_range():
    monthly = pd.DataFrame({"month": pd.period_range("2026-04", periods=6, freq="M"),
                            "income": [3000.0] * 6, "expenses": [2000.0] * 6})
    fig = forecast_fan_chart(monthly, _forecast())
    fans = [t for t in fig.data if t.fill == "toself"]
    assert len(fans) == 2 and sorted(fans[0].y[1:3]) == [2800.0, 3300.0]
    assert forecast_fan_chart(monthly.head(0), _forecast()) is None


def test_page_asks_for_login():
    page = str(get_project_root() / "dashboard" / "pages" / "3_Forecast.py")
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception and [w.value for w in at.warning] == ["Please log in first."]
