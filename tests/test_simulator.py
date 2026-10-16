"""Tests for the what-if simulator and page (step 8.5)."""

from dataclasses import dataclass

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.components.charts import what_if_chart
from src.forecasting.simulator import Scenario, compare, simulate
from src.utils.helpers import get_project_root


@dataclass
class FakeForecast:
    series: pd.DataFrame
    balance: dict
    target_month: pd.Period = pd.Period("2026-10", "M")

    def get(self, s):
        return self.series.set_index("series").loc[s].to_dict()


def _forecast():
    series = pd.DataFrame({"series": ["income", "expenses", "expenses:Dining", "expenses:Rent"],
                           "lower": [2700.0, 1800.0, 60.0, 990.0], "predicted": [3000.0, 2000.0, 100.0, 1000.0],
                           "upper": [3200.0, 2400.0, 150.0, 1010.0], "champion": "lightgbm"})
    return FakeForecast(series, {"current": 500.0, "predicted": 1500.0, "lower": 800.0, "upper": 1900.0})


def test_no_change_gives_the_same_forecast():
    f = _forecast()
    s = simulate(f, Scenario())
    pd.testing.assert_frame_equal(s.series, f.series)
    assert s.balance == f.balance and compare(f, s) == {"income": 0, "expenses": 0, "balance": 0}


def test_changes_flow_into_spending_and_balance():
    f = _forecast()
    s = simulate(f, Scenario(income_change=-0.10, category_changes={"Dining": -0.5, "Unknown": 0.3}, one_off=250))
    get = lambda name: s.series.set_index("series").loc[name]
    assert get("income")[["lower", "predicted", "upper"]].tolist() == [2430.0, 2700.0, 2880.0]
    assert get("expenses:Dining")["predicted"] == 50.0 and get("expenses:Rent")["predicted"] == 1000.0
    assert get("expenses")["predicted"] == 2000.0 - 50 + 250                     # categories shift the total
    assert compare(f, s) == pytest.approx({"income": -300.0, "expenses": 200.0, "balance": -500.0})
    assert s.balance["predicted"] == 500 + 2700 - 2200
    assert s.balance["lower"] == 500 + 2430 - (2400 + 200)                       # low income with high spending
    assert f.series.set_index("series").loc["income", "predicted"] == 3000.0     # original untouched


def test_never_below_zero():
    s = simulate(_forecast(), Scenario(income_change=-2.0, category_changes={"Dining": -1.0}, one_off=-5000))
    assert (s.series[["lower", "predicted", "upper"]] >= 0).all().all()
    assert s.series.set_index("series").loc["income", "predicted"] == 0.0


def test_what_if_chart_shows_both():
    fig = what_if_chart(_forecast(), simulate(_forecast(), Scenario(one_off=100)))
    assert [t.name for t in fig.data] == ["Forecast", "Your scenario"]
    assert fig.data[1].y[1] == 2100.0


def test_page_asks_for_login():
    page = str(get_project_root() / "dashboard" / "pages" / "4_What_If.py")
    at = AppTest.from_file(page, default_timeout=60).run()
    assert not at.exception and [w.value for w in at.warning] == ["Please log in first."]
