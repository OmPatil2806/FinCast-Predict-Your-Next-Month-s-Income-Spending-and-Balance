"""Metrics checked against values worked out by hand."""

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import coverage, mae, pinball_loss, rmse, smape, summarize

ACTUAL = [100.0, 200.0, 0.0, 50.0]
PREDICTED = [110.0, 180.0, 0.0, 80.0]          # errors: -10, +20, 0, -30


def test_mae_and_rmse():
    assert mae(ACTUAL, PREDICTED) == pytest.approx((10 + 20 + 0 + 30) / 4)              # 15
    assert rmse(ACTUAL, PREDICTED) == pytest.approx(np.sqrt((100 + 400 + 0 + 900) / 4))  # 18.71


def test_smape_handles_zero_months():
    # 2|a-p|/(|a|+|p|): 20/210, 40/380, 0 (both zero), 60/130
    expected = 100 * (20 / 210 + 40 / 380 + 0 + 60 / 130) / 4
    assert smape(ACTUAL, PREDICTED) == pytest.approx(expected)
    assert smape([0, 0], [0, 0]) == 0.0
    assert smape([0], [10]) == pytest.approx(200.0)                                      # worst case


def test_coverage():
    assert coverage([5, 10, 15, 20], [0, 0, 0, 0], [10, 10, 10, 10]) == 0.5               # 5 and 10 inside (edges count)


def test_pinball_loss():
    # quantile 0.9: under-forecast by 10 costs 9, over-forecast by 10 costs 1
    assert pinball_loss([100], [90], 0.9) == pytest.approx(9.0)
    assert pinball_loss([100], [110], 0.9) == pytest.approx(1.0)
    assert pinball_loss(ACTUAL, PREDICTED, 0.5) == pytest.approx(mae(ACTUAL, PREDICTED) / 2)   # median = MAE / 2
    with pytest.raises(ValueError):
        pinball_loss([1], [1], 1.5)


def test_lengths_must_match():
    with pytest.raises(ValueError):
        mae([1, 2], [1])


def test_summarize_per_group():
    df = pd.DataFrame({"series": ["income", "income", "expenses", "expenses"],
                       "actual": ACTUAL, "predicted": PREDICTED,
                       "lower": [90, 150, -5, 60], "upper": [120, 210, 5, 90]})
    table = summarize(df, "series", lower="lower", upper="upper").set_index("series")
    assert table.loc["income", "mae"] == pytest.approx(15.0)
    assert table.loc["expenses", "mae"] == pytest.approx(15.0)
    assert table.loc["income", "coverage"] == 1.0
    assert table.loc["expenses", "coverage"] == 0.5                                       # 50 is below 60
    assert table.loc["income", "n"] == 2
