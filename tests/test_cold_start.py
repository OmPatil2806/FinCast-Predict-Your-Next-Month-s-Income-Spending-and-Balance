"""Tests for the cold-start calibration (step 8.2)."""

import numpy as np
import pandas as pd
import pytest

from src.training.calibrate_cold_start import calibrate, choose_factor, widen_around


def _predictions(rng, n, spread, history, origin, family="income"):
    actual = rng.normal(100, 20, n)
    return pd.DataFrame({"family": family, "history": history, "origin": origin, "actual": actual,
                         "predicted": 100.0, "lower": 100.0 - spread, "upper": 100.0 + spread})


def test_widen_around_keeps_the_point_and_floors_at_zero():
    p = pd.DataFrame({"predicted": [100.0, 10.0], "lower": [90.0, 2.0], "upper": [120.0, 15.0]})
    w = widen_around(p, 2.0)
    assert w["lower"].tolist() == [80.0, 0.0] and w["upper"].tolist() == [140.0, 20.0]
    assert w["predicted"].equals(p["predicted"])


def test_choose_factor_reaches_the_target():
    rng = np.random.default_rng(0)
    narrow = _predictions(rng, 4000, spread=15, history=3, origin="a")      # ~55 % coverage
    factor = choose_factor(narrow, 0.80)
    w = widen_around(narrow, factor)
    assert factor > 1.5 and ((w["actual"] >= w["lower"]) & (w["actual"] <= w["upper"])).mean() >= 0.80
    assert choose_factor(_predictions(rng, 4000, spread=40, history=3, origin="a"), 0.80) == 1.0


def test_calibrate_picks_on_early_origins_and_checks_on_late_ones():
    rng = np.random.default_rng(1)
    parts = [_predictions(rng, 1500, spread, h, origin, family)
             for family in ("income", "spend") for origin in ("o1", "o2", "o3", "o4")
             for h, spread in ((3, 15), (12, 26))]
    result = calibrate(pd.concat(parts, ignore_index=True), cold_start_months=6)
    assert result["pick_origins"] == ["o1", "o2"] and result["check_origins"] == ["o3", "o4"]
    table = result["by_history"].set_index(["family", "history"])
    assert table.loc[("income", 3), "factor"] > 1 and table.loc[("income", 12), "factor"] == 1.0
    assert table.loc[("income", 3), "check_coverage_widened"] == pytest.approx(0.80, abs=0.04)
