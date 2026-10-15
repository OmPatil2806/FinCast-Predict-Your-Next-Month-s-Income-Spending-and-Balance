"""Tests for the anomaly features, planting and detector (step 7.10)."""

import sys
import subprocess

import numpy as np
import pandas as pd
import pytest

from src.features.anomaly_features import MODEL_FEATURES, NEVER, build_anomaly_features, rule_flags
from src.ingestion.synthetic_generator import generate_dataset
from src.training.train_anomaly import label, plant_anomalies, scores_report, train_final


def _tx(rows):
    return pd.DataFrame(rows, columns=["user_id", "date", "description", "amount", "category"])


def test_features_use_only_earlier_transactions():
    rows = [("u1", f"2025-0{m}-05", "Netflix.com", -12.99, "Subscriptions") for m in range(1, 8)]
    base = build_anomaly_features(_tx(rows))
    later = build_anomaly_features(_tx(rows + [("u1", "2025-09-05", "Netflix.com", -999.0, "Subscriptions")]))
    pd.testing.assert_frame_equal(base[MODEL_FEATURES], later[MODEL_FEATURES].iloc[:len(base)])


def test_price_rise_new_merchant_and_duplicate_stand_out():
    rows = [("u1", f"2025-0{m}-05", "Netflix.com", -12.99, "Subscriptions") for m in range(1, 8)]
    rows += [("u1", f"2025-0{m}-10", "REWE Markt", -40.0 - m, "Groceries") for m in range(1, 8)]
    rows += [("u1", "2025-08-05", "Netflix.com", -17.99, "Subscriptions"),       # price rise
             ("u1", "2025-08-12", "Zahnarzt Praxis Dr. Klein", -700.0, "Health"),  # big, new merchant
             ("u1", "2025-08-20", "REWE Markt", -55.0, "Groceries"),
             ("u1", "2025-08-21", "REWE Markt", -55.0, "Groceries")]               # duplicate
    f = build_anomaly_features(_tx(rows)).set_index("description", append=True)
    netflix = f.xs("Netflix.com", level=1)
    assert netflix["merchant_z"].iloc[-1] >= 3 and (netflix["merchant_z"].iloc[:-1] < 3).all()
    dentist = f.xs("Zahnarzt Praxis Dr. Klein", level=1).iloc[0]
    assert dentist["merchant_seen"] == 0 and dentist["days_since_merchant"] == NEVER and dentist["new_merchant_z"] > 2
    rewe = f.xs("REWE Markt", level=1)
    assert rewe["days_since_same"].iloc[-1] == 1 and (rewe["days_since_same"].iloc[:-1] > 3).all()
    flags = rule_flags(f.reset_index(level=1))
    # flagged: the price rise, the first EUR 55 at REWE (usual 41-47) and its duplicate. The dentist is NOT:
    # a first payment in a new category has no history to compare with (known gap of the rule).
    assert flags.tolist()[-4:] == [True, False, True, True] and flags.sum() == 3


def test_planting_only_touches_the_window():
    transactions, _ = generate_dataset(n_users=4, months=12, start="2024-01-01", seed=2)
    planted = plant_anomalies(transactions, "2024-07-01", "2024-12-31", n_duplicates=20, n_price_rises=10, seed=1)
    assert (planted["planted"] == "duplicate").sum() == 20 and (planted["planted"] == "price_rise").sum() == 10
    assert (planted.loc[planted["planted"] != "", "date"] >= pd.Timestamp("2024-07-01")).all()
    assert len(planted) == len(transactions) + 20
    kinds = label(build_anomaly_features(planted))
    assert set(kinds.unique()) <= {"", "shock", "duplicate", "price_rise"}


def test_scores_report_counts():
    kinds = pd.Series(["", "", "shock", "duplicate", ""])
    report = scores_report(kinds, np.array([False, True, True, False, False]))
    assert report["precision"] == 0.5 and report["recall"] == 0.5
    assert report["recall_shock"] == 1.0 and report["recall_duplicate"] == 0.0
    assert report["false_alarm_rate"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("method", ["forest", "rule"])
def test_detector_flags_close_to_its_rate(method):
    transactions, _ = generate_dataset(n_users=5, months=12, start="2024-01-01", seed=4)
    detector = train_final(transactions, contamination=0.02, method=method)
    out = detector.detect(transactions)
    assert {"anomaly_score", "is_anomaly", "reason"} <= set(out.columns)
    if method == "forest":
        assert 0.01 <= out["is_anomaly"].mean() <= 0.03
    else:
        assert (out["is_anomaly"].to_numpy() == rule_flags(out)).all()
    assert (out.loc[out["is_anomaly"], "reason"] != "").all() and (out.loc[~out["is_anomaly"], "reason"] == "").all()


def test_detector_loads_without_mlflow():
    code = "import sys, src.features.anomaly_features; print('mlflow' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() == "False"
