"""Drift check (step 9.3): normal synthetic users pass, a very different spending mix is flagged."""

import pandas as pd
import pytest

from src.ingestion.synthetic_generator import generate_dataset
from src.monitoring.drift import build_reference, check_drift, distance, load_reference, window_mixes
from src.preprocessing.aggregate_monthly import aggregate_user_month_category


def test_distance_is_total_variation():
    ref = pd.Series({"Rent": 0.5, "Food": 0.5})
    mixes = pd.DataFrame({"Rent": [0.5, 1.0, 0.0], "Food": [0.5, 0.0, 0.0], "Pets": [0.0, 0.0, 1.0]})
    assert distance(mixes, ref).tolist() == [0.0, 0.5, 1.0]


def test_window_mixes_sum_to_one():
    transactions, _ = generate_dataset(n_users=2, months=6, start="2025-01-01", seed=1)
    mixes = window_mixes(aggregate_user_month_category(transactions))
    assert len(mixes) == 2 * 4                                   # 6 months -> 4 complete 3-month windows per user
    assert mixes.sum(axis=1).round(6).eq(1.0).all()


@pytest.fixture(scope="module")
def reference():
    transactions, _ = generate_dataset(n_users=40, months=12, start="2025-01-01", seed=7)
    return build_reference(aggregate_user_month_category(transactions), seed=1)


def test_reference_alarm_rate_is_near_the_target(reference):
    assert 0 < reference["threshold"] < 1
    assert reference["check"]["alarm_rate"] < 0.2
    assert abs(sum(reference["reference_mix"].values()) - 1) < 1e-3


def test_normal_user_passes_and_unusual_user_is_flagged(reference):
    transactions, _ = generate_dataset(n_users=1, months=6, start="2025-04-01", seed=99)
    mine = transactions.drop(columns=["user_id", "archetype"])
    normal = check_drift(mine, reference, as_of="2025-10-15")
    assert normal is not None and normal["distance"] < 1

    skewed = mine.copy()
    rent = skewed["category"] == "Rent"
    skewed.loc[rent, "amount"] *= 4                                # rent suddenly 4x as big
    result = check_drift(skewed, reference, as_of="2025-10-15")
    assert result["drifted"] and result["category"] == "Rent" and "Rent is" in result["message"]

    pets = mine.copy()
    pets.loc[pets["amount"] < 0, "category"] = "Pets"              # everything in a category the models never saw
    assert "have not seen" in check_drift(pets, reference, as_of="2025-10-15")["message"]


def test_too_little_history_gives_no_verdict(reference):
    tx = pd.DataFrame({"date": ["2025-09-01", "2025-09-10"], "amount": [-10.0, -20.0], "category": ["Rent", "Dining"]})
    assert check_drift(tx, reference, as_of="2025-10-15") is None


def test_committed_reference_loads():
    ref = load_reference()
    assert ref is not None and set(ref) >= {"reference_mix", "usual_range", "threshold", "check"}
