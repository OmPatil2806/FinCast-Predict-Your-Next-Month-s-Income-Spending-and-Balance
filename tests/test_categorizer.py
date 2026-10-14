"""Tests for the keyword rules of the categoriser."""

import pandas as pd
import pytest

from src.preprocessing.categorizer import rule_categorize, rule_coverage


def categorize_one(description: str, amount: float) -> str | None:
    df = pd.DataFrame({"description": [description], "amount": [amount]})
    return rule_categorize(df).loc[0, "category"]


@pytest.mark.parametrize("description, amount, expected", [
    ("REWE Markt Berlin", -42.37, "Groceries"),
    ("Miete Wohnung Juli", -850.0, "Rent"),
    ("Gehalt ACME GmbH", 3200.0, "Income"),
    ("Netflix.com", -13.99, "Subscriptions"),        # not Savings ("etf" inside "netflix")
    ("TotalEnergies", -60.0, None),                  # not Utilities ("energie" inside the word)
    ("E.ON Energie Abschlag", -70.0, "Utilities"),
    ("Aldi Talk", -7.99, "Utilities"),               # not Groceries
    ("ALDI SUED", -29.67, "Groceries"),
    ("Amazon Prime", -8.99, "Subscriptions"),        # not Shopping
    ("Amazon EU Marketplace", -34.99, "Shopping"),
    ("Rechnung RE-202401-17 Mueller GmbH", 1200.0, "Income"),
    ("Tierarzt Rechnung", -120.0, None),             # outgoing "Rechnung" is not income
    ("rewe city 0815", -5.0, "Groceries"),           # case does not matter
    ("Kaufland Berlin", -30.0, None),                # deliberately left for the text model
])
def test_rules(description, amount, expected):
    assert categorize_one(description, amount) == expected


def test_source_column_and_index_are_kept():
    df = pd.DataFrame({"description": ["REWE", "Vapiano"], "amount": [-10.0, -20.0]}, index=[7, 9])
    result = rule_categorize(df)
    assert list(result.index) == [7, 9]
    assert result["category_source"].tolist() == ["rule", None]


def test_custom_rules_first_match_wins():
    rules = {"fallback": "Other", "categories": [
        {"name": "A", "keywords": ["shop"]},
        {"name": "B", "keywords": ["shop", "store"]},
    ]}
    df = pd.DataFrame({"description": ["Coffee Shop", "Big Store"], "amount": [-1.0, -1.0]})
    assert rule_categorize(df, rules)["category"].tolist() == ["A", "B"]


def test_coverage_report():
    df = pd.DataFrame({
        "description": ["REWE", "Netto Marken-Discount", "Gehalt", "Vapiano"],
        "amount": [-10.0, -20.0, 3000.0, -15.0],
        "category": ["Groceries", "Groceries", "Income", "Dining"],
    })
    report = rule_coverage(df)
    assert report["coverage"] == 0.5
    assert report["accuracy_when_matched"] == 1.0
    assert report["coverage_by_category"] == {"Dining": 0.0, "Groceries": 0.5, "Income": 1.0}


# ---------- text model helpers and training (step 5.3) ----------

from src.preprocessing.categorizer import direction, merchant_key, normalize_description  # noqa: E402
from src.training.train_categorizer import build_model, distinct_examples, predict_with_rules  # noqa: E402


def test_normalize_description():
    text = pd.Series(["McDonalds 1180", "  REWE   City 0815 ", None])
    assert normalize_description(text).tolist() == ["mcdonalds 0000", "rewe city 0000", ""]
    assert merchant_key(normalize_description(text)).tolist() == ["mcdonalds", "rewe", ""]
    assert direction(pd.Series([10.0, -3.0])).tolist() == ["in", "out"]


def test_model_learns_and_rules_come_first():
    transactions = pd.DataFrame({
        "description": ["Kaufland Berlin", "Kaufland Hamburg", "Vapiano 12", "Vapiano 99", "Gehalt", "REWE"] * 3,
        "amount": [-30.0, -25.0, -15.0, -18.0, 3000.0, -12.0] * 3,
        "category": ["Groceries", "Groceries", "Dining", "Dining", "Income", "Groceries"] * 3,
    })
    examples = distinct_examples(transactions)
    assert len(examples) == 5            # "Vapiano 12" and "Vapiano 99" collapse to one text
    model = build_model().fit(examples[["text", "direction"]], examples["category"])
    new = pd.DataFrame({"description": ["Kaufland Muenchen", "Vapiano 7", "REWE City"], "amount": [-9.0, -9.0, -9.0]})
    result = predict_with_rules(model, new)
    assert result["category"].tolist() == ["Groceries", "Dining", "Groceries"]
    assert result["category_source"].tolist() == ["model", "model", "rule"]



# ---------- categorize(): rules + model + threshold (step 5.4) ----------

import numpy as np  # noqa: E402

from src.preprocessing.categorizer import categorize  # noqa: E402


class FixedModel:
    """Stand-in model whose top answer is always 'Dining' with a chosen confidence."""

    classes_ = np.array(["Dining", "Groceries", "Health", "Leisure"])

    def __init__(self, confidence):
        self.confidence = confidence

    def predict_proba(self, X):
        rest = (1 - self.confidence) / 3          # spread the remainder so Dining stays on top
        return np.tile([self.confidence, rest, rest, rest], (len(X), 1))


ROWS = pd.DataFrame({"description": ["REWE Markt", "Vapiano"], "amount": [-10.0, -20.0]})


def test_rules_win_and_confident_model_is_used():
    result = categorize(ROWS, FixedModel(0.9), threshold=0.5)
    assert result["category"].tolist() == ["Groceries", "Dining"]
    assert result["category_source"].tolist() == ["rule", "model"]
    assert result["confidence"].tolist() == [1.0, 0.9]
    assert result["needs_review"].tolist() == [False, False]


def test_unsure_model_falls_back_to_other_and_flags_review():
    result = categorize(ROWS, FixedModel(0.3), threshold=0.5)
    assert result.loc[1, "category"] == "Other"
    assert result.loc[1, "needs_review"]
    assert result.loc[0, "category"] == "Groceries"          # the rule is untouched


def test_without_model_rules_only():
    result = categorize(ROWS, model=None)
    assert result["category"].tolist() == ["Groceries", "Other"]
    assert result["category_source"].tolist() == ["rule", None]
    assert result["needs_review"].tolist() == [False, True]
