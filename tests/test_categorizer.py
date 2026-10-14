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
