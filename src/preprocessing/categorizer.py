"""Assign a category to each transaction.

Step 5.2: keyword rules from config/categories.yaml (rule_categorize).
Step 5.4 adds the text model for rows the rules cannot match, wrapped in one categorize(df).

Rule semantics (see the comments in categories.yaml):
  - lower-cased description, keyword must match at the start of a word
  - optional sign: 'income' only for amount > 0, 'expense' only for amount < 0
  - categories are tried top to bottom, the first match wins
"""

import re

import pandas as pd

from src.utils.helpers import load_categories

RULE = "rule"


def _compiled_rules(rules: dict | None = None) -> list[tuple[str, str | None, re.Pattern]]:
    rules = rules or load_categories()
    compiled = []
    for category in rules["categories"]:
        keywords = sorted(category["keywords"], key=len, reverse=True)   # longest first, e.g. "aldi talk" before "aldi"
        pattern = re.compile(r"\b(?:" + "|".join(map(re.escape, keywords)) + ")", re.IGNORECASE)
        compiled.append((category["name"], category.get("sign"), pattern))
    return compiled


def rule_categorize(df: pd.DataFrame, rules: dict | None = None) -> pd.DataFrame:
    """Return df's index with two columns: category and category_source.

    Rows no rule matches get category None and category_source None (they go to the model).
    df needs a description and an amount column.
    """
    text = df["description"].fillna("").astype(str)
    amount = pd.to_numeric(df["amount"], errors="coerce")
    category = pd.Series(None, index=df.index, dtype=object)

    for name, sign, pattern in _compiled_rules(rules):
        open_rows = category.isna()
        if sign == "income":
            open_rows &= amount > 0
        elif sign == "expense":
            open_rows &= amount < 0
        if not open_rows.any():
            continue
        hits = text[open_rows].str.contains(pattern)
        category[hits[hits].index] = name

    matched = category.notna()
    result = pd.DataFrame({"category": category, "category_source": RULE}, index=df.index, dtype=object)
    result.loc[~matched, ["category", "category_source"]] = None   # None, not NaN: saved as NULL in SQLite
    return result


def rule_coverage(df: pd.DataFrame, rules: dict | None = None,
                  true_category: str | None = "category") -> dict:
    """How much the rules cover: overall share of rows, and (if true labels exist) per true category
    plus the accuracy of the rows the rules did match."""
    predicted = rule_categorize(df, rules)["category"]
    covered = predicted.notna()
    report = {"rows": len(df), "covered_rows": int(covered.sum()),
              "coverage": float(covered.mean()) if len(df) else 0.0}
    if true_category and true_category in df.columns:
        truth = df[true_category]
        report["accuracy_when_matched"] = float((predicted[covered] == truth[covered]).mean()) if covered.any() else None
        report["coverage_by_category"] = covered.groupby(truth).mean().round(3).to_dict()
    return report
