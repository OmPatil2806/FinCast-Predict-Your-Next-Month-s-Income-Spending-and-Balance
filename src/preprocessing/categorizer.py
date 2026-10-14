"""Assign a category to each transaction.

Step 5.2: keyword rules from config/categories.yaml (rule_categorize).
Step 5.4: categorize(df, model) - rules first, the text model for the rest, and "Other" plus a
review flag when the model is not confident enough.

Rule semantics (see the comments in categories.yaml):
  - lower-cased description, keyword must match at the start of a word
  - optional sign: 'income' only for amount > 0, 'expense' only for amount < 0
  - categories are tried top to bottom, the first match wins
"""

import re
from pathlib import Path

import joblib
import pandas as pd

from src.utils.helpers import load_categories, load_config, resolve_path

RULE = "rule"
MODEL = "model"


def normalize_description(text: pd.Series) -> pd.Series:
    """Text the model sees: lower case, every digit as 0, single spaces.

    "McDonalds 1180" and "McDonalds 2231" become the same "mcdonalds 0000", so store and
    invoice numbers do not split one merchant into many.
    """
    return (text.fillna("").astype(str).str.lower()
            .str.replace(r"\d", "0", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip())


def direction(amount: pd.Series) -> pd.Series:
    """'in' for money received, 'out' for money spent (a model feature next to the text)."""
    return pd.to_numeric(amount, errors="coerce").gt(0).map({True: "in", False: "out"})


def merchant_key(normalized: pd.Series) -> pd.Series:
    """First word of the normalised description, used to keep a merchant's spellings together
    when splitting train/test ("rewe markt berlin" and "rewe city 0000" -> "rewe")."""
    return normalized.str.split().str[0].fillna("")


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


# ---------- rules + model (step 5.4) ----------

def model_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """The two columns the text model expects."""
    return pd.DataFrame({"text": normalize_description(df["description"]).values,
                         "direction": direction(df["amount"]).values}, index=df.index)


def categorize(df: pd.DataFrame, model=None, threshold: float | None = None) -> pd.DataFrame:
    """Categorise transactions: rules first, the model for the rest.

    Returns df's index with columns:
      category         final category (the fallback "Other" when unsure)
      category_source  "rule" or "model" (None if there was no model and no rule matched)
      confidence       1.0 for rules, the model's top probability otherwise
      needs_review     True when the category is a fallback the user should check
    Without a model, rows no rule matches become the fallback with needs_review=True.
    """
    rules = load_categories()
    threshold = load_config()["categorizer"]["confidence_threshold"] if threshold is None else threshold

    result = rule_categorize(df, rules)
    result["confidence"] = result["category"].notna().astype(float)
    result["needs_review"] = False

    open_rows = result["category"].isna()
    if open_rows.any() and model is not None:
        probabilities = model.predict_proba(model_inputs(df[open_rows]))
        best = probabilities.argmax(axis=1)
        result.loc[open_rows, "category"] = model.classes_[best]
        result.loc[open_rows, "category_source"] = MODEL
        result.loc[open_rows, "confidence"] = probabilities.max(axis=1).round(4)
        unsure = open_rows & (result["confidence"] < threshold)
        result.loc[unsure, "category"] = rules["fallback"]
        result.loc[unsure, "needs_review"] = True
    elif open_rows.any():
        result.loc[open_rows, "category"] = rules["fallback"]
        result.loc[open_rows, "needs_review"] = True
    return result


def categorizer_path() -> Path:
    config = load_config()
    return resolve_path(config["paths"]["models"]) / config["categorizer"]["model_file"]


def load_categorizer(path: str | Path | None = None):
    """The trained text model from models/, or None if it has not been trained yet (rules only then)."""
    path = Path(path) if path else categorizer_path()
    return joblib.load(path) if path.exists() else None
