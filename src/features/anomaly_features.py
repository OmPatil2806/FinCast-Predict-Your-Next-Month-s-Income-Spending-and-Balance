"""Features for spotting unusual expenses, one row per transaction (step 7.10).

Every feature compares a transaction only with the SAME user's EARLIER transactions (cumulative
sums shifted by one row), so nothing from the future leaks in and a user's first months simply give
neutral values.

    log_amount          log(1 + amount spent)
    category_z          amount vs the user's earlier amounts in this category (z-score, log scale)
    merchant_z          amount vs the user's earlier amounts at this merchant (z-score, log scale;
                        std floored at 0.05 ~ 5 %, so a fixed-price subscription that gets dearer stands out)
    user_z              amount vs all the user's earlier expenses
    merchant_seen       log(1 + earlier transactions at this merchant, capped at 12); 0 = a new merchant
    new_merchant_z      user_z for a first payment to a merchant, else 0 (a big one-off at a new place)
    days_since_merchant days since the last charge at this merchant (capped at 400; 400 = never)
    days_since_same     days since the last charge at this merchant with the SAME amount (duplicate charges)
    day_of_month, weekday

MODEL_FEATURES is what the Isolation Forest sees. log_amount, user_z, day_of_month and weekday are kept
out: in the first try they made the forest flag every rent payment and quarterly tax prepayment
(large, early in the month, but perfectly normal for that merchant). Only features that compare a
charge with the user's own history at that category / merchant remain.

rule_flags is the simple alternative: any z-score >= 3 or the same amount again within 3 days.

AnomalyDetector wraps a fitted Isolation Forest (its score ranks how unusual an expense is), the flag
method chosen in step 7.10 ("rule" or "forest", whichever won on the validation months), the score
threshold, and a short plain-language reason per flag. It is what models/anomaly.joblib holds and loads without MLflow.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.preprocessing.categorizer import merchant_key, normalize_description

FEATURES = ["log_amount", "category_z", "merchant_z", "user_z", "merchant_seen", "new_merchant_z",
            "days_since_merchant", "days_since_same", "day_of_month", "weekday"]
MODEL_FEATURES = ["category_z", "merchant_z", "merchant_seen", "new_merchant_z", "days_since_merchant",
                  "days_since_same"]
SEEN_CAP = 12
NEVER = 400.0                  # "days since" for a merchant / amount never seen before
MIN_HISTORY = 3                # earlier transactions needed before a z-score means anything
STD_FLOOR = 0.05


def _running_z(values: pd.Series, keys: list[pd.Series], floor: float = 0.0) -> pd.Series:
    """z-score of each value against the earlier values of its group (same order as `values`)."""
    grouped = values.groupby(keys, sort=False)
    n = grouped.cumcount()                                         # earlier rows in the group
    total = grouped.cumsum() - values
    squares = (values ** 2).groupby(keys, sort=False).cumsum() - values ** 2
    mean = total / n.where(n > 0)
    var = (squares / n.where(n > 0) - mean ** 2).clip(lower=0)
    std = np.sqrt(var).clip(lower=floor) if floor else np.sqrt(var)
    z = (values - mean) / std.where(std > 0)
    return z.where(n >= MIN_HISTORY).fillna(0.0).clip(-10, 10)


def build_anomaly_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Expenses only (amount < 0), sorted by user and date, with the feature columns added.

    Needs user_id, date, description, amount, category. Rows on the same day keep their input order.
    """
    df = transactions[pd.to_numeric(transactions["amount"], errors="coerce") < 0].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["_order"] = np.arange(len(df))
    df = df.sort_values(["user_id", "date", "_order"], kind="stable").drop(columns="_order").reset_index(drop=True)

    df["merchant"] = merchant_key(normalize_description(df["description"]))
    spent = -df["amount"].astype(float)
    df["log_amount"] = np.log1p(spent)
    user, category, merchant = df["user_id"], df["category"].fillna("Other"), df["merchant"]

    df["category_z"] = _running_z(df["log_amount"], [user, category])
    df["merchant_z"] = _running_z(df["log_amount"], [user, merchant], floor=STD_FLOOR)
    df["user_z"] = _running_z(df["log_amount"], [user])
    seen = df.groupby([user, merchant], sort=False).cumcount()
    df["merchant_seen"] = np.log1p(seen.clip(upper=SEEN_CAP))
    df["new_merchant_z"] = df["user_z"].where(seen == 0, 0.0)

    def days_since(keys) -> pd.Series:
        return df.groupby(keys, sort=False)["date"].diff().dt.days.fillna(NEVER).clip(upper=NEVER)

    df["days_since_merchant"] = days_since([user, merchant])
    df["days_since_same"] = days_since([user, merchant, spent.round(2)])
    df["day_of_month"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    return df


def rule_flags(features: pd.DataFrame) -> np.ndarray:
    """Simple rule on the same features: unusually large for the category or merchant, or a repeat."""
    return ((features["category_z"] >= 3) | (features["merchant_z"] >= 3)
            | (features["days_since_same"] <= 3)).to_numpy()


def reasons(features: pd.DataFrame) -> pd.Series:
    """A short, plain reason for each row (used for flagged ones)."""
    out = pd.Series("unusual pattern", index=features.index)
    out[features["category_z"] >= 3] = "much larger than usual for this category"
    out[(features["merchant_seen"] == 0) & (features["user_z"] >= 2)] = "large payment to a new merchant"
    out[features["merchant_z"] >= 3] = "higher than usual at this merchant"
    out[features["days_since_same"] <= 3] = "same merchant and amount again within 3 days (possible duplicate)"
    return out


@dataclass
class AnomalyDetector:
    """A fitted Isolation Forest plus the threshold that turns scores into flags."""
    model: object                      # sklearn IsolationForest
    threshold: float                   # flag when score >= threshold
    features: list
    contamination: float
    method: str = "forest"             # what decides is_anomaly: "forest" (score >= threshold) or "rule"
    version: int | None = None

    def score(self, features: pd.DataFrame) -> np.ndarray:
        """Higher = more unusual (the negated Isolation Forest decision score)."""
        return -self.model.score_samples(features[self.features].to_numpy(dtype=float))

    def detect(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Transactions -> expense rows with features, anomaly_score, is_anomaly and reason."""
        features = build_anomaly_features(transactions)
        features["anomaly_score"] = self.score(features)
        features["is_anomaly"] = (rule_flags(features) if self.method == "rule"
                                  else features["anomaly_score"] >= self.threshold)
        features["reason"] = reasons(features).where(features["is_anomaly"], "")
        return features
