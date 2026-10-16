"""Step 8.6: the user's unusual expenses, for the Anomalies page.

    load_detector()                      the exported detector (models/anomaly.joblib, step 7.10)
    review_list(transactions, detector)  flagged expenses, most unusual first, with the user's verdict

The detector compares each expense only with the user's own EARLIER transactions, so the first
months of a new user produce few or no flags; history_note() says so. A user's Confirm / Dismiss is
stored in transactions.anomaly_status; dismissed flags are hidden unless asked for.
"""

from pathlib import Path

import joblib
import pandas as pd

from src.utils.helpers import load_config, resolve_path

MIN_MONTHS = 3          # below this the checks have almost nothing to compare with
COLUMNS = ["id", "date", "description", "amount", "category", "anomaly_score", "reason", "anomaly_status"]


def load_detector(models_dir: str | Path | None = None):
    config = load_config()
    models_dir = Path(models_dir) if models_dir else resolve_path(config["paths"]["models"])
    path = models_dir / config["anomaly"]["model_file"]
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run `python pipelines/run_training.py`")
    return joblib.load(path)


def review_list(transactions: pd.DataFrame, detector, since=None, include_dismissed: bool = False) -> pd.DataFrame:
    """Flagged expenses (optionally from `since` on), most unusual first."""
    if transactions.empty:
        return pd.DataFrame(columns=COLUMNS)
    data = transactions.copy()
    if "anomaly_status" not in data:
        data["anomaly_status"] = None
    if "user_id" not in data:                       # one user's rows from the app database
        data["user_id"] = "user"
    scored = detector.detect(data)
    flagged = scored[scored["is_anomaly"]]
    if since is not None:
        flagged = flagged[flagged["date"] >= pd.Timestamp(since)]
    if not include_dismissed:
        flagged = flagged[flagged["anomaly_status"].fillna("") != "dismissed"]
    return flagged.sort_values("anomaly_score", ascending=False)[COLUMNS].reset_index(drop=True)


def history_note(transactions: pd.DataFrame) -> str | None:
    """A short note when the history is too short for the checks to mean much."""
    if transactions.empty:
        return None
    months = pd.to_datetime(transactions["date"]).dt.to_period("M").nunique()
    if months < MIN_MONTHS:
        return (f"You have {months} month{'s' if months != 1 else ''} of transactions. Unusual spending is "
                "found by comparing with your own earlier payments, so expect few flags until you have "
                f"{MIN_MONTHS} or more months.")
    return None
