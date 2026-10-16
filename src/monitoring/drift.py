"""Step 9.3: does this user's spending look like the data the models learned from?

Spending mix = each category's share of spending over the last 3 complete months.

build_reference(monthly)  from the training table (step 5.7): every user's rolling 3-month mixes, their
                          average (the reference mix), the typical 10-90 % range of each share, and the
                          alarm threshold = the 95th percentile of the training windows' distance to the
                          reference. The threshold is chosen on half of the users and checked on the
                          other half. Saved to models/drift_reference.json (small, committed, so the app
                          needs no training data).
check_drift(transactions) the user's distance to the reference mix and, above the threshold, a plain note
                          naming the category furthest outside its usual range.

Distance = total variation distance: half the sum of |share - reference share| over all categories, so
0 = identical mix and 1 = nothing in common. Spending in a category the models never saw counts fully.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.aggregate_monthly import TOTAL, aggregate_user_month_category
from src.utils.helpers import display_path, load_config, resolve_path

WINDOW = 3
QUANTILE = 0.95
REFERENCE_FILE = "drift_reference.json"


def window_mixes(monthly: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """One row per user and month: category shares of spending over that month and the window-1 before it."""
    data = monthly[(monthly["category"] != TOTAL) & (monthly["category"] != "Income")]
    wide = data.pivot_table(index=["user_id", "month"], columns="category", values="expenses", aggfunc="sum").fillna(0.0)
    rolled = wide.groupby(level="user_id", group_keys=False).apply(lambda g: g.rolling(window, min_periods=window).sum())
    rolled = rolled.dropna()
    totals = rolled.sum(axis=1)
    return rolled[totals > 0].div(totals[totals > 0], axis=0)


def distance(mixes: pd.DataFrame, reference: pd.Series) -> pd.Series:
    """Total variation distance of each row's mix to the reference mix."""
    columns = mixes.columns.union(reference.index)
    a = mixes.reindex(columns=columns, fill_value=0.0)
    b = reference.reindex(columns, fill_value=0.0)
    return 0.5 * (a - b).abs().sum(axis=1)


def build_reference(monthly: pd.DataFrame, seed: int = 42, quantile: float = QUANTILE) -> dict:
    mixes = window_mixes(monthly)
    users = np.array(sorted(mixes.index.get_level_values("user_id").unique()))
    rng = np.random.default_rng(seed)
    pick = set(rng.choice(users, size=len(users) // 2, replace=False))
    in_pick = mixes.index.get_level_values("user_id").isin(pick)

    reference = mixes[in_pick].mean()
    threshold = float(distance(mixes[in_pick], reference).quantile(quantile))
    check = distance(mixes[~in_pick], reference)
    final_reference = mixes.mean()
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window_months": WINDOW,
        "reference_mix": final_reference.round(4).to_dict(),
        "usual_range": {c: [round(float(mixes[c].quantile(0.1)), 4), round(float(mixes[c].quantile(0.9)), 4)]
                        for c in mixes.columns},
        "threshold": round(threshold, 4),
        "threshold_quantile": quantile,
        "check": {"windows": int(len(check)), "alarm_rate": round(float((check > threshold).mean()), 4),
                  "users": int(len(users) - len(pick))},
    }


def save_reference(reference: dict, path: Path | None = None) -> Path:
    path = path or resolve_path(load_config()["paths"]["models"]) / REFERENCE_FILE
    path.write_text(json.dumps(reference, indent=2))
    return path


def load_reference(path: Path | None = None) -> dict | None:
    path = path or resolve_path(load_config()["paths"]["models"]) / REFERENCE_FILE
    return json.loads(path.read_text()) if path.exists() else None


def user_mix(transactions: pd.DataFrame, as_of=None) -> pd.Series | None:
    """Spending shares over the user's last 3 complete months (None with fewer months or no spending)."""
    from src.forecasting.predictor import complete_history
    history = complete_history(transactions, as_of)
    if history.empty:
        return None
    monthly = aggregate_user_month_category(history.assign(user_id="user"))
    mixes = window_mixes(monthly)
    return None if mixes.empty else mixes.iloc[-1]


def check_drift(transactions: pd.DataFrame, reference: dict | None = None, as_of=None) -> dict | None:
    """{'distance', 'threshold', 'drifted', 'category', 'share', 'usual', 'message'} or None if not enough data."""
    reference = reference if reference is not None else load_reference()
    if reference is None:
        return None
    mix = user_mix(transactions, as_of)
    if mix is None:
        return None
    ref = pd.Series(reference["reference_mix"])
    dist = float(distance(mix.to_frame().T, ref).iloc[0])
    result = {"distance": round(dist, 4), "threshold": reference["threshold"],
              "drifted": dist > reference["threshold"], "category": None, "share": None, "usual": None,
              "message": None}
    if not result["drifted"]:
        return result

    def outside(category: str) -> float:
        share = float(mix.get(category, 0.0))
        low, high = reference["usual_range"].get(category, [0.0, 0.0])
        return max(low - share, share - high, 0.0)

    category = max(mix.index.union(ref.index), key=outside)
    share = float(mix.get(category, 0.0))
    usual = reference["usual_range"].get(category)
    if usual:
        detail = (f"{category} is {share:.0%} of your spending over the last {WINDOW} months; for the people the "
                  f"models learned from it is usually {usual[0]:.0%}–{usual[1]:.0%}.")
    else:
        detail = f"{share:.0%} of your spending is in “{category}”, a category the models have not seen."
    result.update(category=category, share=round(share, 4), usual=usual, message=(
        "Your spending pattern is quite different from the data FinCast's models learned from, so its "
        f"forecasts may be less accurate for you. {detail}"))
    return result


def main() -> None:
    config = load_config()
    monthly = pd.read_parquet(resolve_path(config["paths"]["data_processed"]) / config["processed"]["monthly_file"])
    reference = build_reference(monthly, seed=config["project"]["seed"])
    path = save_reference(reference)
    c = reference["check"]
    print(f"Reference mix from {WINDOW}-month windows; alarm threshold {reference['threshold']:.3f} "
          f"(95th percentile on half the users). On the other {c['users']} users it fires for "
          f"{c['alarm_rate']:.1%} of {c['windows']:,} windows. Saved {display_path(path)}")


if __name__ == "__main__":
    main()
