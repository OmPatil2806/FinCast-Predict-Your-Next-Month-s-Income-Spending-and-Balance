"""Honest evaluation for time series: always train on the past, test on the future.

backtest_folds        rolling origin over time: for each test month T, train on every row whose
                      target month is before T (all known by the end of T-1), test on rows whose
                      target month is T. Expanding window, one month ahead, no random splits.
holdout_user_folds    new-user (cold start) case: users the model never saw, with their history
                      cut to 3, 6 or 12 months, features rebuilt from only that short history.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    test_month: pd.Period
    train_idx: np.ndarray          # row positions in the feature table
    test_idx: np.ndarray


@dataclass
class HoldoutFold:
    history_months: int
    origin: pd.Period              # last month the new users' data covers
    test_users: list
    train: pd.DataFrame            # other users, rows whose target month <= origin
    test: pd.DataFrame             # held-out users, one forecast each, from the truncated history


def backtest_folds(table: pd.DataFrame, min_train_months: int = 12) -> list[Fold]:
    """One fold per test month, oldest first. Rows without a target are never used.

    The first test month leaves at least `min_train_months` target months for training.
    """
    target_months = table["target_month"]
    has_target = table["target"].notna().to_numpy()
    months = sorted(target_months[has_target].unique())
    folds = []
    for test_month in months[min_train_months:]:
        train = np.flatnonzero(has_target & (target_months < test_month).to_numpy())
        test = np.flatnonzero(has_target & (target_months == test_month).to_numpy())
        folds.append(Fold(test_month, train, test))
    return folds


def holdout_user_folds(monthly: pd.DataFrame, build, users_share: float = 0.2,
                       histories=(3, 6, 12), origin: pd.Period | None = None,
                       seed: int = 42) -> list[HoldoutFold]:
    """Cold-start evaluation on users the model never saw.

    monthly: the user x month x category table (step 5.7); build: a function monthly -> (features, info),
    normally build_features. For each history length h the held-out users keep only the h months up to
    `origin` (plus origin+1, which is only used as the target), their features are rebuilt from that,
    and the forecast made at `origin` is the test row. Default origin: second-to-last month.
    """
    data = monthly.copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M") if not isinstance(
        data["month"].dtype, pd.PeriodDtype) else data["month"]
    users = np.array(sorted(data["user_id"].unique()))
    rng = np.random.default_rng(seed)
    test_users = sorted(rng.choice(users, size=max(1, int(len(users) * users_share)), replace=False))
    origin = origin or sorted(data["month"].unique())[-2]

    full, _ = build(data[~data["user_id"].isin(test_users)])
    train = full[full["target"].notna() & (full["target_month"] <= origin)]

    folds = []
    for h in histories:
        window = data["user_id"].isin(test_users) & data["month"].between(origin - (h - 1), origin + 1)
        rebuilt, _ = build(data[window])
        test = rebuilt[(rebuilt["month"] == origin) & rebuilt["target"].notna()]
        folds.append(HoldoutFold(h, origin, list(test_users), train, test))
    return folds
