"""Error measures for the forecasts. All take array-likes of actual and predicted values (in euro).

mae            mean absolute error - the main score, easy to explain ("off by €x on average")
rmse           root mean squared error - punishes big misses more
smape          symmetric mean absolute percentage error, 0-200 %; months where both values are 0
               count as a perfect 0 % instead of dividing by zero
coverage       share of actual values inside [lower, upper] (target for a 10-90 % range: ~80 %)
pinball_loss   the standard score of a quantile forecast (lower is better)
summarize      any of the above per group, e.g. per series or per user type
"""

import numpy as np
import pandas as pd


def _arrays(*values) -> list[np.ndarray]:
    arrays = [np.asarray(v, dtype=float) for v in values]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError("All inputs must have the same length.")
    return arrays


def mae(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def smape(actual, predicted) -> float:
    a, p = _arrays(actual, predicted)
    denominator = np.abs(a) + np.abs(p)
    ratio = np.divide(2 * np.abs(a - p), denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return float(100 * np.mean(ratio))


def coverage(actual, lower, upper) -> float:
    a, lo, hi = _arrays(actual, lower, upper)
    return float(np.mean((a >= lo) & (a <= hi)))


def pinball_loss(actual, predicted, quantile: float) -> float:
    """Under-forecasts cost `quantile` per euro, over-forecasts cost `1 - quantile` per euro."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    a, p = _arrays(actual, predicted)
    diff = a - p
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def point_metrics(actual, predicted) -> dict:
    return {"mae": mae(actual, predicted), "rmse": rmse(actual, predicted),
            "smape": smape(actual, predicted), "n": int(len(np.asarray(actual)))}


def summarize(df: pd.DataFrame, by: str | list[str], actual: str = "actual", predicted: str = "predicted",
              lower: str | None = None, upper: str | None = None) -> pd.DataFrame:
    """Point metrics (plus coverage when lower/upper are given) per group, as a tidy table."""
    def one(group: pd.DataFrame) -> pd.Series:
        row = point_metrics(group[actual], group[predicted])
        if lower and upper:
            row["coverage"] = coverage(group[actual], group[lower], group[upper])
        return pd.Series(row)

    table = df.groupby(by).apply(one, include_groups=False)
    table["n"] = table["n"].astype(int)
    return table.reset_index()
