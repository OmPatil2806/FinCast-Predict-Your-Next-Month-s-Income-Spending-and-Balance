"""Step 8.5: what-if scenarios on top of next month's forecast.

A Scenario says how the user wants to change next month:
    income_change     e.g. -0.10 = 10 % less income (a lost side job, a raise of +0.05, ...)
    category_changes  e.g. {"Dining": -0.30} = 30 % less on eating out
    one_off           an extra one-time cost in euro (a holiday, a new laptop); negative = extra money in

simulate(forecast, scenario) returns a new Forecast with the same shape, so the page can show it with the
same cards, table and risk check. Rules:
    - income, each changed category and its range are scaled by (1 + change); never below 0
    - total spending moves by the sum of the category changes in euro plus the one-off cost
      (the total has its own model, so it is shifted, not rebuilt from the categories)
    - the end balance is recomputed: current balance + income - spending, the range pairing low income
      with high spending exactly as in the predictor
Nothing is retrained and nothing is saved: a scenario is a quick "what if", not a new forecast.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from src.features.build_features import EXPENSES, INCOME

CATEGORY_PREFIX = EXPENSES + ":"


@dataclass
class Scenario:
    income_change: float = 0.0
    category_changes: dict = field(default_factory=dict)
    one_off: float = 0.0

    def is_empty(self) -> bool:
        return not (self.income_change or self.one_off or any(self.category_changes.values()))


def _scale(row, factor: float) -> dict:
    factor = max(factor, 0.0)
    return {k: max(row[k] * factor, 0.0) for k in ("lower", "predicted", "upper")}


def simulate(forecast, scenario: Scenario):
    """The forecast after the scenario's changes (a new Forecast; the original is not modified)."""
    series = forecast.series.copy()
    index = series.set_index("series").index
    values = series.set_index("series")[["lower", "predicted", "upper"]].astype(float)

    values.loc[INCOME] = list(_scale(values.loc[INCOME], 1 + scenario.income_change).values())

    spending_shift = float(scenario.one_off)
    for category, change in scenario.category_changes.items():
        key = CATEGORY_PREFIX + category
        if not change or key not in index:
            continue
        before = values.loc[key, "predicted"]
        values.loc[key] = list(_scale(values.loc[key], 1 + change).values())
        spending_shift += values.loc[key, "predicted"] - before

    total = values.loc[EXPENSES] + spending_shift
    values.loc[EXPENSES] = np.clip(total.to_numpy(), 0, None)

    series[["lower", "predicted", "upper"]] = values.loc[series["series"]].to_numpy()
    income, spend, current = values.loc[INCOME], values.loc[EXPENSES], forecast.balance["current"]
    balance = {"current": current,
               "predicted": round(float(current + income["predicted"] - spend["predicted"]), 2),
               "lower": round(float(current + income["lower"] - spend["upper"]), 2),
               "upper": round(float(current + income["upper"] - spend["lower"]), 2)}
    return replace(forecast, series=series, balance=balance)


def compare(before, after) -> dict:
    """Change in income, spending and end balance between two forecasts (euro, after - before)."""
    def get(f, s):
        return float(f.series.set_index("series").loc[s, "predicted"])
    return {"income": get(after, INCOME) - get(before, INCOME),
            "expenses": get(after, EXPENSES) - get(before, EXPENSES),
            "balance": after.balance["predicted"] - before.balance["predicted"]}
