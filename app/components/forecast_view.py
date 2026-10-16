"""Turn a predictor Forecast (step 8.1) into what the Forecast page shows: plain model names, the
category table and the negative-balance warning. Kept free of Streamlit so it can be tested."""

import pandas as pd

from app.components.cards import format_euro

MODEL_LABELS = {
    "lightgbm": "ML model",
    "last_month": "Same as last month",
    "same_month_last_year": "Same month last year",
    "mean_3": "3-month average",
}


def model_label(champion: str) -> str:
    """'lightgbm' -> 'ML model', 'mean_3 (new series)' -> '3-month average (new category)'."""
    name, _, extra = champion.partition(" ")
    label = MODEL_LABELS.get(name, name)
    return f"{label} (new category)" if "new series" in extra else label


def range_text(lower: float, upper: float) -> str:
    return f"{format_euro(lower, 0)} – {format_euro(upper, 0)}"


def category_table(forecast) -> pd.DataFrame:
    """One row per spending category, largest forecast first."""
    rows = forecast.series[forecast.series["series"].str.startswith("expenses:")].copy()
    rows["category"] = rows["series"].str.removeprefix("expenses:")
    rows["change"] = (rows["predicted"] - rows["last_value"]).round(0) + 0.0      # + 0.0 turns -0 into 0
    rows["range"] = [range_text(lo, hi) for lo, hi in zip(rows["lower"], rows["upper"])]
    rows["model"] = rows["champion"].map(model_label)
    return (rows.sort_values("predicted", ascending=False)
            [["category", "last_value", "predicted", "change", "range", "model"]].reset_index(drop=True))


def balance_risk(forecast) -> tuple[str, str] | None:
    """('error' | 'warning', message) when the end balance could go below zero, else None."""
    balance, month = forecast.balance, forecast.target_month.strftime("%B %Y")
    if balance["predicted"] < 0:
        return "error", (f"Your balance is expected to end {month} below zero, at "
                         f"{format_euro(balance['predicted'], 0)}. Look at the biggest categories below.")
    if balance["lower"] < 0:
        return "warning", (f"Risk of a negative balance: in a weak month (low income, high spending) your balance "
                           f"could end {month} at {format_euro(balance['lower'], 0)}.")
    return None
