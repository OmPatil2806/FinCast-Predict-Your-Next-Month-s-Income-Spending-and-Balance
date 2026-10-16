"""Step 8.7: when will a savings goal be reached?

    savings_pace(monthly, forecast_savings)   the user's typical monthly savings: the average of the last
                                              6 complete months and next month's forecast savings
                                              (one noisy month moves it less than a forecast alone)
    project_goal(target, saved, pace, ...)    remaining amount, months needed, projected date, the amount
                                              needed per month to make the target date, and a status:

        reached      saved >= target
        no_progress  the user is not saving (pace <= 0), so no date can be projected
        overdue      the target date has passed and the goal is not reached
        on_track     projected date on or before the target date
        behind       projected date after the target date
        no_deadline  no target date; only the projected date is shown

Every goal is projected as if it got all of the savings, so with several goals the page says so.
"""

import math
from dataclasses import dataclass

import pandas as pd

HISTORY_MONTHS = 6
DAYS_PER_MONTH = 30.44


@dataclass
class GoalProjection:
    remaining: float
    pace: float                              # euro saved per month
    months_needed: float | None              # None when not saving
    projected_date: pd.Timestamp | None      # end of the month the goal is reached
    required_per_month: float | None         # to make the target date (None without a date / when reached)
    months_left: float | None                # until the target date
    status: str
    progress: float                          # share of the target already saved, 0..1


def savings_pace(monthly: pd.DataFrame, forecast_savings: float | None = None,
                 months: int = HISTORY_MONTHS) -> float:
    """Average monthly savings over the last `months` complete months plus next month's forecast."""
    values = list(monthly["savings"].tail(months).astype(float)) if not monthly.empty else []
    if forecast_savings is not None:
        values.append(float(forecast_savings))
    return round(sum(values) / len(values), 2) if values else 0.0


def project_goal(target_amount: float, saved_amount: float, pace: float, target_date=None,
                 today=None) -> GoalProjection:
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    remaining = round(max(float(target_amount) - float(saved_amount), 0.0), 2)
    deadline = pd.Timestamp(target_date).normalize() if target_date not in (None, "") and not pd.isna(target_date) else None
    months_left = None if deadline is None else (deadline - today).days / DAYS_PER_MONTH

    progress = min(max(float(saved_amount) / float(target_amount), 0.0), 1.0) if target_amount > 0 else 0.0

    def result(months_needed, projected, required, status):
        return GoalProjection(remaining, pace, months_needed, projected, required, months_left, status, progress)

    if remaining == 0:
        return result(0.0, today, None, "reached")
    required = None
    if deadline is not None and months_left > 0:
        required = round(remaining / max(months_left, 1.0), 2)     # a goal due this month needs it all now
    if pace <= 0:
        return result(None, None, required, "overdue" if deadline is not None and deadline < today else "no_progress")
    months_needed = remaining / pace
    projected = (today + pd.DateOffset(months=math.ceil(months_needed))) + pd.offsets.MonthEnd(0)
    if deadline is None:
        status = "no_deadline"
    elif deadline < today:
        status = "overdue"
    else:
        status = "on_track" if projected <= deadline + pd.offsets.MonthEnd(0) else "behind"
    return result(round(months_needed, 1), projected, required, status)
