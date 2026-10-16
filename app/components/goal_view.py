"""Plain-language text for a goal projection (step 8.7). Kept free of Streamlit so it can be tested."""

import pandas as pd

from app.components.cards import format_euro

STATUS = {
    "reached": ("🎉", "Reached"),
    "on_track": ("✅", "On track"),
    "behind": ("⚠️", "Behind"),
    "overdue": ("❌", "Target date passed"),
    "no_progress": ("⏸️", "Not saving at the moment"),
    "no_deadline": ("📅", "No target date"),
}


def status_label(status: str) -> str:
    icon, text = STATUS[status]
    return f"{icon} {text}"


def goal_summary(goal: dict, projection) -> list[str]:
    """Two or three short lines explaining where the goal stands."""
    lines = [f"{format_euro(goal['saved_amount'], 0)} of {format_euro(goal['target_amount'], 0)} saved "
             f"({projection.progress:.0%})"]
    deadline = goal.get("target_date")
    deadline = pd.Timestamp(deadline).strftime("%d %B %Y").lstrip("0") if pd.notna(deadline) and deadline else None
    if projection.status == "reached":
        return lines + ["Goal reached — well done."]
    if projection.projected_date is not None:
        lines.append(f"At your pace of {format_euro(projection.pace, 0)} a month you reach it around "
                     f"**{projection.projected_date:%B %Y}** ({projection.months_needed:g} months).")
    else:
        lines.append("You are not saving at the moment, so no date can be projected.")
    if projection.status == "overdue":
        lines.append(f"The target date ({deadline}) has passed; set a new date to get a plan.")
    elif projection.required_per_month is not None:
        gap = projection.required_per_month - projection.pace
        extra = f" — {format_euro(gap, 0)} more than now" if gap > 0 else ""
        lines.append(f"To make {deadline}: save {format_euro(projection.required_per_month, 0)} a month{extra}.")
    return lines
