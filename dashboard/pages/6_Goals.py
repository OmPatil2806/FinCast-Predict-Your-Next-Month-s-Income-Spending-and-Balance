"""Goals page: savings goals with progress, projected date and an on-track check."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.auth.session import require_login  # noqa: E402
from app.components.cache import bump_cache, forecast as cached_forecast  # noqa: E402
from app.components.cards import format_euro  # noqa: E402
from app.components.goal_view import goal_summary, status_label  # noqa: E402
from app.components.layout import setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import create_goal, delete_goal, get_goals, get_transactions, update_goal  # noqa: E402
from src.forecasting.goals import HISTORY_MONTHS, project_goal, savings_pace  # noqa: E402
from src.forecasting.predictor import NotEnoughHistory, complete_history  # noqa: E402
from src.preprocessing.aggregate_monthly import aggregate_monthly  # noqa: E402

st.set_page_config(page_title="Goals · FinCast", page_icon="💶", layout="wide")
setup_database()


def forecast_savings(user_id: int) -> float | None:
    """Next month's forecast income - spending, or None when there is too little history."""
    try:
        forecast = cached_forecast(user_id)
    except (NotEnoughHistory, FileNotFoundError):
        return None
    return forecast.get("income")["predicted"] - forecast.get("expenses")["predicted"]


def add_goal_form(user_id: int, expanded: bool) -> None:
    with st.expander("➕ Add a goal", expanded=expanded):
        with st.form("add_goal", clear_on_submit=True):
            name = st.text_input("Name", placeholder="Emergency fund, holiday, new laptop…")
            left, middle, right = st.columns(3)
            target = left.number_input("Target (€)", min_value=0.0, step=100.0, value=1000.0)
            saved = middle.number_input("Already saved (€)", min_value=0.0, step=50.0, value=0.0)
            date = right.date_input("Target date (optional)", value=None, min_value=pd.Timestamp.today().date())
            if st.form_submit_button("Add goal"):
                try:
                    create_goal(user_id, name, target, saved, date)
                    bump_cache()
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))


def show_goal(user_id: int, goal: dict, pace: float) -> None:
    projection = project_goal(goal["target_amount"], goal["saved_amount"], pace, goal["target_date"])
    with st.container(border=True):
        title, badge = st.columns([3, 1], vertical_alignment="center")
        title.markdown(f"#### {goal['name']}")
        badge.markdown(f"**{status_label(projection.status)}**")
        lines = goal_summary(goal, projection)
        st.progress(projection.progress, text=lines[0])
        for line in lines[1:]:
            st.markdown(line)
        with st.expander("Edit or delete"):
            with st.form(f"edit_{goal['id']}"):
                name = st.text_input("Name", value=goal["name"])
                left, middle, right = st.columns(3)
                target = left.number_input("Target (€)", min_value=0.0, step=100.0, value=float(goal["target_amount"]))
                saved = middle.number_input("Already saved (€)", min_value=0.0, step=50.0,
                                            value=float(goal["saved_amount"]))
                current = pd.Timestamp(goal["target_date"]).date() if goal["target_date"] else None
                date = right.date_input("Target date (optional)", value=current)
                if st.form_submit_button("Save changes"):
                    try:
                        update_goal(user_id, goal["id"], name=name, target_amount=target, saved_amount=saved,
                                    target_date=date)
                        bump_cache()
                        st.rerun()
                    except ValueError as err:
                        st.error(str(err))
            sure = st.checkbox("Yes, delete this goal", key=f"sure_{goal['id']}")
            if st.button("Delete goal", key=f"delete_{goal['id']}", disabled=not sure):
                delete_goal(user_id, goal["id"])
                bump_cache()
                st.rerun()


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("Goals")
    st.caption("Set a savings goal and see when you will reach it at your current pace.")

    transactions = get_transactions(user_id)
    monthly = aggregate_monthly(complete_history(transactions)) if not transactions.empty else pd.DataFrame()
    pace = savings_pace(monthly, forecast_savings(user_id))
    st.metric("Your savings pace", f"{format_euro(pace, 0)} / month", border=True,
              help=f"Average of your last {HISTORY_MONTHS} complete months' savings (income − spending) and next "
                   "month's forecast savings.")
    if transactions.empty:
        st.caption("Import transactions to get a savings pace; until then goals show progress only.")

    goals = get_goals(user_id)
    add_goal_form(user_id, expanded=goals.empty)
    if goals.empty:
        return
    if len(goals) > 1:
        st.caption("Each goal is projected as if all your savings went to it; with several goals they share it.")
    for goal in goals.to_dict("records"):
        show_goal(user_id, goal, pace)


main()
show_footer()
