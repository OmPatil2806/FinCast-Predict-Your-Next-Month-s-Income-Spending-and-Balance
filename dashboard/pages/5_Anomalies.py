"""Anomalies page: unusual expenses, most unusual first, with Confirm / Dismiss."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.auth.session import require_login  # noqa: E402
from app.components.cache import anomaly_flags, bump_cache  # noqa: E402
from app.components.cards import format_euro  # noqa: E402
from app.components.layout import empty_state, setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import get_transactions, set_anomaly_status  # noqa: E402
from src.forecasting.anomalies import history_note  # noqa: E402

PERIODS = {"Last 3 months": 3, "Last 6 months": 6, "Last 12 months": 12, "All time": None}

st.set_page_config(page_title="Anomalies · FinCast", page_icon="💶", layout="wide")
setup_database()


def set_status(user_id: int, transaction_id: int, status: str | None) -> None:
    set_anomaly_status(user_id, transaction_id, status)
    bump_cache()


def show_flag(user_id: int, row: pd.Series) -> None:
    with st.container(border=True):
        info, actions = st.columns([4, 1.4], vertical_alignment="center")
        status = row["anomaly_status"]
        badge = {"confirmed": " · ✅ confirmed", "dismissed": " · dismissed"}.get(status, "")
        info.markdown(f"**{row['description']}** · {format_euro(row['amount'])}{badge}")
        info.caption(f"{row['date']:%d.%m.%Y} · {row['category'] or 'Uncategorised'} · {row['reason']}")
        tid = int(row["id"])
        if status in ("confirmed", "dismissed"):
            actions.button("Undo", key=f"undo_{tid}", on_click=set_status, args=(user_id, tid, None),
                           width="stretch")
        else:
            confirm, dismiss = actions.columns(2)
            confirm.button("Confirm", key=f"confirm_{tid}", on_click=set_status, args=(user_id, tid, "confirmed"),
                           help="Yes, this is unusual - keep it on the list.", width="stretch")
            dismiss.button("Dismiss", key=f"dismiss_{tid}", on_click=set_status, args=(user_id, tid, "dismissed"),
                           help="This is fine - hide it.", width="stretch")


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("Anomalies")
    st.caption("Expenses that look unusual compared with your own history: much larger than normal for the "
               "category or the merchant, or the same charge twice within a few days.")

    transactions = get_transactions(user_id)
    if transactions.empty:
        empty_state("No transactions yet. Import a bank CSV to check it for unusual spending.")
        return
    note = history_note(transactions)
    if note:
        st.info(note)

    left, right = st.columns([2, 1], vertical_alignment="bottom")
    period = left.selectbox("Period", list(PERIODS), index=2)
    show_dismissed = right.toggle("Show dismissed", value=False)
    months = PERIODS[period]
    since = None if months is None else (pd.Timestamp.today().to_period("M") - (months - 1)).to_timestamp()

    try:
        with st.spinner("Checking your transactions…"):
            flags = anomaly_flags(user_id, since=since, include_dismissed=show_dismissed)
    except FileNotFoundError:
        st.error("The anomaly model is missing. Run `python pipelines/run_training.py` once.")
        return

    open_flags = int(flags["anomaly_status"].isna().sum() + (flags["anomaly_status"] == "flagged").sum())
    confirmed = int((flags["anomaly_status"] == "confirmed").sum())
    a, b = st.columns(2)
    a.metric("To review", open_flags, border=True)
    b.metric("Confirmed", confirmed, border=True)

    if flags.empty:
        st.success(f"Nothing unusual found ({period.lower()}).")
        return
    for _, row in flags.iterrows():
        show_flag(user_id, row)


main()
show_footer()
