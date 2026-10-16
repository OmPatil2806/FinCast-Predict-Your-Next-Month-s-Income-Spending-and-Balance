"""Page parts shared by every Streamlit page: user sidebar, empty states, page links, disclaimer footer."""

from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitPageNotFoundError

from app.auth.session import current_username, is_logged_in, logout_user
from app.database.db import init_db

DISCLAIMER = "For educational purposes, not financial advice."
IMPORT_PAGE = "pages/2_Import_Transactions.py"
PAGES = [  # (file, label, icon, what it is for) - the order of the sidebar
    ("pages/1_Overview.py", "Overview", "📊", "Where your money stands: income, spending, savings, categories."),
    (IMPORT_PAGE, "Import transactions", "📥", "Upload a bank CSV (or the sample data) or add a transaction."),
    ("pages/3_Forecast.py", "Forecast", "🔮", "Next month's income, spending and balance, with a range."),
    ("pages/4_What_If.py", "What-If", "🎚️", "Change next month and see your balance move."),
    ("pages/5_Anomalies.py", "Anomalies", "🔎", "Unusual or duplicate charges to confirm or dismiss."),
    ("pages/6_Goals.py", "Goals", "🎯", "Savings goals and when you will reach them."),
    ("pages/7_Profile.py", "Profile", "👤", "Settings, password, download or delete your data."),
]


@st.cache_resource
def setup_database() -> Path:
    """Create the database and tables once per app start (safe to call from every page)."""
    return init_db()


def show_sidebar() -> None:
    """Show who is logged in and a Logout button (only when someone is logged in)."""
    if not is_logged_in():
        return
    with st.sidebar:
        st.markdown(f"Logged in as **{current_username()}**")
        if st.button("Logout", width="stretch"):
            logout_user()
            st.rerun()


def page_link(page: str, label: str, icon: str | None = None) -> None:
    """st.page_link that also works when a page is run on its own (tests, `streamlit run page.py`)."""
    try:
        st.page_link(page, label=label, icon=icon)
    except StreamlitPageNotFoundError:
        st.markdown(f"Open **{label}** in the sidebar.")


def empty_state(message: str) -> None:
    """The same 'nothing here yet' message on every page, with a link to Import."""
    st.info(message)
    page_link(IMPORT_PAGE, "Go to Import transactions", "📥")


def show_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
