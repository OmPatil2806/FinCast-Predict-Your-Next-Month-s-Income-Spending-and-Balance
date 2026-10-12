"""Page parts shared by every Streamlit page: user sidebar and disclaimer footer."""

from pathlib import Path

import streamlit as st

from app.auth.session import current_username, is_logged_in, logout_user
from app.database.db import init_db

DISCLAIMER = "For educational purposes, not financial advice."


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


def show_footer() -> None:
    st.divider()
    st.caption(DISCLAIMER)
