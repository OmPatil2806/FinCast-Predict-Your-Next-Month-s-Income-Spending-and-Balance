"""Login state for the Streamlit app, kept in st.session_state."""

import streamlit as st

SESSION_KEYS = ("user_id", "username")


def login_user(user: dict) -> None:
    """Remember the logged-in user (a dict from auth_service.login)."""
    st.session_state["user_id"] = user["id"]
    st.session_state["username"] = user["username"]


def logout_user() -> None:
    """Forget the logged-in user."""
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def current_user_id() -> int | None:
    return st.session_state.get("user_id")


def current_username() -> str | None:
    return st.session_state.get("username")


def is_logged_in() -> bool:
    return current_user_id() is not None


def require_login() -> int:
    """Stop the page with a warning if nobody is logged in; otherwise return the user id.

    Call this at the top of every page except Home.
    """
    if not is_logged_in():
        st.warning("Please log in first.")
        st.stop()
    return current_user_id()
