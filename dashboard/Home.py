"""FinCast entry point: login and registration. Run with: streamlit run dashboard/Home.py"""

import sys
from pathlib import Path

# Streamlit runs this file from dashboard/, so add the project root to the import path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.auth.auth_service import LOGIN_FAILED, AuthError, login, register  # noqa: E402
from app.auth.session import current_username, is_logged_in, login_user  # noqa: E402
from app.components.layout import setup_database, show_footer, show_sidebar  # noqa: E402

st.set_page_config(page_title="FinCast", page_icon="💶", layout="centered")
setup_database()


def show_login_form() -> None:
    with st.form("login_form"):
        identifier = st.text_input("Username or email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", width="stretch")
    if submitted:
        user = login(identifier, password)
        if user is None:
            st.error(LOGIN_FAILED)
        else:
            login_user(user)
            st.rerun()


def show_register_form() -> None:
    with st.form("register_form"):
        username = st.text_input("Username", help="3-30 characters: letters, numbers or _")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password", help="At least 8 characters")
        confirm = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button("Create account", width="stretch")
    if submitted:
        if password != confirm:
            st.error("Passwords do not match.")
            return
        try:
            user_id = register(username, email, password)
        except AuthError as err:
            st.error(str(err))
            return
        login_user({"id": user_id, "username": username.strip()})
        st.rerun()


st.title("💶 FinCast")
st.caption("Predict your next month's income, spending and balance.")

if is_logged_in():
    show_sidebar()
    st.success(f"Welcome, {current_username()}!")
    st.markdown(
        "Use the pages in the sidebar to see your **Overview**. "
        "Import, forecasts, what-if scenarios and goals arrive in the next phases."
    )
else:
    login_tab, register_tab = st.tabs(["Login", "Register"])
    with login_tab:
        show_login_form()
    with register_tab:
        show_register_form()

show_footer()
