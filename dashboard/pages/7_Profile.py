"""Profile page: settings, password, download all data, delete the account."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.auth.auth_service import AuthError, change_password  # noqa: E402
from app.auth.session import current_username, logout_user, require_login  # noqa: E402
from app.components.layout import setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import count_user_rows, delete_user, get_profile, get_user_by_id, update_profile  # noqa: E402
from app.database.export import user_data_zip  # noqa: E402

INCOME_TYPES = {"salary": "Salary (regular pay)", "freelance": "Freelance / self-employed"}

st.set_page_config(page_title="Profile · FinCast", page_icon="💶", layout="centered")
setup_database()


def show_settings(user_id: int) -> None:
    profile = get_profile(user_id) or {}
    st.subheader("Settings")
    with st.form("profile"):
        income_type = st.radio("How do you earn your income?", list(INCOME_TYPES), format_func=INCOME_TYPES.get,
                               index=list(INCOME_TYPES).index(profile.get("income_type", "salary")), horizontal=True)
        opening = st.number_input("Opening balance (€)", value=float(profile.get("opening_balance") or 0.0), step=100.0,
                                  help="Your account balance before the first imported transaction. "
                                       "The balance forecast starts from it.")
        st.text_input("Currency", value=profile.get("currency", "EUR"), disabled=True,
                      help="FinCast works in euro (Germany).")
        if st.form_submit_button("Save settings"):
            update_profile(user_id, income_type=income_type, opening_balance=opening)
            st.cache_data.clear()                      # forecasts on other pages use the opening balance
            st.success("Settings saved.")


def show_password(user_id: int) -> None:
    st.subheader("Change password")
    with st.form("password", clear_on_submit=True):
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password", help="At least 8 characters")
        confirm = st.text_input("Confirm new password", type="password")
        if st.form_submit_button("Change password"):
            if new != confirm:
                st.error("The new passwords do not match.")
                return
            try:
                change_password(user_id, current, new)
                st.success("Password changed.")
            except AuthError as err:
                st.error(str(err))


def show_download(user_id: int) -> None:
    st.subheader("Your data")
    counts = count_user_rows(user_id)
    st.caption(f"Stored for you: {counts['transactions']:,} transactions, {counts['forecasts']:,} forecast values, "
               f"{counts['goals']:,} goals.")
    st.download_button("Download all my data (ZIP of CSV files)", data=user_data_zip(user_id),
                       file_name=f"fincast_{current_username()}_{pd.Timestamp.today():%Y-%m-%d}.zip",
                       mime="application/zip", icon="⬇️")


def show_delete(user_id: int) -> None:
    username = (get_user_by_id(user_id) or {}).get("username", "")
    st.subheader("Delete account")
    with st.container(border=True):
        st.markdown("This permanently deletes your account and **all** your data: transactions, forecasts, "
                    "goals and settings. It cannot be undone. Download your data first if you want to keep it.")
        typed = st.text_input(f"Type your username **{username}** to confirm", key="delete_confirm")
        if st.button("Delete my account and all data", type="primary", disabled=typed.strip() != username):
            delete_user(user_id)
            logout_user()
            st.cache_data.clear()
            st.session_state["account_deleted"] = True
            st.rerun()


def main() -> None:
    if st.session_state.pop("account_deleted", False):
        st.title("Profile")
        st.success("Your account and all its data have been deleted.")
        return
    user_id = require_login()
    show_sidebar()
    st.title("Profile")
    st.caption(f"Logged in as {current_username()}.")
    show_settings(user_id)
    show_password(user_id)
    show_download(user_id)
    show_delete(user_id)


main()
show_footer()
