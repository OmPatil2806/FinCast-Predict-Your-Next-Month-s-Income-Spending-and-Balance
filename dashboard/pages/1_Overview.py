"""Overview page: where am I now? KPIs, 12-month trend, savings, categories, recent transactions."""

import sys
from pathlib import Path

# Pages live in dashboard/pages/, so the project root is two folders up.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from streamlit.errors import StreamlitPageNotFoundError  # noqa: E402

from app.auth.session import current_username, require_login  # noqa: E402
from app.components.cards import kpi_row  # noqa: E402
from app.components.charts import (  # noqa: E402
    category_bar_chart, income_vs_spending_chart, monthly_savings_bar_chart,
)
from app.components.layout import setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import get_transactions  # noqa: E402
from src.preprocessing.aggregate_monthly import (  # noqa: E402
    UNCATEGORISED, aggregate_by_category, aggregate_monthly,
)

TREND_MONTHS = 12
RECENT_ROWS = 10

st.set_page_config(page_title="Overview · FinCast", page_icon="💶", layout="wide")
setup_database()


def show_empty_state() -> None:
    st.info("No transactions yet. Import a bank CSV (or the sample data) to see your overview.")
    try:
        st.page_link("pages/2_Import_Transactions.py", label="Go to Import transactions", icon="📥")
    except StreamlitPageNotFoundError:  # page run on its own, outside the multipage app
        st.markdown("Open **Import Transactions** in the sidebar.")


def show_category_section(window: pd.DataFrame) -> None:
    st.subheader("Spending by category")
    by_category = aggregate_by_category(window)
    if by_category.empty:
        st.caption("No spending in this period.")
    elif (by_category["category"] == UNCATEGORISED).all():
        st.caption(
            "Your transactions don't have categories yet. Automatic categories arrive in Phase 5; "
            "you can already set one when adding a transaction by hand."
        )
    else:
        st.plotly_chart(category_bar_chart(by_category), width="stretch")


def show_recent(transactions: pd.DataFrame) -> None:
    st.subheader("Recent transactions")
    recent = transactions.head(RECENT_ROWS)[["date", "description", "amount", "category"]].copy()
    recent["category"] = recent["category"].fillna("—")
    st.dataframe(
        recent, width="stretch", hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
            "description": "Description",
            "amount": st.column_config.NumberColumn("Amount (€)", format="%.2f"),
            "category": "Category",
        },
    )


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("Overview")
    st.caption(f"Hello, {current_username()} — here is where your money stands.")

    transactions = get_transactions(user_id)
    if transactions.empty:
        show_empty_state()
        return

    monthly = aggregate_monthly(transactions).tail(TREND_MONTHS)
    first_month = monthly["month"].iloc[0]
    window = transactions[transactions["date"].dt.to_period("M") >= first_month]

    kpi_row(monthly)

    st.subheader("Income vs spending")
    st.plotly_chart(income_vs_spending_chart(monthly), width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Savings per month")
        st.plotly_chart(monthly_savings_bar_chart(monthly), width="stretch")
    with right:
        show_category_section(window)

    show_recent(transactions)


main()
show_footer()
