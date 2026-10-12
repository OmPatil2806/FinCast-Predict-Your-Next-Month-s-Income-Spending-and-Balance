"""Import page: upload a bank CSV (or use the sample), map columns, check and save;
or add a single transaction by hand."""

import datetime as dt
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.auth.session import require_login  # noqa: E402
from app.components.layout import setup_database, show_footer, show_sidebar  # noqa: E402
from app.database.crud import add_transactions, next_occurrence  # noqa: E402
from src.ingestion.bank_csv_parser import (  # noqa: E402
    STANDARD_COLUMNS, ParseError, guess_columns, parse_bank_csv, read_raw_csv,
)
from src.preprocessing.cleaning import clean_transactions  # noqa: E402
from src.utils.helpers import load_config, resolve_path  # noqa: E402

SAMPLE_FILE = "sample_bank_export.csv"
# Category names from the blueprint; Phase 5 moves them to config/categories.yaml.
CATEGORIES = [
    "Groceries", "Rent", "Utilities", "Transport", "Dining", "Leisure", "Health",
    "Shopping", "Subscriptions", "Savings", "Income", "Other",
]
REMOVAL_REASONS = {
    "missing_date": "missing date",
    "empty_description": "empty description",
    "missing_amount": "missing amount",
    "zero_amount": "zero amount",
    "future_date": "date in the future",
}

st.set_page_config(page_title="Import · FinCast", page_icon="💶", layout="wide")
setup_database()


def euro(value: float) -> str:
    return f"€{value:,.2f}"


def choose_file() -> None:
    """Remember the chosen file's bytes in the session (upload or sample)."""
    uploaded = st.file_uploader("Upload a bank export (.csv)", type=["csv"])
    if uploaded is not None and uploaded.file_id != st.session_state.get("import_upload_id"):
        st.session_state["import_upload_id"] = uploaded.file_id
        st.session_state["import_bytes"] = uploaded.getvalue()
        st.session_state["import_name"] = uploaded.name
        st.session_state["import_source"] = "csv"

    if st.button("Use sample data", help="A made-up 3-month bank export to try FinCast"):
        sample = resolve_path(load_config()["paths"]["data_sample"]) / SAMPLE_FILE
        st.session_state["import_bytes"] = sample.read_bytes()
        st.session_state["import_name"] = SAMPLE_FILE
        st.session_state["import_source"] = "demo"


def choose_columns(raw: pd.DataFrame, file_key: str) -> dict[str, str] | None:
    """Dropdowns for the date, description and amount columns, pre-filled by guess_columns."""
    st.subheader("2. Match the columns")
    guessed = guess_columns(list(raw.columns))
    options = [""] + list(raw.columns)
    mapping = {}
    for col, target in zip(st.columns(3), STANDARD_COLUMNS):
        with col:
            default = guessed.get(target, "")
            mapping[target] = st.selectbox(
                target.capitalize(), options, index=options.index(default), key=f"map_{target}_{file_key}"
            )
    if not all(mapping.values()):
        st.info("Choose a column for date, description and amount to continue.")
        return None
    if len(set(mapping.values())) < len(STANDARD_COLUMNS):
        st.warning("Each field needs a different column.")
        return None
    return mapping


def show_summary(clean: pd.DataFrame, report: dict) -> None:
    income = clean.loc[clean["amount"] > 0, "amount"].sum()
    spending = -clean.loc[clean["amount"] < 0, "amount"].sum()
    first, last = clean["date"].min(), clean["date"].max()
    st.markdown(
        f"**{len(clean)} rows ready** · {first:%d %b %Y} – {last:%d %b %Y} · "
        f"income {euro(income)} · spending {euro(spending)}"
    )
    skipped = {REMOVAL_REASONS[k]: v for k, v in report.items() if k in REMOVAL_REASONS and v}
    if skipped:
        details = ", ".join(f"{n} {reason}" for reason, n in skipped.items())
        st.caption(f"{report['removed_rows']} rows skipped: {details}")


def import_section(user_id: int) -> None:
    st.subheader("1. Choose a file")
    choose_file()

    data = st.session_state.get("import_bytes")
    if data is None:
        st.info("Upload a CSV export from your bank, or click **Use sample data** to try the app.")
        return
    file_key = hashlib.md5(data).hexdigest()[:8]
    st.caption(f"File: {st.session_state.get('import_name', 'uploaded file')}")

    try:
        raw, _ = read_raw_csv(data)
    except ParseError as err:
        st.error(str(err))
        return
    with st.expander("Raw file preview (first 10 rows)"):
        st.dataframe(raw.head(10), width="stretch", hide_index=True)

    mapping = choose_columns(raw, file_key)
    if mapping is None:
        return

    st.subheader("3. Check and save")
    try:
        parsed = parse_bank_csv(data, column_map=mapping)
    except ParseError as err:
        st.error(str(err))
        return
    clean, report = clean_transactions(parsed)
    if clean.empty:
        st.warning("No valid transactions left after cleaning.")
        return

    show_summary(clean, report)
    st.dataframe(
        clean[["date", "description", "amount"]],
        width="stretch", hide_index=True,
        column_config={
            "date": st.column_config.DateColumn("Date", format="DD.MM.YYYY"),
            "amount": st.column_config.NumberColumn("Amount (€)", format="%.2f"),
        },
    )

    if st.button("Save transactions", type="primary"):
        result = add_transactions(user_id, clean, source=st.session_state.get("import_source", "csv"))
        st.success(f"✅ {result['added']} added, {result['skipped']} already existed.")


def manual_entry_section(user_id: int) -> None:
    st.divider()
    st.subheader("Or add a transaction by hand")
    today = dt.date.today()
    with st.form("manual_entry", clear_on_submit=True):
        left, right = st.columns(2)
        with left:
            date = st.date_input("Date", value=today, max_value=today, format="DD.MM.YYYY")
            description = st.text_input("Description", placeholder="e.g. Coffee with Anna")
        with right:
            amount = st.number_input("Amount (€)", min_value=0.0, value=None, step=0.01, format="%.2f")
            kind = st.radio("Type", ["Expense", "Income"], horizontal=True)
        category = st.selectbox("Category (optional)", [""] + CATEGORIES)
        submitted = st.form_submit_button("Add transaction")

    if not submitted:
        return
    if not description.strip():
        st.error("Please enter a description.")
        return
    if not amount:
        st.error("Please enter an amount greater than 0.")
        return

    signed = -amount if kind == "Expense" else amount
    entry = pd.DataFrame({"date": [date], "description": [description], "amount": [signed]})
    clean, _ = clean_transactions(entry)
    if clean.empty:
        st.error("This transaction is not valid (check the date and amount).")
        return
    row = clean.iloc[0]
    clean["occurrence"] = next_occurrence(user_id, row["date"], row["description"], row["amount"])
    if category:
        clean["category"] = category
        clean["category_source"] = "user"
    add_transactions(user_id, clean, source="manual")
    sign = "−" if row["amount"] < 0 else "+"
    st.success(f"✅ Added: {row['description']}, {sign}{euro(abs(row['amount']))} on {row['date']:%d.%m.%Y}.")


def main() -> None:
    user_id = require_login()
    show_sidebar()
    st.title("Import transactions")
    import_section(user_id)
    manual_entry_section(user_id)


main()
show_footer()
