"""Overview page: where am I now? Placeholder until Phase 3 adds KPIs and charts."""

import sys
from pathlib import Path

# Pages live in dashboard/pages/, so the project root is two folders up.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app.auth.session import current_username, require_login  # noqa: E402
from app.components.layout import show_footer, show_sidebar  # noqa: E402

st.set_page_config(page_title="Overview · FinCast", page_icon="💶", layout="wide")

require_login()
show_sidebar()

st.title("Overview")
st.write(f"Hello, **{current_username()}**!")
st.info("Your income and spending charts will appear here once transaction import is built in Phase 3.")

show_footer()
