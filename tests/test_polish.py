"""Polish checks across the app (step 8.10): footer everywhere, shared empty states and page list."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.components.charts import tint
from app.components.layout import DISCLAIMER, PAGES
from src.forecasting.predictor import NotEnoughHistory
from src.utils.helpers import get_project_root

ROOT = get_project_root()
PAGE_FILES = sorted((ROOT / "dashboard" / "pages").glob("*.py"))


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.stem)
def test_logged_out_pages_show_login_link_and_disclaimer(page):
    at = AppTest.from_file(str(page), default_timeout=60).run()
    assert not at.exception
    assert [w.value for w in at.warning] == ["Please log in first."]
    assert DISCLAIMER in [c.value for c in at.caption]


def test_home_shows_disclaimer_when_logged_out():
    at = AppTest.from_file(str(ROOT / "dashboard" / "Home.py"), default_timeout=60).run()
    assert not at.exception and DISCLAIMER in [c.value for c in at.caption]


def test_page_list_matches_the_page_files():
    listed = {Path(page).name for page, *_ in PAGES}
    assert listed == {p.name for p in PAGE_FILES}


def test_every_page_ends_with_the_footer():
    for page in PAGE_FILES:
        assert page.read_text().rstrip().endswith("show_footer()"), page.name


def test_not_enough_history_messages():
    assert str(NotEnoughHistory(0, 3)).startswith("No complete month of transactions yet")
    assert "Import 1 more month " in str(NotEnoughHistory(2, 3))


def test_tint_derives_fills_from_the_series_colour():
    assert tint("#2a78d6", 0.16) == "rgba(42,120,214,0.16)"
