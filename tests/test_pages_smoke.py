"""Every page renders without errors for a logged-in user with real-looking data (step 9.4).

The app database is a temporary file (FINCAST_CONFIG points database.path at it); the models are the
committed ones. Covers the parts only a running page reaches: KPI cards, charts, layout helpers, caching.
"""

import copy

import pandas as pd
import pytest
import yaml
from streamlit.testing.v1 import AppTest

from app.auth.auth_service import register
from app.database.crud import add_transactions, create_goal
from app.database.db import init_db
from src.ingestion.synthetic_generator import generate_dataset
from src.utils.helpers import CONFIG_ENV, get_project_root, load_config

ROOT = get_project_root()
PAGES = ["Home.py"] + sorted(p.name for p in (ROOT / "dashboard" / "pages").glob("*.py"))


@pytest.fixture(scope="module")
def app_db(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("app")
    config = copy.deepcopy(load_config())
    config["database"]["path"] = str(tmp / "app.db")
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    patch = pytest.MonkeyPatch()
    patch.setenv(CONFIG_ENV, str(path))
    load_config.cache_clear()

    db = init_db()
    users = {}
    transactions, _ = generate_dataset(n_users=1, months=14, start="2025-08-01", seed=12)
    mine = transactions.drop(columns=["user_id", "archetype"])
    mine["occurrence"] = mine.groupby(["date", "description", "amount"]).cumcount() + 1
    users["full"] = register("smoke_full", "full@example.com", "secret123")
    add_transactions(users["full"], mine)
    create_goal(users["full"], "Holiday", 2000, 500, "2027-06-30")
    users["empty"] = register("smoke_empty", "empty@example.com", "secret123")
    yield db, users
    patch.undo()
    load_config.cache_clear()


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("who", ["full", "empty"])
def test_page_renders(app_db, page, who):
    _, users = app_db
    file = ROOT / "dashboard" / (page if page == "Home.py" else f"pages/{page}")
    at = AppTest.from_file(str(file), default_timeout=120)
    at.session_state["user_id"] = users[who]
    at.session_state["username"] = f"smoke_{who}"
    at.run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert "For educational purposes, not financial advice." in [c.value for c in at.caption]
    if who == "full" and page == "1_Overview.py":
        assert [m.label for m in at.metric] == ["Income", "Spending", "Savings", "Savings rate"]
    if who == "full" and page == "3_Forecast.py":
        assert [m.label for m in at.metric][:3] == ["Income", "Spending", "Balance at month end"]
