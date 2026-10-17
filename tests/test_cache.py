"""Caching (step 8.9): the cache key follows every change a forecast or anomaly check depends on."""

import re

import pandas as pd

from app.auth.auth_service import register
from app.database.crud import add_transactions, data_version, get_transactions, set_anomaly_status, update_profile
from app.database.db import init_db
from src.utils.helpers import get_project_root


def test_data_version_changes_with_the_data(tmp_path):
    db = init_db(tmp_path / "app.db")
    user = register("om_patil", "om@example.com", "secret123", db_path=db)
    other = register("other", "other@example.com", "secret123", db_path=db)
    versions = [data_version(user, db_path=db)]

    add_transactions(user, pd.DataFrame({"date": ["2026-09-01"], "description": ["Shop"], "amount": [-10.0]}), db_path=db)
    versions.append(data_version(user, db_path=db))
    tid = int(get_transactions(user, db_path=db)["id"].iloc[0])
    set_anomaly_status(user, tid, "dismissed", db_path=db)
    versions.append(data_version(user, db_path=db))
    update_profile(user, opening_balance=500, db_path=db)
    versions.append(data_version(user, db_path=db))
    update_profile(user, income_type="freelance", db_path=db)
    versions.append(data_version(user, db_path=db))
    assert len(set(versions)) == len(versions)                     # every change gives a new key

    before = data_version(user, db_path=db)
    add_transactions(other, pd.DataFrame({"date": ["2026-09-01"], "description": ["X"], "amount": [-1.0]}), db_path=db)
    assert data_version(user, db_path=db) == before                # other users do not invalidate it


def test_models_are_only_loaded_through_the_shared_cache():
    pages = (get_project_root() / "dashboard").rglob("*.py")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        assert not re.search(r"load_(forecasters|detector|categorizer)\(", text), page.name
        assert "st.cache_data.clear()" not in text, page.name      # bump_cache() instead of clearing everyone
