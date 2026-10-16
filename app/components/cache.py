"""One place for caching in the Streamlit app (step 8.9).

    st.cache_resource  models, loaded once per server process and shared by every page and user:
                       forecasters(), detector(), categorizer()
    st.cache_data      per-user results with a time-to-live, keyed by cache_key(user_id):
                       forecast(), anomaly_flags(), data_zip()

cache_key(user_id) = (data fingerprint from the database, this session's write counter). The fingerprint
catches new imports, verdicts and settings even from another browser tab; the counter is bumped by
bump_cache() after every write this session makes (import, category edit, goal, settings ...), which also
covers changes the fingerprint cannot see, such as renaming a category.
"""

import streamlit as st

from app.database.crud import data_version
from src.forecasting.anomalies import load_detector, review_list
from src.forecasting.predictor import forecast_user, load_forecasters

TTL_SECONDS = 600
EPOCH_KEY = "cache_epoch"


# ---------- models (shared, loaded once) ----------

@st.cache_resource(show_spinner="Loading the forecasting models…")
def forecasters():
    return load_forecasters()


@st.cache_resource(show_spinner=False)
def detector():
    return load_detector()


@st.cache_resource(show_spinner=False)
def categorizer():
    from src.preprocessing.categorizer import load_categorizer
    return load_categorizer()


# ---------- per-user results ----------

def cache_key(user_id: int) -> tuple:
    return data_version(user_id), st.session_state.get(EPOCH_KEY, 0)


def bump_cache() -> None:
    """Call after this session changed the user's data: the next read recomputes."""
    st.session_state[EPOCH_KEY] = st.session_state.get(EPOCH_KEY, 0) + 1


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _forecast(user_id: int, key: tuple):
    return forecast_user(user_id, forecasters=forecasters())


def forecast(user_id: int):
    """Next month's forecast (raises NotEnoughHistory / FileNotFoundError like forecast_user)."""
    return _forecast(user_id, cache_key(user_id))


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _anomaly_flags(user_id: int, key: tuple, since, include_dismissed: bool):
    from app.database.crud import get_transactions
    return review_list(get_transactions(user_id), detector(), since=since, include_dismissed=include_dismissed)


def anomaly_flags(user_id: int, since=None, include_dismissed: bool = False):
    return _anomaly_flags(user_id, cache_key(user_id), since, include_dismissed)


@st.cache_data(ttl=TTL_SECONDS, show_spinner=False)
def _data_zip(user_id: int, key: tuple) -> bytes:
    from app.database.export import user_data_zip
    return user_data_zip(user_id)


def data_zip(user_id: int) -> bytes:
    return _data_zip(user_id, cache_key(user_id))
