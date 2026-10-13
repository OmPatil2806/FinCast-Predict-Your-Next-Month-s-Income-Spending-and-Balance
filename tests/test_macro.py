"""Tests for the ECB macro loader. The HTTP responses below are small hand-written fixtures
in the ECB 'csvdata' format (not real figures), so the tests run without internet."""

import pandas as pd
import pytest
import requests

from src.ingestion import macro_api

INFLATION_CSV = """KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
ICP.M.DE.N.000000.4.ANR,M,DE,2024-01,3.1
ICP.M.DE.N.000000.4.ANR,M,DE,2024-02,2.7
ICP.M.DE.N.000000.4.ANR,M,DE,2024-03,2.3
"""
RATE_CSV = """KEY,FREQ,REF_AREA,TIME_PERIOD,OBS_VALUE
FM.B.U2.EUR.4F.KR.MRR_FR.LEV,B,U2,2024-01-02,4.5
FM.B.U2.EUR.4F.KR.MRR_FR.LEV,B,U2,2024-01-31,4.5
FM.B.U2.EUR.4F.KR.MRR_FR.LEV,B,U2,2024-02-29,4.25
"""


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def fake_get(url, params=None, timeout=None):
    return FakeResponse(INFLATION_CSV if "ICP" in url else RATE_CSV)


def offline_get(*args, **kwargs):
    raise requests.ConnectionError("no internet")


@pytest.fixture
def caches(tmp_path):
    return tmp_path / "raw" / "macro.csv", tmp_path / "sample" / "macro_cached.csv"


def test_parse_monthly_and_daily_series():
    inflation = macro_api.parse_ecb_csv(INFLATION_CSV)
    assert inflation.index.astype(str).tolist() == ["2024-01", "2024-02", "2024-03"]
    assert inflation.tolist() == [3.1, 2.7, 2.3]
    rate = macro_api.parse_ecb_csv(RATE_CSV)
    assert rate.tolist() == [4.5, 4.25]          # last business day of each month


def test_fetch_joins_series_and_carries_rate_forward(monkeypatch):
    monkeypatch.setattr(macro_api.requests, "get", fake_get)
    macro = macro_api.fetch_macro(start="2024-01")
    assert list(macro.columns) == macro_api.MACRO_COLUMNS
    assert macro["month"].astype(str).tolist() == ["2024-01", "2024-02", "2024-03"]
    assert macro["ecb_rate"].tolist() == [4.5, 4.25, 4.25]   # March has no rate row yet


def test_refresh_writes_both_caches(monkeypatch, caches):
    monkeypatch.setattr(macro_api.requests, "get", fake_get)
    macro = macro_api.load_macro(refresh=True, cache_paths=caches)
    assert all(path.exists() for path in caches)
    assert len(macro) == 3


def test_offline_falls_back_to_cache(monkeypatch, caches):
    monkeypatch.setattr(macro_api.requests, "get", fake_get)
    macro_api.load_macro(refresh=True, cache_paths=caches)
    caches[0].unlink()                          # only the committed copy is left
    monkeypatch.setattr(macro_api.requests, "get", offline_get)
    macro = macro_api.load_macro(refresh=True, cache_paths=caches)
    assert macro["inflation_de"].tolist() == [3.1, 2.7, 2.3]
    assert isinstance(macro["month"].dtype, pd.PeriodDtype)


def test_no_internet_and_no_cache_gives_clear_error(monkeypatch, caches):
    monkeypatch.setattr(macro_api.requests, "get", offline_get)
    with pytest.raises(macro_api.MacroDataUnavailable, match="Run `python -m src.ingestion.macro_api`"):
        macro_api.load_macro(refresh=True, cache_paths=caches)
