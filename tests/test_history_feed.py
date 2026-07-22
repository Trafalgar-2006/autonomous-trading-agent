"""Tests for the deep-history feed's normalization (no network)."""

import pandas as pd

from src.data.history_feed import HistoryFeed


def test_normalize_flat_columns():
    df = pd.DataFrame(
        {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5],
         "Volume": [100]},
        index=pd.to_datetime(["2008-10-01"]),
    )
    out = HistoryFeed._normalize(df)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert out["close"].iloc[0] == 1.5


def test_normalize_multiindex_columns():
    cols = pd.MultiIndex.from_product([["Open", "High", "Low", "Close", "Volume"],
                                       ["AAPL"]])
    df = pd.DataFrame([[1.0, 2.0, 0.5, 1.5, 100]],
                      index=pd.to_datetime(["2008-10-01"]), columns=cols)
    out = HistoryFeed._normalize(df)
    assert set(["open", "high", "low", "close", "volume"]).issubset(out.columns)


def test_normalize_drops_incomplete():
    df = pd.DataFrame({"Open": [1.0], "Close": [1.5]},
                      index=pd.to_datetime(["2008-10-01"]))
    assert HistoryFeed._normalize(df) is None


def test_normalize_empty_is_none():
    assert HistoryFeed._normalize(pd.DataFrame()) is None
