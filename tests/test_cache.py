"""Tests for the disk-cached market-data feed."""

from datetime import datetime

import numpy as np
import pandas as pd

from src.data.cache import CachedFeed


class FakeFeed:
    """Returns synthetic bars and counts how many times it's called."""

    def __init__(self, n=600):
        self.calls = 0
        self.n = n

    def get_bars_multi(self, symbols, days=365, timeframe="1Day"):
        self.calls += 1
        end = pd.Timestamp(datetime.utcnow().date())
        idx = pd.date_range(end=end, periods=self.n, freq="D")
        out = {}
        for s in symbols:
            close = 100 + np.cumsum(np.random.RandomState(hash(s) % 7).randn(self.n))
            out[s] = pd.DataFrame(
                {"open": close, "high": close + 1, "low": close - 1,
                 "close": close, "volume": 1e6},
                index=idx,
            )
        return out


def test_first_call_fetches_second_call_hits_cache(tmp_path):
    feed = FakeFeed()
    cf = CachedFeed(feed, cache_dir=str(tmp_path / "cache"))

    a = cf.get_bars_multi(["AAA", "BBB"], days=200)
    assert set(a.keys()) == {"AAA", "BBB"}
    assert feed.calls == 1

    # Second identical request must be served from disk — no new fetch.
    b = cf.get_bars_multi(["AAA", "BBB"], days=200)
    assert set(b.keys()) == {"AAA", "BBB"}
    assert feed.calls == 1  # unchanged


def test_slices_to_requested_window(tmp_path):
    feed = FakeFeed(n=600)
    cf = CachedFeed(feed, cache_dir=str(tmp_path / "cache"))
    data = cf.get_bars_multi(["AAA"], days=100)
    df = data["AAA"]
    span = (df.index.max() - df.index.min()).days
    assert span <= 101  # roughly the requested window, not the full 600


def test_offline_uses_cache_only(tmp_path):
    feed = FakeFeed()
    cdir = str(tmp_path / "cache")
    CachedFeed(feed, cache_dir=cdir).get_bars_multi(["AAA"], days=200)  # populate

    offline = CachedFeed(feed, cache_dir=cdir, offline=True)
    feed.calls = 0
    data = offline.get_bars_multi(["AAA"], days=200)
    assert "AAA" in data
    assert feed.calls == 0  # never hit the feed
