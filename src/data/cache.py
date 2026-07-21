"""
Cached market-data feed — a disk cache in front of MarketDataFeed.

Research and walk-forward sweeps fetch the same history over and over. This
wrapper stores each symbol's bars on disk (pickled DataFrames) and serves
subsequent requests from disk, only hitting the API for cache misses or stale
data. It is intentionally NOT used by the live trading loop (which must always
pull fresh bars) — it is a research accelerator.

Usage:
    from src.data.feed import MarketDataFeed
    from src.data.cache import CachedFeed
    feed = CachedFeed(MarketDataFeed())
    data = feed.get_bars_multi(["AAPL", "MSFT"], days=365)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CachedFeed:
    """Disk-cached wrapper around a MarketDataFeed-like object."""

    def __init__(
        self,
        feed,
        cache_dir: str = "data/cache",
        refresh_after_days: int = 1,
        offline: bool = False,
    ):
        """
        Args:
            feed:               underlying feed with get_bars_multi(symbols, days, timeframe).
            cache_dir:          where pickled bars live.
            refresh_after_days: re-fetch if the newest cached bar is older than
                                this many days (weekends tolerated with a buffer).
            offline:            if True, never hit the API — serve cache only.
        """
        self.feed = feed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refresh_after_days = refresh_after_days
        self.offline = offline
        self.fetch_count = 0  # for observability / tests

    def _default_timeframe(self) -> str:
        return getattr(getattr(self.feed, "config", None), "timeframe", "1Day")

    def _path(self, symbol: str, timeframe: str) -> Path:
        safe = symbol.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe}__{timeframe}.pkl"

    def _load(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._path(symbol, timeframe)
        if not path.exists():
            return None
        try:
            df = pd.read_pickle(path)
            return df if isinstance(df, pd.DataFrame) and not df.empty else None
        except Exception as e:
            logger.debug(f"Cache read failed for {symbol}: {e}")
            return None

    def _save(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        try:
            df.to_pickle(self._path(symbol, timeframe))
        except Exception as e:
            logger.warning(f"Cache write failed for {symbol}: {e}")

    def get_bars_multi(
        self,
        symbols: list[str],
        days: int = 365,
        timeframe: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Return bars for each symbol, served from cache where possible."""
        tf = timeframe or self._default_timeframe()
        today = pd.Timestamp(datetime.utcnow().date())
        needed_start = today - pd.Timedelta(days=days)
        # weekend/holiday buffer on top of the freshness window
        stale_before = today - timedelta(days=self.refresh_after_days + 4)

        cached: dict[str, pd.DataFrame] = {}
        misses: list[str] = []

        for sym in symbols:
            df = self._load(sym, tf)
            if df is not None:
                covers_start = df.index.min() <= needed_start
                fresh_enough = df.index.max() >= stale_before
                if covers_start and (fresh_enough or self.offline):
                    cached[sym] = df
                    continue
            misses.append(sym)

        if misses and not self.offline:
            # Fetch a generous history so the cache is reusable for longer windows.
            fetch_days = max(days, 500)
            self.fetch_count += 1
            fetched = self.feed.get_bars_multi(misses, days=fetch_days, timeframe=tf)
            for sym, df in fetched.items():
                existing = self._load(sym, tf)
                if existing is not None:
                    df = pd.concat([existing, df])
                    df = df[~df.index.duplicated(keep="last")].sort_index()
                self._save(sym, tf, df)
                cached[sym] = df

        # Slice everything down to the requested window.
        result: dict[str, pd.DataFrame] = {}
        for sym, df in cached.items():
            sliced = df[df.index >= needed_start]
            if not sliced.empty:
                result[sym] = sliced
        return result

    def get_bars(
        self,
        symbol: str,
        days: int = 365,
        timeframe: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Convenience single-symbol fetch."""
        data = self.get_bars_multi([symbol], days=days, timeframe=timeframe)
        return data.get(symbol)
