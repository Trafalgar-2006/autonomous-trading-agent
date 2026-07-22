"""
Deep-history data feed (Yahoo Finance via yfinance).

Alpaca's free feed only reaches ~2019, which excludes the 2008-2009 momentum
crash — the exact failure mode a momentum strategy must be stress-tested
against. This feed pulls split/dividend-adjusted daily bars back to ~2000 so
the research harness can backtest through real crashes.

RESEARCH ONLY. It is deliberately not wired into the live trading loop (Yahoo
is fine for backtests but not a dependency you want between you and a live
order). Same `get_bars_multi` / `get_bars` interface as MarketDataFeed, so it
drops into CachedFeed and WalkForward unchanged.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class HistoryFeed:
    """Yahoo-Finance-backed feed for long-history research."""

    class _Cfg:
        timeframe = "1Day"

    def __init__(self):
        self.config = self._Cfg()

    @staticmethod
    def _normalize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Coerce a yfinance frame into our lowercase OHLCV shape."""
        if df is None or df.empty:
            return None
        df = df.copy()
        # A single-symbol download can still return a 2-level column index.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        rename = {"adj close": "close"}
        df = df.rename(columns=rename)
        needed = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in needed):
            return None
        df = df[needed]
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        return df.dropna()

    def get_bars_multi(self, symbols: list[str], days: int = 365,
                       timeframe: Optional[str] = None) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
        # yfinance uses '-' for crypto pairs (BTC-USD), not '/'.
        yf_symbols = [s.replace("/", "-") for s in symbols]
        sym_map = dict(zip(yf_symbols, symbols, strict=False))

        result: dict[str, pd.DataFrame] = {}
        try:
            raw = yf.download(yf_symbols, start=start, interval="1d",
                              auto_adjust=True, progress=False, group_by="ticker",
                              threads=True)
        except Exception as e:
            logger.error(f"yfinance download failed: {e}")
            return {}

        if raw is None or raw.empty:
            return {}

        # Multi-symbol: top column level is the ticker. Single symbol: flat.
        if isinstance(raw.columns, pd.MultiIndex):
            for yf_sym in yf_symbols:
                if yf_sym in raw.columns.get_level_values(0):
                    df = self._normalize(raw[yf_sym])
                    if df is not None and not df.empty:
                        result[sym_map[yf_sym]] = df
        else:
            df = self._normalize(raw)
            if df is not None and not df.empty:
                result[symbols[0]] = df

        logger.info(f"HistoryFeed: fetched {len(result)}/{len(symbols)} symbols "
                    f"from {start}")
        return result

    def get_bars(self, symbol: str, days: int = 365,
                 timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        return self.get_bars_multi([symbol], days=days, timeframe=timeframe).get(symbol)
