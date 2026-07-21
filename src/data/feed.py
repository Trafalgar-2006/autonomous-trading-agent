"""
Market Data Feed — fetches OHLCV data from Alpaca.

Handles both US equities (StockHistoricalDataClient) and crypto
(CryptoHistoricalDataClient). Crypto symbols are identified by a "/" in the
ticker (e.g. "BTC/USD") and routed to the crypto client automatically.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..core.config import Config

logger = logging.getLogger(__name__)

# Map config timeframe strings to Alpaca TimeFrame
TIMEFRAME_MAP = {
    "1Min": TimeFrame.Minute,
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame.Hour,
    "1Day": TimeFrame.Day,
}


def is_crypto(symbol: str) -> bool:
    """Crypto pairs are written with a slash, e.g. BTC/USD."""
    return "/" in symbol


class MarketDataFeed:
    """Fetches market data from Alpaca (equities + crypto)."""

    def __init__(self):
        self.config = Config()
        api_key = self.config.alpaca_api_key
        secret_key = self.config.alpaca_secret_key

        # Crypto data is public on Alpaca — the client works with or without keys.
        self.crypto_client = CryptoHistoricalDataClient(api_key or None, secret_key or None)

        if api_key and api_key != "your_api_key_here":
            self.client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
            logger.info("MarketDataFeed initialized with Alpaca Data API")
        else:
            self.client = None
            logger.warning("MarketDataFeed: No API keys — equity data unavailable (crypto still works)")

    # ── Per-symbol extraction from a multi-index bars DataFrame ────
    @staticmethod
    def _extract(df: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        if df is None or df.empty:
            return result
        for symbol in symbols:
            try:
                if isinstance(df.index, pd.MultiIndex):
                    if symbol not in df.index.get_level_values(0):
                        continue
                    sdf = df.loc[symbol].copy()
                else:
                    sdf = df.copy()
                if sdf.empty:
                    continue
                sdf.index = pd.to_datetime(sdf.index)
                sdf.index = sdf.index.tz_localize(None) if sdf.index.tz else sdf.index
                sdf.columns = [c.lower() for c in sdf.columns]
                result[symbol] = sdf
            except Exception as e:
                logger.debug(f"No data for {symbol}: {e}")
        return result

    def _fetch_stock(self, symbols, tf, start) -> dict:
        if self.client is None or not symbols:
            return {}
        try:
            req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start)
            return self._extract(self.client.get_stock_bars(req).df, symbols)
        except Exception as e:
            logger.error(f"Error fetching stock data: {e}")
            return {}

    def _fetch_crypto(self, symbols, tf, start) -> dict:
        if not symbols:
            return {}
        try:
            req = CryptoBarsRequest(symbol_or_symbols=symbols, timeframe=tf, start=start)
            return self._extract(self.crypto_client.get_crypto_bars(req).df, symbols)
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return {}

    def get_bars(self, symbol: str, days: int = 365,
                 timeframe: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars for a single symbol (equity or crypto)."""
        data = self.get_bars_multi([symbol], days=days, timeframe=timeframe)
        return data.get(symbol)

    def get_bars_multi(self, symbols: list[str], days: int = 365,
                       timeframe: Optional[str] = None) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for multiple symbols, routing crypto vs equities."""
        tf = TIMEFRAME_MAP.get(timeframe or self.config.timeframe, TimeFrame.Day)
        start = datetime.utcnow() - timedelta(days=days)

        crypto_syms = [s for s in symbols if is_crypto(s)]
        equity_syms = [s for s in symbols if not is_crypto(s)]

        result: dict[str, pd.DataFrame] = {}
        result.update(self._fetch_stock(equity_syms, tf, start))
        result.update(self._fetch_crypto(crypto_syms, tf, start))

        if symbols:
            logger.info(f"Fetched data for {len(result)}/{len(symbols)} symbols")
        return result
