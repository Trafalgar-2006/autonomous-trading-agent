"""
Market Data Feed — fetches OHLCV data from Alpaca.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
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


class MarketDataFeed:
    """Fetches market data from Alpaca Data API."""

    def __init__(self):
        self.config = Config()
        api_key = self.config.alpaca_api_key
        secret_key = self.config.alpaca_secret_key

        if api_key and api_key != "your_api_key_here":
            self.client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=secret_key,
            )
            logger.info("MarketDataFeed initialized with Alpaca Data API")
        else:
            self.client = None
            logger.warning("MarketDataFeed: No API keys — data feed unavailable")

    def get_bars(
        self,
        symbol: str,
        days: int = 365,
        timeframe: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars for a single symbol."""
        if self.client is None:
            return None

        tf = TIMEFRAME_MAP.get(timeframe or self.config.timeframe, TimeFrame.Day)
        start = datetime.utcnow() - timedelta(days=days)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
            )
            bars = self.client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None

            # If multi-index (symbol, timestamp), select the symbol level
            if isinstance(df.index, pd.MultiIndex):
                df = df.loc[symbol]

            # Ensure index is DatetimeIndex
            df.index = pd.to_datetime(df.index)
            df.index = df.index.tz_localize(None) if df.index.tz else df.index

            # Standard column names
            df.columns = [c.lower() for c in df.columns]

            logger.debug(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_bars_multi(
        self,
        symbols: list[str],
        days: int = 365,
        timeframe: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for multiple symbols."""
        if self.client is None:
            return {}

        tf = TIMEFRAME_MAP.get(timeframe or self.config.timeframe, TimeFrame.Day)
        start = datetime.utcnow() - timedelta(days=days)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=tf,
                start=start,
            )
            bars = self.client.get_stock_bars(request)
            df = bars.df

            if df.empty:
                logger.warning("No data returned for any symbols")
                return {}

            result = {}
            for symbol in symbols:
                try:
                    if isinstance(df.index, pd.MultiIndex):
                        symbol_df = df.loc[symbol].copy()
                    else:
                        symbol_df = df[df.index.get_level_values(0) == symbol].copy()

                    if symbol_df.empty:
                        continue

                    symbol_df.index = pd.to_datetime(symbol_df.index)
                    symbol_df.index = (
                        symbol_df.index.tz_localize(None)
                        if symbol_df.index.tz
                        else symbol_df.index
                    )
                    symbol_df.columns = [c.lower() for c in symbol_df.columns]
                    result[symbol] = symbol_df

                except Exception as e:
                    logger.debug(f"No data for {symbol}: {e}")

            logger.info(f"Fetched data for {len(result)}/{len(symbols)} symbols")
            return result

        except Exception as e:
            logger.error(f"Error fetching multi-symbol data: {e}")
            return {}
