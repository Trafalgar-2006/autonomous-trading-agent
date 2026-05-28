"""
Feature Engine — computes technical indicators for strategy analysis.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """Computes technical indicators on OHLCV DataFrames."""

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical features on a DataFrame."""
        if df is None or df.empty:
            return df

        df = df.copy()

        try:
            df = self._add_moving_averages(df)
            df = self._add_rsi(df)
            df = self._add_macd(df)
            df = self._add_bollinger_bands(df)
            df = self._add_atr(df)
            df = self._add_adx(df)
            df = self._add_volume_features(df)
            df = self._add_donchian_channels(df)
            df = self._add_zscore(df)
            df = self._add_returns(df)
            df = self._add_volatility(df)
        except Exception as e:
            logger.error(f"Error computing features: {e}")

        return df

    @staticmethod
    def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        df["ema_9"] = close.ewm(span=9, adjust=False).mean()
        df["ema_21"] = close.ewm(span=21, adjust=False).mean()
        df["sma_50"] = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()
        return df

    @staticmethod
    def _add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _add_macd(df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def _add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        sma = df["close"].rolling(period).mean()
        rolling_std = df["close"].rolling(period).std()
        df["bb_upper"] = sma + (rolling_std * std)
        df["bb_middle"] = sma
        df["bb_lower"] = sma - (rolling_std * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        return df

    @staticmethod
    def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = tr.rolling(period).mean()
        df["atr_pct"] = df["atr"] / close
        return df

    @staticmethod
    def _add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        df["adx"] = dx.rolling(period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        return df

    @staticmethod
    def _add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
        df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        return df

    @staticmethod
    def _add_donchian_channels(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        df["donchian_high"] = df["high"].rolling(period).max()
        df["donchian_low"] = df["low"].rolling(period).min()
        df["donchian_mid"] = (df["donchian_high"] + df["donchian_low"]) / 2
        return df

    @staticmethod
    def _add_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        mean = df["close"].rolling(period).mean()
        std = df["close"].rolling(period).std()
        df["zscore"] = (df["close"] - mean) / std.replace(0, np.nan)
        return df

    @staticmethod
    def _add_returns(df: pd.DataFrame) -> pd.DataFrame:
        df["return_1d"] = df["close"].pct_change(1)
        df["return_5d"] = df["close"].pct_change(5)
        df["return_20d"] = df["close"].pct_change(20)
        return df

    @staticmethod
    def _add_volatility(df: pd.DataFrame) -> pd.DataFrame:
        returns = df["close"].pct_change()
        df["volatility_20d"] = returns.rolling(20).std() * np.sqrt(252)
        df["volatility_60d"] = returns.rolling(60).std() * np.sqrt(252)
        return df
