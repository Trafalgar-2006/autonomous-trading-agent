"""
Mean Reversion Strategy — trades price returning to the mean.

Signals based on:
- Bollinger Band extremes
- RSI overbought/oversold
- Z-score deviation
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..core.models import Signal, SignalAction
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI."""

    def __init__(self):
        super().__init__("mean_reversion")

    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Analyze for mean reversion signals."""
        if df is None or len(df) < 30:
            return None

        latest = df.iloc[-1]

        required = ["rsi_14", "bb_upper", "bb_lower", "bb_middle", "zscore", "atr"]
        if not all(col in df.columns and pd.notna(latest.get(col)) for col in required):
            return None

        price = latest["close"]
        rsi = latest["rsi_14"]
        bb_upper = latest["bb_upper"]
        bb_lower = latest["bb_lower"]
        bb_middle = latest["bb_middle"]
        zscore = latest["zscore"]
        atr = latest["atr"]

        rsi_oversold = self.params.get("rsi_oversold", 30)
        rsi_overbought = self.params.get("rsi_overbought", 70)
        zscore_entry = self.params.get("zscore_entry", 2.0)
        min_confidence = self.params.get("min_confidence", 0.5)

        # ── BUY signal (oversold, price at/below lower BB) ────
        at_lower_bb = price <= bb_lower
        rsi_low = rsi < rsi_oversold
        zscore_low = zscore < -zscore_entry

        buy_conditions = {
            "at_lower_bb": at_lower_bb,
            "rsi_oversold": rsi_low,
            "zscore_oversold": zscore_low,
        }

        buy_score = sum(buy_conditions.values()) / len(buy_conditions)

        if buy_score >= 0.66:  # At least 2 of 3
            confidence = min(1.0, buy_score * 0.7 + 0.2)
            if confidence >= min_confidence:
                stop_loss = price - (2.0 * atr)
                take_profit = bb_middle  # Target the mean
                return Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    strategy=self.name,
                    confidence=confidence,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning={
                        "rsi_14": round(rsi, 2),
                        "price": round(price, 2),
                        "bb_lower": round(bb_lower, 2),
                        "bb_middle": round(bb_middle, 2),
                        "zscore": round(zscore, 2),
                        "atr": round(atr, 4),
                        "entry_reason": "oversold_at_lower_bb",
                        "conditions": buy_conditions,
                        "take_profit": round(take_profit, 2),
                    },
                )

        # ── SELL signal (overbought, price at/above upper BB) ─
        at_upper_bb = price >= bb_upper
        rsi_high = rsi > rsi_overbought
        zscore_high = zscore > zscore_entry

        sell_conditions = {
            "at_upper_bb": at_upper_bb,
            "rsi_overbought": rsi_high,
            "zscore_overbought": zscore_high,
        }

        sell_score = sum(sell_conditions.values()) / len(sell_conditions)

        if sell_score >= 0.66:
            confidence = min(1.0, sell_score * 0.7 + 0.2)
            if confidence >= min_confidence * 0.8:
                return Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    strategy=self.name,
                    confidence=confidence,
                    entry_price=price,
                    reasoning={
                        "rsi_14": round(rsi, 2),
                        "price": round(price, 2),
                        "bb_upper": round(bb_upper, 2),
                        "exit_reason": "overbought_at_upper_bb",
                    },
                )

        return None
