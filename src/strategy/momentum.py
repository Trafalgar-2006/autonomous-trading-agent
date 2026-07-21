"""
Momentum Strategy — trades in the direction of the prevailing trend.

Signals based on:
- RSI momentum
- MACD crossovers
- EMA alignment
- ADX trend strength
"""

from __future__ import annotations

import logging

import pandas as pd

from ..core.models import Signal, SignalAction
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """Trend-following momentum strategy."""

    def __init__(self):
        super().__init__("momentum")

    def analyze(self, symbol: str, df: pd.DataFrame) -> Signal | None:
        """Analyze for momentum signals."""
        if df is None or len(df) < 50:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Required features
        required = ["rsi_14", "macd", "macd_signal", "ema_9", "ema_21", "sma_50", "adx", "atr"]
        if not all(col in df.columns and pd.notna(latest.get(col)) for col in required):
            return None

        rsi = latest["rsi_14"]
        macd = latest["macd"]
        macd_sig = latest["macd_signal"]
        prev_macd = prev["macd"]
        prev_macd_sig = prev["macd_signal"]
        ema_9 = latest["ema_9"]
        ema_21 = latest["ema_21"]
        sma_50 = latest["sma_50"]
        adx = latest["adx"]
        atr = latest["atr"]
        price = latest["close"]

        adx_threshold = self.params.get("adx_threshold", 25)
        min_confidence = self.params.get("min_confidence", 0.5)

        # ── BUY conditions ────────────────────────────────────
        macd_cross_up = (prev_macd <= prev_macd_sig) and (macd > macd_sig)
        ema_aligned = ema_9 > ema_21
        above_sma50 = price > sma_50
        rsi_bullish = 40 < rsi < 70
        trending = adx > adx_threshold

        buy_conditions = {
            "macd_cross_up": macd_cross_up,
            "ema_aligned": ema_aligned,
            "above_sma50": above_sma50,
            "rsi_bullish": rsi_bullish,
            "trending": trending,
        }

        buy_score = sum(buy_conditions.values()) / len(buy_conditions)

        if buy_score >= 0.6:
            confidence = min(1.0, buy_score * 0.8 + (adx / 100) * 0.2)
            if confidence >= min_confidence:
                stop_loss = price - (2.0 * atr)
                take_profit = price + (3.0 * atr)
                return Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    strategy=self.name,
                    confidence=confidence,
                    entry_price=price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning={
                        "price": round(price, 2),
                        "rsi": round(rsi, 2),
                        "macd": round(macd, 4),
                        "adx": round(adx, 2),
                        "atr": round(atr, 4),
                        "conditions": buy_conditions,
                        "stop_loss": round(stop_loss, 2),
                        "take_profit": round(take_profit, 2),
                    },
                )

        # ── SELL conditions ───────────────────────────────────
        macd_cross_down = (prev_macd >= prev_macd_sig) and (macd < macd_sig)
        ema_bearish = ema_9 < ema_21
        below_sma50 = price < sma_50
        rsi_bearish = rsi > 70 or rsi < 30

        sell_conditions = {
            "macd_cross_down": macd_cross_down,
            "ema_bearish": ema_bearish,
            "below_sma50": below_sma50,
            "rsi_bearish": rsi_bearish,
        }

        sell_score = sum(sell_conditions.values()) / len(sell_conditions)

        if sell_score >= 0.5:
            confidence = min(1.0, sell_score * 0.8)
            if confidence >= min_confidence * 0.8:
                return Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    strategy=self.name,
                    confidence=confidence,
                    entry_price=price,
                    reasoning={
                        "price": round(price, 2),
                        "rsi": round(rsi, 2),
                        "macd": round(macd, 4),
                        "adx": round(adx, 2),
                        "exit_reason": "momentum_reversal",
                        "conditions": sell_conditions,
                    },
                )

        return None
