"""
Breakout Strategy — trades price breaking out of ranges.

Signals based on:
- Donchian Channel breakouts
- Volume surges
- ADX trend strength confirmation
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..core.models import Signal, SignalAction
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class BreakoutStrategy(BaseStrategy):
    """Donchian channel breakout strategy."""

    def __init__(self):
        super().__init__("breakout")

    def analyze(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Analyze for breakout signals."""
        if df is None or len(df) < 30:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        required = ["donchian_high", "donchian_low", "donchian_mid",
                    "volume_ratio", "adx", "atr"]
        if not all(col in df.columns and pd.notna(latest.get(col)) for col in required):
            return None

        price = latest["close"]
        prev_close = prev["close"]
        donchian_high = latest["donchian_high"]
        donchian_low = latest["donchian_low"]
        donchian_mid = latest["donchian_mid"]
        vol_ratio = latest["volume_ratio"]
        adx = latest["adx"]
        atr = latest["atr"]

        vol_surge_mult = self.params.get("volume_surge_multiplier", 1.5)
        adx_threshold = self.params.get("adx_threshold", 25)
        min_confidence = self.params.get("min_confidence", 0.5)

        # ── Long Breakout ─────────────────────────────────────
        # Price breaks above Donchian high with volume and trend
        long_breakout = price > donchian_high and prev_close <= donchian_high
        volume_surge = vol_ratio > vol_surge_mult
        adx_trending = adx > adx_threshold

        if long_breakout:
            buy_conditions = {
                "long_breakout": True,
                "volume_surge": volume_surge,
                "adx_trending": adx_trending,
            }

            buy_score = sum(buy_conditions.values()) / len(buy_conditions)

            if buy_score >= 0.66:
                confidence = min(1.0, buy_score * 0.7 + 0.15)

                if confidence >= min_confidence:
                    stop_loss = donchian_mid
                    risk = price - stop_loss
                    take_profit = price + (risk * 2.0) if risk > 0 else price + (3 * atr)
                    risk_reward = (take_profit - price) / risk if risk > 0 else 0
                    breakout_pct = ((price - donchian_high) / donchian_high) * 100

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
                            "donchian_high": round(donchian_high, 2),
                            "donchian_low": round(donchian_low, 2),
                            "donchian_mid": round(donchian_mid, 2),
                            "vol_ratio": round(vol_ratio, 2),
                            "adx": round(adx, 2),
                            "atr": round(atr, 4),
                            "breakout_pct": round(breakout_pct, 2),
                            "conditions": buy_conditions,
                            "risk_reward": round(risk_reward, 2),
                        },
                    )

        # ── Short Breakout (sell signal) ──────────────────────
        short_breakout = price < donchian_low and prev_close >= donchian_low

        if short_breakout:
            sell_conditions = {
                "short_breakout": True,
                "volume_surge": volume_surge,
                "adx_trending": adx_trending,
            }

            sell_score = sum(sell_conditions.values()) / len(sell_conditions)

            if sell_score >= 0.66:
                confidence = min(1.0, sell_score * 0.7 + 0.15)
                if confidence >= min_confidence * 0.8:
                    return Signal(
                        symbol=symbol,
                        action=SignalAction.SELL,
                        strategy=self.name,
                        confidence=confidence,
                        entry_price=price,
                        reasoning={
                            "price": round(price, 2),
                            "donchian_low": round(donchian_low, 2),
                            "vol_ratio": round(vol_ratio, 2),
                            "adx": round(adx, 2),
                            "exit_reason": "downside_breakout",
                            "conditions": sell_conditions,
                        },
                    )

        return None
