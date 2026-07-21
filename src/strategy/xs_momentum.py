"""
Cross-Sectional Momentum strategy (live).

Ranks the whole universe by trailing return over a lookback (default 12 months)
and holds the top-N names, rotating monthly-ish as ranks change. This is the
classic momentum factor (Jegadeesh & Titman) and — unlike the per-symbol TA
strategies — it was the first configuration to beat SPY buy-and-hold on a
walk-forward, out-of-sample test over a broad universe.

Unlike the ensemble, this does NOT use the SPY market filter (walk-forward
evidence showed the filter hurts relative-momentum returns).
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..core.models import Signal, SignalAction

logger = logging.getLogger(__name__)


class CrossSectionalMomentum:
    """Universe-level momentum: long the top-N trailing-return names."""

    name = "xs_momentum"

    def __init__(self, lookback: int = 252, top_n: int = 8,
                 vol_target: Optional[float] = None):
        self.lookback = lookback
        self.top_n = top_n
        self.vol_target = vol_target

    def rank(self, enriched: dict[str, pd.DataFrame]) -> list[tuple[str, float]]:
        """Return (symbol, trailing_return) sorted best-first."""
        scores = []
        for sym, df in enriched.items():
            if df is None or len(df) <= self.lookback:
                continue
            price = float(df["close"].iloc[-1])
            past = float(df["close"].iloc[-1 - self.lookback])
            if past > 0 and price > 0:
                scores.append((sym, price / past - 1.0))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def build_signals(self, enriched: dict[str, pd.DataFrame],
                      held: Optional[set] = None) -> list[Signal]:
        """
        Emit BUY for top-N names not held and SELL for held names that have
        dropped out of the top-N (rotation). `held` is the set of currently
        held symbols; when None, only BUY signals for the top-N are produced.
        """
        held = held or set()
        ranked = self.rank(enriched)
        if not ranked:
            return []
        top = [s for s, _ in ranked[: self.top_n]]
        score_map = dict(ranked)

        signals: list[Signal] = []

        # BUY the top names we don't already hold.
        for sym in top:
            if sym in held:
                continue
            df = enriched[sym]
            row = df.iloc[-1]
            price = float(row["close"])
            atr = float(row["atr"]) if "atr" in row and pd.notna(row["atr"]) else price * 0.02
            reasoning = {
                "atr": atr,
                "xs_return": round(score_map.get(sym, 0.0), 4),
                "entry_reason": "cross_sectional_momentum",
            }
            if self.vol_target and "volatility_20d" in row and pd.notna(row["volatility_20d"]):
                vol = float(row["volatility_20d"])
                if vol > 0:
                    reasoning["size_mult"] = float(min(2.0, max(0.5, self.vol_target / vol)))
            signals.append(Signal(
                symbol=sym, action=SignalAction.BUY, strategy=self.name,
                confidence=0.8, entry_price=price,
                stop_loss=price - 2.0 * atr, take_profit=price + 3.0 * atr,
                reasoning=reasoning,
            ))

        # SELL held names that fell out of the top (rotation).
        top_set = set(top)
        for sym in held:
            if sym in top_set or sym not in enriched:
                continue
            price = float(enriched[sym]["close"].iloc[-1])
            signals.append(Signal(
                symbol=sym, action=SignalAction.SELL, strategy=self.name,
                confidence=1.0, entry_price=price,
                reasoning={"exit_reason": "fell_out_of_top"},
            ))

        return signals
