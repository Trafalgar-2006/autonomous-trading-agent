"""
Stop-loss / trailing-stop math — pure functions, no side effects.

Kept separate from OrderManager so the money-critical exit logic can be unit
tested in isolation without a broker, database, or event loop.
"""

from __future__ import annotations


def compute_stop_level(
    entry: float,
    price: float,
    atr: float,
    high_watermark: float,
    max_loss_pct: float = 0.05,
) -> tuple[float, float]:
    """
    Compute the current stop level for a LONG position.

    Rules (each phase can only tighten the stop, never loosen it):
      * Base:   max(hard cap at ``max_loss_pct`` below entry, 2x ATR below entry)
      * >= 1x ATR profit: raise stop to just above breakeven (entry + 0.1x ATR)
      * >= 2x ATR profit: trail at 1.5x ATR below the high watermark

    Args:
        entry:          average entry price.
        price:          current price.
        atr:            average true range at entry (falls back to 2% of entry
                        if non-positive).
        high_watermark: highest price seen since entry (>= entry).
        max_loss_pct:   hard cap on loss as a fraction of entry.

    Returns:
        (stop_level, profit_in_atr)
    """
    if atr <= 0:
        atr = entry * 0.02

    hard_stop = entry * (1 - max_loss_pct)
    initial_stop = entry - 2.0 * atr
    stop = max(hard_stop, initial_stop)

    profit_in_atr = (price - entry) / atr if atr > 0 else 0.0

    # Phase 1: lock in breakeven once the trade is 1x ATR in profit.
    if profit_in_atr >= 1.0:
        stop = max(stop, entry + 0.1 * atr)

    # Phase 2: trail below the high watermark once 2x ATR in profit.
    if profit_in_atr >= 2.0:
        stop = max(stop, high_watermark - 1.5 * atr)

    return stop, profit_in_atr
