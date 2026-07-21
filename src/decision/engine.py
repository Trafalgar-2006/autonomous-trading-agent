"""
Decision Engine — turns a raw strategy signal into a structured trade decision.

Inspired by the seb.ai "Scan -> Signals -> Trade Plan -> Risk Check -> Final
Decision" workflow, adapted to our quantitative pipeline. Every actionable
signal produces a DecisionMemo classified as:

    APPROVED  — passes every gate; eligible to trade
    WATCHLIST — interesting, but a soft gate failed (R:R / confidence / trend)
    REJECTED  — blocked by a hard risk rule

The heavy lifting of sizing and hard limits stays in RiskManager; this layer
adds the human-legible verdict, the trade plan (with invalidation + R:R), and
the reasoning trail.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..core.config import Config
from ..core.models import (
    Signal, SignalAction, Position, DecisionMemo, DecisionStatus, MarketRegime,
)
from ..risk.manager import RiskManager

logger = logging.getLogger(__name__)


def compute_risk_reward(entry: Optional[float], stop: Optional[float],
                        target: Optional[float]) -> Optional[float]:
    """Reward:risk for a long setup. None if it can't be computed."""
    if not entry or not stop or not target:
        return None
    risk = entry - stop
    reward = target - entry
    if risk <= 0:
        return None
    return reward / risk


def _risk_level(signal: Signal, regime: Optional[MarketRegime]) -> str:
    """Coarse risk label from volatility context."""
    if regime == MarketRegime.HIGH_VOLATILITY:
        return "high"
    atr = signal.reasoning.get("atr")
    entry = signal.entry_price
    if atr and entry and entry > 0:
        atr_pct = atr / entry
        if atr_pct > 0.03:
            return "high"
        if atr_pct > 0.015:
            return "medium"
        return "low"
    return "medium"


def _invalidation(signal: Signal) -> Optional[float]:
    """
    Level that proves the setup wrong (distinct from the risk stop).

    For breakouts, dropping back to the lower channel invalidates the thesis;
    otherwise fall back to the stop level.
    """
    r = signal.reasoning
    if "donchian_low" in r:
        return r["donchian_low"]
    return signal.stop_loss


class DecisionEngine:
    """Builds DecisionMemos from signals using risk checks + soft gates."""

    def __init__(self, risk_manager: Optional[RiskManager] = None):
        self.config = Config()
        self.risk_manager = risk_manager or RiskManager()

    def build(
        self,
        signal: Signal,
        positions: list[Position],
        equity: float,
        cash: float,
        market_ok: bool = True,
    ) -> DecisionMemo:
        """Evaluate a signal and produce a classified decision memo.

        `market_ok` is the SPY>SMA regime flag; when the market filter is on and
        this is False, new longs are demoted to WATCHLIST.
        """
        entry = signal.entry_price
        stop = signal.stop_loss
        target = signal.take_profit
        rr = compute_risk_reward(entry, stop, target)
        regime = signal.regime

        memo = DecisionMemo(
            symbol=signal.symbol,
            action=signal.action,
            status=DecisionStatus.REJECTED,  # default; refined below
            strategy=signal.strategy,
            signal_strength=signal.confidence,
            risk_level=_risk_level(signal, regime),
            entry=entry,
            target=target,
            stop=stop,
            invalidation=_invalidation(signal),
            timeframe=self.config.timeframe,
            risk_reward=rr,
            regime=regime,
            rationale=signal.reasoning.get("entry_reason")
                      or signal.reasoning.get("exit_reason")
                      or f"{signal.strategy} setup",
            signal_id=signal.id,
        )

        # Ask the risk manager for a sized order (enforces hard limits).
        order = self.risk_manager.evaluate(signal, positions, equity, cash)

        # ── SELL / exit signals: a valid order means we can close ───────────
        if signal.action == SignalAction.SELL:
            if order is not None:
                memo.status = DecisionStatus.APPROVED
                memo.quantity = order.quantity
                memo.reasons = ["exit signal for an open position"]
            else:
                memo.status = DecisionStatus.REJECTED
                memo.reasons = ["no open position to exit"]
            return memo

        # ── BUY signals ────────────────────────────────────────────────────
        if order is None:
            memo.status = DecisionStatus.REJECTED
            memo.reasons = [self._rejection_reason(signal, positions, equity)]
            return memo

        memo.quantity = order.quantity
        if entry and stop:
            memo.dollar_risk = abs(entry - stop) * order.quantity

        # Soft gates -> WATCHLIST rather than outright rejection.
        # Use a small tolerance so a setup sitting exactly at the threshold
        # (e.g. momentum's natural 3xATR/2xATR = 1.5) isn't split by float noise.
        soft_reasons: list[str] = []
        if self.config.market_filter and not market_ok:
            soft_reasons.append(f"market filter: SPY below {self.config.market_filter_sma}-day SMA")
        if rr is not None and rr < self.config.min_risk_reward - 1e-6:
            soft_reasons.append(f"R:R {rr:.2f} < min {self.config.min_risk_reward}")
        if rr is None:
            soft_reasons.append("no target set (R:R unknown)")
        if signal.confidence < self.config.strong_confidence:
            soft_reasons.append(
                f"confidence {signal.confidence:.0%} < strong "
                f"{self.config.strong_confidence:.0%}")
        if signal.reasoning.get("weekly_trend") == "bearish":
            soft_reasons.append("weekly trend is bearish")

        if soft_reasons:
            memo.status = DecisionStatus.WATCHLIST
            memo.reasons = soft_reasons
        else:
            memo.status = DecisionStatus.APPROVED
            memo.reasons = ["passes risk, R:R and confidence gates"]

        return memo

    def _rejection_reason(self, signal: Signal, positions: list[Position],
                          equity: float) -> str:
        """Best-effort human reason for why the risk manager blocked a BUY."""
        status = self.risk_manager.status
        if status.get("cooldown_active"):
            return "circuit breaker active"
        if any(p.symbol == signal.symbol for p in positions):
            return f"already holding {signal.symbol}"
        if len(positions) >= self.config.max_open_positions:
            return f"max open positions ({self.config.max_open_positions}) reached"
        exposure = sum(p.market_value for p in positions)
        if exposure >= equity * self.config.max_total_exposure:
            return f"max exposure ({self.config.max_total_exposure:.0%}) reached"
        return "position size below minimum or insufficient capital"
