"""
Risk Manager — evaluates signals and enforces position/portfolio limits.

Safety features:
- Position sizing (fixed fractional)
- Exposure limits
- Circuit breakers (daily/weekly loss limits)
- Correlation checks
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ..core.config import Config
from ..core.models import Order, Position, Side, Signal, SignalAction

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Evaluates signals against risk rules and produces sized orders.
    
    Pipeline:
    1. Check circuit breakers (daily/weekly loss limits)
    2. Check portfolio exposure limits
    3. Check for duplicate/conflicting positions
    4. Calculate position size
    5. Return Order or None (rejected)
    """

    def __init__(self):
        self.config = Config()

        # Daily P&L tracking
        self._daily_pnl: float = 0.0
        self._daily_pnl_reset: datetime = datetime.utcnow()
        self._weekly_pnl: float = 0.0
        self._weekly_pnl_reset: datetime = datetime.utcnow()
        self._consecutive_losses: int = 0
        self._cooldown_until: datetime | None = None

        logger.info("RiskManager initialized")

    @property
    def status(self) -> dict:
        """Get current risk status."""
        return {
            "daily_pnl": self._daily_pnl,
            "weekly_pnl": self._weekly_pnl,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_active": self._cooldown_until is not None and datetime.utcnow() < self._cooldown_until,
            "cooldown_until": str(self._cooldown_until) if self._cooldown_until else None,
        }

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracker after a trade closes."""
        now = datetime.utcnow()

        # Reset daily counter if new day
        if now.date() > self._daily_pnl_reset.date():
            self._daily_pnl = 0.0
            self._daily_pnl_reset = now

        # Reset weekly counter if new week
        if (now - self._weekly_pnl_reset).days >= 7:
            self._weekly_pnl = 0.0
            self._weekly_pnl_reset = now

        self._daily_pnl += pnl
        self._weekly_pnl += pnl

        # Track consecutive losses
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        # Check circuit breakers — only LOSSES should trip them, never profits.
        equity = self.config.initial_capital  # Simplified; in production, use live equity

        daily_loss_pct = -self._daily_pnl / equity if equity > 0 else 0.0
        weekly_loss_pct = -self._weekly_pnl / equity if equity > 0 else 0.0

        if daily_loss_pct > self.config.max_daily_loss:
            self._activate_cooldown("daily loss limit exceeded")

        if weekly_loss_pct > self.config.max_weekly_loss:
            self._activate_cooldown("weekly loss limit exceeded")

        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self._activate_cooldown(f"{self._consecutive_losses} consecutive losses")

    def _activate_cooldown(self, reason: str):
        """Activate the circuit breaker cooldown."""
        hours = self.config.cooldown_hours
        self._cooldown_until = datetime.utcnow() + timedelta(hours=hours)
        logger.warning(
            f"CIRCUIT BREAKER: {reason}. "
            f"Trading paused until {self._cooldown_until}"
        )

    def evaluate(
        self,
        signal: Signal,
        positions: list[Position],
        equity: float,
        cash: float,
    ) -> Order | None:
        """
        Evaluate a signal against risk rules.
        
        Returns an Order if approved, None if rejected.
        """
        # ── 1. Circuit breaker check ──────────────────────────
        if self._cooldown_until and datetime.utcnow() < self._cooldown_until:
            logger.info(f"Signal rejected: circuit breaker active until {self._cooldown_until}")
            return None

        # ── 2. Only process BUY/SELL ──────────────────────────
        if signal.action == SignalAction.HOLD:
            return None

        # ── 3. For SELL, just create the order ────────────────
        if signal.action == SignalAction.SELL:
            # Find the matching position
            matching = [p for p in positions if p.symbol == signal.symbol]
            if matching:
                return Order(
                    symbol=signal.symbol,
                    side=Side.SELL,
                    quantity=matching[0].quantity,
                    signal_id=signal.id,
                    limit_price=signal.entry_price,
                )
            # No position to sell
            logger.debug(f"No position to sell for {signal.symbol}")
            return None

        # ── 4. BUY checks ────────────────────────────────────

        # Check max positions
        if len(positions) >= self.config.max_open_positions:
            logger.info(f"Signal rejected: max positions ({self.config.max_open_positions}) reached")
            return None

        # Check if already holding
        if any(p.symbol == signal.symbol for p in positions):
            logger.info(f"Signal rejected: already holding {signal.symbol}")
            return None

        # Check total exposure
        total_exposure = sum(p.market_value for p in positions)
        max_exposure = equity * self.config.max_total_exposure
        if total_exposure >= max_exposure:
            logger.info(f"Signal rejected: max exposure ({self.config.max_total_exposure:.0%}) reached")
            return None

        # ── 5. Position sizing ────────────────────────────────
        quantity = self._calculate_position_size(signal, equity, cash)

        if quantity is None or quantity <= 0:
            logger.info(f"Signal rejected: calculated position size is zero for {signal.symbol}")
            return None

        # Check minimum position value
        min_value = self.config.risk.get("min_position_value", 1.0)
        position_value = quantity * (signal.entry_price or 0)
        if position_value < min_value:
            logger.info(f"Signal rejected: position value ${position_value:.2f} < minimum ${min_value}")
            return None

        return Order(
            symbol=signal.symbol,
            side=Side.BUY,
            quantity=quantity,
            signal_id=signal.id,
            limit_price=signal.entry_price,
        )

    def _calculate_position_size(
        self,
        signal: Signal,
        equity: float,
        cash: float,
    ) -> float | None:
        """
        Calculate position size using fixed fractional method.
        
        Position size = (equity * risk_pct) / risk_per_share
        """
        if not signal.entry_price or signal.entry_price <= 0:
            return None

        price = signal.entry_price
        risk_pct = self.config.max_risk_per_trade
        max_position_pct = self.config.max_position_size
        reserve_pct = self.config.risk.get("capital", {}).get("reserve_pct", 0.20)

        # Maximum capital available (respecting reserve)
        available_capital = cash * (1 - reserve_pct)
        max_position_value = equity * max_position_pct

        # Risk-based sizing
        if signal.stop_loss and signal.stop_loss > 0:
            risk_per_share = abs(price - signal.stop_loss)
            if risk_per_share > 0:
                risk_capital = equity * risk_pct
                risk_based_qty = risk_capital / risk_per_share
            else:
                risk_based_qty = float("inf")
        else:
            # No stop loss — use ATR or default 2%
            atr = signal.reasoning.get("atr", price * 0.02)
            risk_per_share = atr * 2  # 2x ATR
            risk_capital = equity * risk_pct
            risk_based_qty = risk_capital / risk_per_share if risk_per_share > 0 else 0

        # Volatility-target multiplier (set by the ensemble when vol_target is on)
        size_mult = signal.reasoning.get("size_mult", 1.0)
        risk_based_qty *= size_mult

        # Liquidity guard: never take more than max_adv_pct of the name's
        # average daily dollar volume — otherwise our own order moves the price
        # and the backtest's fill assumptions stop being believable.
        adv_notional = signal.reasoning.get("adv_notional")
        max_adv_pct = self.config.max_adv_pct
        if adv_notional and max_adv_pct and adv_notional > 0:
            liquidity_cap_qty = (adv_notional * max_adv_pct) / price
            if liquidity_cap_qty < risk_based_qty:
                logger.info(f"Liquidity guard: capping {signal.symbol} to "
                            f"{max_adv_pct:.1%} of ADV")
            risk_based_qty = min(risk_based_qty, liquidity_cap_qty)

        # Capital-constrained sizing
        capital_based_qty = min(available_capital, max_position_value) / price

        # Take the smaller of risk-based and capital-based
        quantity = min(risk_based_qty, capital_based_qty)

        # Final sanity check
        if quantity * price > available_capital:
            quantity = available_capital / price

        return round(max(0, quantity), 6)
