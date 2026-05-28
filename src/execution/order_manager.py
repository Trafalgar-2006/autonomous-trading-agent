"""
Order Manager — orchestrates the full lifecycle of trade execution.

Connects signals → risk checks → broker execution → trade logging.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from ..core.config import Config
from ..core.event_bus import EventBus
from ..core.models import (
    Event, EventType, Signal, SignalAction, Order, OrderStatus,
    Trade, Side, Position, TradeOutcome,
)
from ..data.store import DataStore
from ..risk.manager import RiskManager
from .broker import AlpacaBroker

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages the full trade lifecycle:
    
    1. Receives signals from the strategy engine
    2. Passes them through the risk manager
    3. Submits approved orders to the broker
    4. Tracks and logs all trade outcomes
    """

    def __init__(self):
        self.config = Config()
        self.store = DataStore()
        self.bus = EventBus()
        self.risk_manager = RiskManager()
        self.broker = AlpacaBroker()
        
        # Track active trades (signal_id -> Trade)
        self._active_trades: dict[str, Trade] = {}
        
        logger.info("OrderManager initialized")

    async def process_signals(self, signals: list[Signal]):
        """
        Process a batch of signals: risk-check, execute, and log each one.
        """
        if not signals:
            return

        # Get current state
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        equity = account.get("equity", self.config.initial_capital)
        cash = account.get("cash", self.config.initial_capital)

        for signal in signals:
            try:
                await self._process_single_signal(signal, positions, equity, cash)
            except Exception as e:
                logger.error(f"Error processing signal {signal.id} for {signal.symbol}: {e}")

    async def _process_single_signal(
        self,
        signal: Signal,
        positions: list[Position],
        equity: float,
        cash: float,
    ):
        """Process a single signal through the full pipeline."""
        
        # Save the signal regardless of outcome
        self.store.save_signal(signal)

        # ── Risk Check ────────────────────────────────────────
        
        order = self.risk_manager.evaluate(signal, positions, equity, cash)
        
        if not order:
            await self.bus.publish(Event(
                type=EventType.RISK_REJECTED,
                data={"signal_id": signal.id, "symbol": signal.symbol},
                source="order_manager",
            ))
            return

        await self.bus.publish(Event(
            type=EventType.RISK_APPROVED,
            data={"signal_id": signal.id, "symbol": signal.symbol, "quantity": order.quantity},
            source="order_manager",
        ))

        # ── Execute Order ─────────────────────────────────────

        order = self.broker.submit_order(order)

        if order.status == OrderStatus.REJECTED:
            await self.bus.publish(Event(
                type=EventType.ORDER_REJECTED,
                data={"order_id": order.id, "symbol": order.symbol},
                source="order_manager",
            ))
            return

        await self.bus.publish(Event(
            type=EventType.ORDER_SUBMITTED,
            data={
                "order_id": order.id,
                "broker_order_id": order.broker_order_id,
                "symbol": order.symbol,
                "side": order.side.value,
                "quantity": order.quantity,
            },
            source="order_manager",
        ))

        # ── Record Trade ──────────────────────────────────────

        if signal.action == SignalAction.BUY:
            trade = Trade(
                symbol=signal.symbol,
                side=Side.BUY,
                strategy=signal.strategy,
                entry_time=datetime.utcnow(),
                entry_price=signal.entry_price or 0,
                quantity=order.quantity,
                signal_id=signal.id,
                entry_reasoning=signal.reasoning,
            )
            self._active_trades[signal.symbol] = trade
            self.store.save_trade(trade)
            
            logger.info(f"Trade opened: BUY {order.quantity:.4f} {signal.symbol} "
                       f"@ ${signal.entry_price:.2f}")

        elif signal.action == SignalAction.SELL:
            # Close the existing trade
            active_trade = self._active_trades.get(signal.symbol)
            if active_trade:
                active_trade.close(
                    exit_price=signal.entry_price or 0,
                    exit_time=datetime.utcnow(),
                    exit_reason=signal.reasoning.get("exit_reason", "signal"),
                )
                active_trade.exit_reasoning = signal.reasoning
                self.store.save_trade(active_trade)
                
                # Update risk manager
                self.risk_manager.update_daily_pnl(active_trade.pnl)
                
                del self._active_trades[signal.symbol]
                
                emoji = "WIN" if active_trade.outcome == TradeOutcome.WIN else "LOSS"
                logger.info(
                    f"Trade closed ({emoji}): {signal.symbol} "
                    f"PnL=${active_trade.pnl:.4f} ({active_trade.pnl_pct:.2%}) "
                    f"[{active_trade.exit_reason}]"
                )
            
            # Also close via broker
            self.broker.close_position(signal.symbol)

    async def check_stop_losses(self):
        """
        Check all open positions against dynamic trailing stop-loss levels.
        
        Trailing stop logic:
        1. Initial stop: 2x ATR (or 5% hard cap) below entry
        2. After 1x ATR profit: Move stop to breakeven
        3. After 2x ATR profit: Trail stop at 1.5x ATR below highest price
        4. Take-profit: Close at target if set
        """
        positions = self.broker.get_positions()
        if not positions:
            return

        for pos in positions:
            trade = self._active_trades.get(pos.symbol)
            if not trade:
                continue

            price = pos.current_price
            if price <= 0:
                continue

            entry = trade.entry_price
            if entry <= 0:
                continue

            # Get ATR from entry reasoning (set by strategy)
            atr = trade.entry_reasoning.get("atr", entry * 0.02)  # fallback 2%
            max_loss_pct = self.config.risk.get("stop_loss", {}).get("max_loss_pct", 0.05)

            # ── Calculate dynamic stop level ──────────────────
            hard_stop = entry * (1 - max_loss_pct)       # 5% hard cap
            initial_stop = entry - (2.0 * atr)            # ATR-based stop
            current_stop = max(hard_stop, initial_stop)   # Use the tighter one

            profit = price - entry
            profit_in_atr = profit / atr if atr > 0 else 0

            # Track high watermark for trailing
            high_key = f"_high_{pos.symbol}"
            high_watermark = getattr(self, high_key, entry)
            if price > high_watermark:
                high_watermark = price
                setattr(self, high_key, high_watermark)

            # Phase 1: After 1x ATR profit → move to breakeven
            if profit_in_atr >= 1.0:
                breakeven_stop = entry + (0.1 * atr)  # Tiny buffer above entry
                current_stop = max(current_stop, breakeven_stop)

            # Phase 2: After 2x ATR profit → trail at 1.5x ATR below high
            if profit_in_atr >= 2.0:
                trailing_stop = high_watermark - (1.5 * atr)
                current_stop = max(current_stop, trailing_stop)

            # ── Check stop-loss hit ───────────────────────────
            if price <= current_stop:
                stop_reason = "trailing_stop" if profit_in_atr >= 1.0 else "stop_loss_hit"
                logger.warning(
                    f"STOP: {stop_reason.upper()}: {pos.symbol} @ ${price:.2f} "
                    f"(entry=${entry:.2f}, stop=${current_stop:.2f}, "
                    f"P&L={profit_in_atr:.1f}x ATR)"
                )

                stop_signal = Signal(
                    symbol=pos.symbol,
                    action=SignalAction.SELL,
                    strategy=trade.strategy or "risk_manager",
                    confidence=1.0,
                    entry_price=price,
                    reasoning={
                        "exit_reason": stop_reason,
                        "stop_level": round(current_stop, 2),
                        "entry_price": round(entry, 2),
                        "high_watermark": round(high_watermark, 2),
                        "profit_atr_multiple": round(profit_in_atr, 2),
                    },
                )

                await self._process_single_signal(
                    stop_signal, positions,
                    sum(p.market_value for p in positions),
                    0,
                )

                # Clean up watermark tracker
                if hasattr(self, high_key):
                    delattr(self, high_key)
                continue

            # ── Check take-profit ─────────────────────────────
            take_profit = trade.entry_reasoning.get("take_profit")
            if take_profit and price >= take_profit:
                logger.info(
                    f"TAKE PROFIT: {pos.symbol} @ ${price:.2f} "
                    f"(target=${take_profit:.2f})"
                )

                tp_signal = Signal(
                    symbol=pos.symbol,
                    action=SignalAction.SELL,
                    strategy=trade.strategy or "risk_manager",
                    confidence=1.0,
                    entry_price=price,
                    reasoning={
                        "exit_reason": "take_profit",
                        "target_price": round(take_profit, 2),
                        "profit_atr_multiple": round(profit_in_atr, 2),
                    },
                )

                await self._process_single_signal(
                    tp_signal, positions,
                    sum(p.market_value for p in positions),
                    0,
                )

                if hasattr(self, high_key):
                    delattr(self, high_key)

    def emergency_close_all(self):
        """Emergency: close all positions and cancel all orders."""
        logger.critical("EMERGENCY: Closing all positions and cancelling all orders")
        self.broker.cancel_all_orders()
        self.broker.close_all_positions()
        self._active_trades.clear()
