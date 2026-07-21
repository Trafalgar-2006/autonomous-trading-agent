"""
Order Manager — orchestrates the full lifecycle of trade execution.

Connects signals → risk checks → broker execution → trade logging.
"""

from __future__ import annotations

import logging
from datetime import datetime

from ..core.config import Config
from ..core.event_bus import EventBus
from ..core.models import (
    Event, EventType, Signal, SignalAction, OrderStatus,
    Trade, Side, Position, TradeOutcome, DecisionStatus,
)
from ..data.store import DataStore
from ..decision.engine import DecisionEngine
from ..risk.manager import RiskManager
from ..risk.stops import compute_stop_level
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
        # Builds decision memos and shares the same RiskManager state.
        self.decision_engine = DecisionEngine(risk_manager=self.risk_manager)

        # Track active trades (symbol -> Trade)
        self._active_trades: dict[str, Trade] = {}

        # Restore open trades persisted from a previous session so stop-loss /
        # take-profit management survives restarts.
        self._load_open_trades()

        logger.info("OrderManager initialized")

    def _load_open_trades(self):
        """Rehydrate still-open trades from the database into memory."""
        try:
            open_trades = self.store.get_open_trades()
        except Exception as e:
            logger.error(f"Failed to load open trades from store: {e}")
            return
        for t in open_trades:
            # If multiple rows share a symbol, keep the latest (list is ordered
            # by entry_time ascending, so the last write wins).
            self._active_trades[t.symbol] = t
        if self._active_trades:
            logger.info(
                f"Restored {len(self._active_trades)} open trade(s) from database: "
                f"{list(self._active_trades.keys())}"
            )

    def reconcile_open_trades(self):
        """
        Reconcile in-memory trades with what the broker actually holds.

        - Trades whose position was closed while the agent was offline are
          dropped from tracking.
        - Broker positions with no tracked trade are reconstructed so they still
          get stop-loss protection.

        Call once at startup (after the broker is reachable).
        """
        positions = self.broker.get_positions()
        held = {p.symbol for p in positions}

        for symbol in list(self._active_trades.keys()):
            if symbol not in held:
                logger.warning(
                    f"Tracked trade {symbol} is no longer held at the broker "
                    f"(closed while offline) — dropping from active tracking"
                )
                del self._active_trades[symbol]

        for pos in positions:
            if pos.symbol not in self._active_trades:
                self._active_trades[pos.symbol] = self._reconstruct_trade(pos)

    async def process_signals(self, signals: list[Signal], market_ok: bool = True) -> list:
        """
        Process a batch of signals through the decision funnel:
        build a memo (APPROVED/WATCHLIST/REJECTED), persist it, and — only in
        'auto' execution mode — execute the APPROVED ones. In 'propose' mode
        nothing is traded; the memos are left for human review.

        `market_ok` is the SPY>SMA regime flag for the market filter.

        Returns the list of DecisionMemo objects for monitoring/alerts.
        """
        memos: list = []
        if not signals:
            return memos

        # Get current state
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        equity = account.get("equity", self.config.initial_capital)
        cash = account.get("cash", self.config.initial_capital)

        auto = self.config.execution_mode == "auto"

        for signal in signals:
            try:
                memo = self.decision_engine.build(signal, positions, equity, cash,
                                                  market_ok=market_ok)
                self.store.save_decision(memo)
                memos.append(memo)

                if memo.status != DecisionStatus.APPROVED:
                    continue

                if auto:
                    await self._process_single_signal(signal, positions, equity, cash)
                else:
                    logger.info(f"[PROPOSE] APPROVED {signal.action.value.upper()} "
                                f"{signal.symbol} — not executed (awaiting human review)")
            except Exception as e:
                logger.error(f"Error processing signal {signal.id} for {signal.symbol}: {e}")

        return memos

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
            # Persist stop/target context explicitly so trailing-stop and
            # take-profit management works even for strategies that don't put
            # them in `reasoning` (e.g. breakout) and survives a restart.
            entry_reasoning = dict(signal.reasoning)
            if signal.stop_loss:
                entry_reasoning.setdefault("stop_loss", round(signal.stop_loss, 4))
            if signal.take_profit:
                entry_reasoning.setdefault("take_profit", round(signal.take_profit, 4))

            trade = Trade(
                symbol=signal.symbol,
                side=Side.BUY,
                strategy=signal.strategy,
                entry_time=datetime.utcnow(),
                entry_price=signal.entry_price or 0,
                quantity=order.quantity,
                signal_id=signal.id,
                entry_reasoning=entry_reasoning,
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
            # Note: the SELL order submitted above already liquidates the full
            # position quantity, so we do NOT also call broker.close_position()
            # here — doing so caused a redundant second sell order.

    @staticmethod
    def _reconstruct_trade(pos: Position) -> Trade:
        """
        Build a Trade record for a broker position we aren't tracking in memory.

        Used so stop-loss / trailing-stop protection still applies to positions
        opened in a previous session (or held before the agent started). Without
        strategy context we fall back to a percentage-based stop (see
        check_stop_losses, which defaults ATR to 2% of entry when absent).
        """
        logger.info(
            f"Reconstructing untracked position for stop management: "
            f"{pos.symbol} qty={pos.quantity:.4f} @ ${pos.entry_price:.2f}"
        )
        return Trade(
            symbol=pos.symbol,
            side=pos.side,
            strategy="reconstructed",
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            quantity=pos.quantity,
            entry_reasoning={},
        )

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
                # Position exists at the broker but isn't tracked in memory
                # (e.g. opened in a previous run, or held before this agent started).
                # Reconstruct it so trailing stops / stop-losses still protect it.
                trade = self._reconstruct_trade(pos)
                self._active_trades[pos.symbol] = trade

            price = pos.current_price
            if price <= 0:
                continue

            entry = trade.entry_price
            if entry <= 0:
                continue

            # Get ATR from entry reasoning (set by strategy)
            atr = trade.entry_reasoning.get("atr", entry * 0.02)  # fallback 2%
            max_loss_pct = self.config.risk.get("stop_loss", {}).get("max_loss_pct", 0.05)

            # Track high watermark for trailing
            high_key = f"_high_{pos.symbol}"
            high_watermark = getattr(self, high_key, entry)
            if price > high_watermark:
                high_watermark = price
                setattr(self, high_key, high_watermark)

            # Dynamic stop level (pure function — see risk/stops.py)
            current_stop, profit_in_atr = compute_stop_level(
                entry=entry,
                price=price,
                atr=atr,
                high_watermark=high_watermark,
                max_loss_pct=max_loss_pct,
            )

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
