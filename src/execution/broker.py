"""
Broker Integration — Alpaca API wrapper for order execution.
"""

from __future__ import annotations

import logging
from datetime import datetime

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from ..core.config import Config
from ..core.models import Order, OrderStatus, OrderType, Position, Side
from .base import BaseBroker

logger = logging.getLogger(__name__)


class AlpacaBroker(BaseBroker):
    """
    Alpaca Trading API wrapper.
    
    Handles:
    - Submitting market/limit orders (with fractional share support)
    - Getting account info (cash, equity, positions)
    - Cancelling orders
    - Getting open positions
    """

    def __init__(self):
        self.config = Config()
        self.client = None

        api_key = self.config.alpaca_api_key
        secret_key = self.config.alpaca_secret_key

        if not api_key or api_key == "your_api_key_here":
            logger.warning(
                "No Alpaca API keys configured for trading. "
                "Get keys at https://app.alpaca.markets/signup"
            )
        else:
            self.client = TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=self.config.is_paper,
            )
            mode = "PAPER" if self.config.is_paper else "LIVE"
            logger.info(f"AlpacaBroker initialized in {mode} mode")

    def get_account(self) -> dict:
        """Get account information."""
        if self.client is None:
            return {
                "cash": self.config.initial_capital,
                "equity": self.config.initial_capital,
                "buying_power": self.config.initial_capital,
                "portfolio_value": self.config.initial_capital,
                "day_trade_count": 0,
                "pattern_day_trader": False,
                "trading_blocked": False,
                "account_blocked": False,
                "status": "NO_API_KEYS",
            }
        try:
            account = self.client.get_account()
            return {
                "cash": float(account.cash),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value),
                "day_trade_count": account.daytrade_count,
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "account_blocked": account.account_blocked,
                "status": account.status,
            }
        except APIError as e:
            logger.error(f"Error getting account: {e}")
            return {}

    def get_positions(self) -> list[Position]:
        """Get all open positions as Position objects."""
        if self.client is None:
            return []
        try:
            positions = self.client.get_all_positions()
            result = []
            for pos in positions:
                result.append(Position(
                    symbol=pos.symbol,
                    side=Side.BUY if pos.side == "long" else Side.SELL,
                    quantity=float(pos.qty),
                    entry_price=float(pos.avg_entry_price),
                    entry_time=datetime.utcnow(),  # Alpaca doesn't provide entry time directly
                    current_price=float(pos.current_price),
                    unrealized_pnl=float(pos.unrealized_pl),
                    unrealized_pnl_pct=float(pos.unrealized_plpc),
                ))
            return result
        except APIError as e:
            logger.error(f"Error getting positions: {e}")
            return []

    def submit_order(self, order: Order) -> Order:
        """
        Submit an order to Alpaca.
        
        Supports market and limit orders with fractional shares.
        Returns the order with updated status and broker_order_id.
        """
        if self.client is None:
            order.status = OrderStatus.REJECTED
            logger.error("Cannot submit order: No API keys configured")
            return order

        # Refuse to open new risk when the account is blocked or PDT-limited.
        if order.side == Side.BUY and self._is_pdt_blocked():
            order.status = OrderStatus.REJECTED
            return order

        try:
            alpaca_side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL

            if order.order_type == OrderType.LIMIT and order.limit_price:
                request = LimitOrderRequest(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=alpaca_side,
                    time_in_force=TimeInForce.GTC,
                    limit_price=order.limit_price,
                )
            else:
                # Market order — use notional for very small amounts
                if order.quantity * (order.limit_price or 100) < 1.0:
                    # Below $1, use notional
                    request = MarketOrderRequest(
                        symbol=order.symbol,
                        notional=round(order.quantity * (order.limit_price or 100), 2),
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                    )
                else:
                    request = MarketOrderRequest(
                        symbol=order.symbol,
                        qty=round(order.quantity, 6),
                        side=alpaca_side,
                        time_in_force=TimeInForce.DAY,
                    )

            result = self._submit_with_backoff(request)

            order.broker_order_id = str(result.id)
            order.status = OrderStatus.SUBMITTED

            logger.info(
                f"Order submitted: {order.side.value.upper()} {order.quantity:.4f} "
                f"{order.symbol} (broker_id={order.broker_order_id})"
            )
            return order

        except APIError as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected by Alpaca: {e}")
            return order
        except Exception as e:
            order.status = OrderStatus.REJECTED
            logger.error(f"Order error: {e}")
            return order

    def _submit_with_backoff(self, request, attempts: int = 3, base_delay: float = 1.0):
        """
        Submit with exponential backoff on rate limits / transient 5xx.

        Only retries errors that are actually transient — a rejected order
        (insufficient funds, bad symbol) is re-raised immediately rather than
        hammered, because retrying it will never succeed.
        """
        import time

        last_error = None
        for attempt in range(attempts):
            try:
                return self.client.submit_order(request)
            except APIError as e:
                message = str(e).lower()
                transient = ("rate limit" in message or "429" in message
                             or "too many" in message or "503" in message
                             or "internal" in message or "timeout" in message)
                last_error = e
                if not transient or attempt == attempts - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Transient broker error ({e}) — retrying in {delay:.0f}s")
                time.sleep(delay)
        # Unreachable in practice (the loop returns or raises), but keeps the
        # function total rather than falling through with None.
        raise last_error or RuntimeError("order submission failed")

    def _is_pdt_blocked(self) -> bool:
        """
        Pattern Day Trader guard.

        Under-$25k margin accounts are limited to 3 day-trades in 5 business
        days; a 4th can freeze the account for 90 days. We stop before that
        rather than letting the broker reject (or worse, accept) the trade.
        """
        try:
            account = self.get_account()
            if account.get("trading_blocked") or account.get("account_blocked"):
                logger.error("Broker reports the account is blocked — refusing to trade")
                return True
            equity = float(account.get("equity", 0) or 0)
            day_trades = int(account.get("day_trade_count", 0) or 0)
            if equity < 25_000 and day_trades >= 3:
                logger.error(f"PDT guard: {day_trades} day-trades on a "
                             f"${equity:,.0f} account — refusing to trade")
                return True
        except Exception as e:
            logger.debug(f"PDT check skipped: {e}")
        return False

    def get_order(self, broker_order_id: str) -> dict:
        """
        Fetch an order's current state from the broker.

        Returns {status, filled_qty, filled_avg_price} — used to record the
        real fill price and measure slippage against what we expected.
        """
        if self.client is None or not broker_order_id:
            return {}
        try:
            o = self.client.get_order_by_id(broker_order_id)
            return {
                "status": str(getattr(o, "status", "")),
                "filled_qty": float(getattr(o, "filled_qty", 0) or 0),
                "filled_avg_price": float(getattr(o, "filled_avg_price", 0) or 0),
            }
        except Exception as e:
            logger.debug(f"Could not fetch order {broker_order_id}: {e}")
            return {}

    def cancel_order(self, broker_order_id: str) -> bool:
        """Cancel an open order."""
        if self.client is None:
            return False
        try:
            self.client.cancel_order_by_id(broker_order_id)
            logger.info(f"Order cancelled: {broker_order_id}")
            return True
        except APIError as e:
            logger.error(f"Error cancelling order {broker_order_id}: {e}")
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders (emergency)."""
        if self.client is None:
            return False
        try:
            self.client.cancel_orders()
            logger.warning("ALL orders cancelled")
            return True
        except APIError as e:
            logger.error(f"Error cancelling all orders: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close an entire position for a symbol."""
        if self.client is None:
            return False
        try:
            self.client.close_position(symbol)
            logger.info(f"Position closed: {symbol}")
            return True
        except APIError as e:
            logger.error(f"Error closing position {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions (emergency)."""
        if self.client is None:
            return False
        try:
            self.client.close_all_positions(cancel_orders=True)
            logger.warning("ALL positions closed")
            return True
        except APIError as e:
            logger.error(f"Error closing all positions: {e}")
            return False

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        if self.client is None:
            return False
        try:
            clock = self.client.get_clock()
            return clock.is_open
        except APIError:
            return False

    def get_next_market_open(self) -> str | None:
        """Get the next market open time."""
        if self.client is None:
            return None
        try:
            clock = self.client.get_clock()
            return str(clock.next_open)
        except APIError:
            return None
