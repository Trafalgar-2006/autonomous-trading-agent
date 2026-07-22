"""
Broker interface.

Every broker (Alpaca live/paper, the local PaperBroker, and any future IBKR /
Binance adapter) implements this surface, so switching brokers is a config
change rather than a rewrite. The OrderManager only ever talks to this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.models import Order, Position


class BaseBroker(ABC):
    """Abstract broker every adapter must implement."""

    @abstractmethod
    def get_account(self) -> dict:
        """{cash, equity, buying_power, portfolio_value, status, ...}."""

    @abstractmethod
    def get_positions(self) -> list[Position]:
        ...

    @abstractmethod
    def submit_order(self, order: Order) -> Order:
        """Submit an order; return it with status + broker_order_id set."""

    @abstractmethod
    def get_order(self, broker_order_id: str) -> dict:
        """{status, filled_qty, filled_avg_price}."""

    @abstractmethod
    def close_position(self, symbol: str) -> bool:
        ...

    # ── Optional operations with safe defaults ───────────────────
    def cancel_order(self, broker_order_id: str) -> bool:
        return False

    def cancel_all_orders(self) -> bool:
        return False

    def close_all_positions(self) -> bool:
        ok = True
        for pos in self.get_positions():
            ok = self.close_position(pos.symbol) and ok
        return ok

    def is_market_open(self) -> bool:
        return True

    def get_next_market_open(self):
        return None
