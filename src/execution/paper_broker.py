"""
PaperBroker — a fully local, simulated broker (no account, no API keys).

Runs the entire pipeline against an in-memory (file-persisted) portfolio, so
you can develop, demo, and test the whole system — including `run` — with no
Alpaca account at all. It's also what makes broker-independent integration
tests possible.

Fill model: market orders fill immediately at the order's reference price
(the signal's entry price) plus configurable slippage. It's a simulation, not
a market — realistic enough to exercise the plumbing, not a substitute for the
real fills the live broker records.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ..core.config import Config
from ..core.models import Order, OrderStatus, Position, Side
from .base import BaseBroker

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/paper_state.json")


class PaperBroker(BaseBroker):
    """Local simulated broker with file-persisted state."""

    def __init__(self, slippage_bps: float = 5.0, state_path: Path = STATE_PATH):
        self.config = Config()
        self.slippage_bps = slippage_bps
        self.state_path = Path(state_path)
        self._order_seq = 0
        self._orders: dict[str, dict] = {}
        self._load()
        logger.info(f"PaperBroker initialized (simulated) — cash ${self._cash:,.2f}")

    # ── Persistence ───────────────────────────────────────────────
    def _load(self):
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                self._cash = float(data.get("cash", self.config.initial_capital))
                self._positions = data.get("positions", {})  # symbol -> {qty, avg_entry, last}
                return
            except Exception as e:
                logger.warning(f"PaperBroker state unreadable ({e}) — starting fresh")
        self._cash = float(self.config.initial_capital)
        self._positions = {}

    def _save(self):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(json.dumps(
                {"cash": self._cash, "positions": self._positions}, indent=2))
        except Exception as e:
            logger.error(f"PaperBroker could not save state: {e}")

    # ── Account / positions ───────────────────────────────────────
    def _positions_value(self) -> float:
        return sum(p["qty"] * p.get("last", p["avg_entry"]) for p in self._positions.values())

    def get_account(self) -> dict:
        equity = self._cash + self._positions_value()
        return {
            "cash": self._cash,
            "equity": equity,
            "buying_power": self._cash,
            "portfolio_value": equity,
            "day_trade_count": 0,
            "pattern_day_trader": False,
            "trading_blocked": False,
            "account_blocked": False,
            "status": "PAPER_SIM",
        }

    def get_positions(self) -> list[Position]:
        out = []
        for symbol, p in self._positions.items():
            if p["qty"] <= 0:
                continue
            out.append(Position(
                symbol=symbol, side=Side.BUY, quantity=p["qty"],
                entry_price=p["avg_entry"], entry_time=datetime.utcnow(),
                current_price=p.get("last", p["avg_entry"]),
            ))
        return out

    def mark_prices(self, prices: dict[str, float]):
        """Update last-known prices (call each cycle to mark positions to market)."""
        for symbol, px in prices.items():
            if symbol in self._positions and px and px > 0:
                self._positions[symbol]["last"] = float(px)
        self._save()

    # ── Orders ────────────────────────────────────────────────────
    def _fill_price(self, ref: float, side: Side) -> float:
        adj = 1 + self.slippage_bps / 1e4 if side == Side.BUY else 1 - self.slippage_bps / 1e4
        return ref * adj

    def submit_order(self, order: Order) -> Order:
        ref = order.limit_price or 0.0
        if ref <= 0 and order.symbol in self._positions:
            ref = self._positions[order.symbol].get("last", 0.0)
        if ref <= 0:
            order.status = OrderStatus.REJECTED
            logger.warning(f"PaperBroker: no price for {order.symbol} — rejected")
            return order

        fill = self._fill_price(ref, order.side)
        qty = order.quantity

        if order.side == Side.BUY:
            cost = qty * fill
            if cost > self._cash + 1e-6:
                qty = self._cash / fill  # can't spend more than we have
                cost = qty * fill
            if qty <= 0:
                order.status = OrderStatus.REJECTED
                return order
            self._cash -= cost
            pos = self._positions.setdefault(order.symbol, {"qty": 0.0, "avg_entry": fill, "last": fill})
            new_qty = pos["qty"] + qty
            pos["avg_entry"] = (pos["avg_entry"] * pos["qty"] + fill * qty) / new_qty if new_qty else fill
            pos["qty"] = new_qty
            pos["last"] = fill
        else:  # SELL
            pos = self._positions.get(order.symbol)
            held = pos["qty"] if pos else 0.0
            qty = min(qty, held)
            if qty <= 0:
                order.status = OrderStatus.REJECTED
                return order
            self._cash += qty * fill
            pos["qty"] = held - qty
            pos["last"] = fill
            if pos["qty"] <= 1e-9:
                del self._positions[order.symbol]

        self._order_seq += 1
        order.broker_order_id = f"paper-{self._order_seq}"
        order.status = OrderStatus.FILLED
        self._orders[order.broker_order_id] = {
            "status": "filled", "filled_qty": qty, "filled_avg_price": fill}
        self._save()
        logger.info(f"PaperBroker filled: {order.side.value.upper()} {qty:.4f} "
                    f"{order.symbol} @ ${fill:.4f}")
        return order

    def get_order(self, broker_order_id: str) -> dict:
        return self._orders.get(broker_order_id, {})

    def close_position(self, symbol: str) -> bool:
        pos = self._positions.get(symbol)
        if not pos or pos["qty"] <= 0:
            return False
        order = Order(symbol=symbol, side=Side.SELL, quantity=pos["qty"],
                      limit_price=pos.get("last", pos["avg_entry"]))
        self.submit_order(order)
        return True
