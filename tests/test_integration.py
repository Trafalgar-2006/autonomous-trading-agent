"""
Integration tests — the signal → decision → risk → order → fill pipeline
exercised end-to-end against a fake broker (no network, no real account).

These cover the wiring that unit tests miss: that an approved signal actually
reaches the broker, that a rejected one doesn't, that propose mode never
trades, and that fills/slippage get recorded.
"""

import asyncio
from datetime import datetime

import pytest

from src.core.models import (
    DecisionStatus,
    OrderStatus,
    Position,
    Side,
    Signal,
    SignalAction,
)
from src.data.store import DataStore
from src.execution.order_manager import OrderManager


class FakeBroker:
    """Stand-in for AlpacaBroker: records submissions, fills at a set price."""

    def __init__(self, equity=100_000.0, positions=None, fill_price=100.5):
        self._equity = equity
        self._positions = positions or []
        self._fill_price = fill_price
        self.submitted = []
        self.closed = []

    def get_account(self):
        return {"equity": self._equity, "cash": self._equity, "status": "ACTIVE"}

    def get_positions(self):
        return list(self._positions)

    def submit_order(self, order):
        self.submitted.append(order)
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = f"fake-{len(self.submitted)}"
        return order

    def get_order(self, broker_order_id):
        return {"status": "filled", "filled_qty": 10.0,
                "filled_avg_price": self._fill_price}

    def close_position(self, symbol):
        self.closed.append(symbol)
        return True


@pytest.fixture
def om(tmp_path, monkeypatch):
    """OrderManager wired to a fake broker and an isolated database."""
    manager = OrderManager.__new__(OrderManager)  # skip __init__ (no real broker)
    from src.core.config import Config
    from src.core.event_bus import EventBus
    from src.decision.engine import DecisionEngine
    from src.risk.manager import RiskManager

    manager.config = Config()
    manager.store = DataStore(db_path=str(tmp_path / "integration.db"))
    manager.bus = EventBus()
    manager.risk_manager = RiskManager()
    manager.broker = FakeBroker()
    manager.decision_engine = DecisionEngine(risk_manager=manager.risk_manager)
    manager._active_trades = {}
    return manager


def buy_signal(symbol="AAPL", price=100.0, stop=95.0, target=110.0, confidence=0.8):
    return Signal(symbol=symbol, action=SignalAction.BUY, strategy="momentum",
                  confidence=confidence, entry_price=price, stop_loss=stop,
                  take_profit=target, reasoning={"atr": 2.0})


def test_approved_signal_reaches_the_broker(om):
    memos = asyncio.run(om.process_signals([buy_signal()]))
    assert len(memos) == 1
    assert memos[0].status == DecisionStatus.APPROVED
    assert len(om.broker.submitted) == 1
    assert om.broker.submitted[0].symbol == "AAPL"


def test_watchlist_signal_is_not_traded(om):
    # Poor reward:risk -> WATCHLIST -> must not submit an order.
    memos = asyncio.run(om.process_signals([buy_signal(target=101.0)]))
    assert memos[0].status == DecisionStatus.WATCHLIST
    assert om.broker.submitted == []


def test_propose_mode_never_trades(om, monkeypatch):
    monkeypatch.setattr(type(om.config), "execution_mode",
                        property(lambda self: "propose"))
    memos = asyncio.run(om.process_signals([buy_signal()]))
    assert memos[0].status == DecisionStatus.APPROVED   # still approved...
    assert om.broker.submitted == []                     # ...but not executed


def test_fill_and_slippage_are_recorded(om):
    asyncio.run(om.process_signals([buy_signal()]))
    fills = om.store.get_fills()
    assert len(fills) == 1
    # Expected 100.00, filled 100.50 -> ~50 bps of adverse slippage.
    assert 49 < fills[0]["slippage_bps"] < 51
    assert fills[0]["status"] == "filled"


def test_decision_is_persisted_even_when_rejected(om):
    held = Position(symbol="AAPL", side=Side.BUY, quantity=5, entry_price=100,
                    entry_time=datetime.utcnow(), current_price=100)
    om.broker._positions = [held]
    memos = asyncio.run(om.process_signals([buy_signal("AAPL")]))
    assert memos[0].status == DecisionStatus.REJECTED   # already holding
    assert om.broker.submitted == []
    assert len(om.store.get_decisions()) == 1


def test_trade_is_tracked_after_buy(om):
    asyncio.run(om.process_signals([buy_signal("MSFT")]))
    assert "MSFT" in om._active_trades
    assert om._active_trades["MSFT"].strategy == "momentum"
