"""Tests for the local PaperBroker simulation."""

from src.core.models import Order, OrderStatus, Side
from src.execution.base import BaseBroker
from src.execution.paper_broker import PaperBroker


def broker(tmp_path, slippage_bps=0.0):
    return PaperBroker(slippage_bps=slippage_bps, state_path=tmp_path / "p.json")


def test_implements_interface(tmp_path):
    assert isinstance(broker(tmp_path), BaseBroker)


def test_buy_reduces_cash_and_creates_position(tmp_path):
    b = broker(tmp_path)
    start = b.get_account()["cash"]
    b.submit_order(Order(symbol="AAPL", side=Side.BUY, quantity=10, limit_price=100.0))
    assert b.get_account()["cash"] == start - 1000.0
    pos = b.get_positions()
    assert len(pos) == 1 and pos[0].symbol == "AAPL" and pos[0].quantity == 10


def test_mark_to_market_moves_equity(tmp_path):
    b = broker(tmp_path)
    b.submit_order(Order(symbol="AAPL", side=Side.BUY, quantity=10, limit_price=100.0))
    eq_before = b.get_account()["equity"]
    b.mark_prices({"AAPL": 110.0})
    assert b.get_account()["equity"] == eq_before + 100.0  # 10 sh * $10


def test_sell_realizes_pnl(tmp_path):
    b = broker(tmp_path)
    start = b.get_account()["cash"]
    b.submit_order(Order(symbol="AAPL", side=Side.BUY, quantity=10, limit_price=100.0))
    b.mark_prices({"AAPL": 120.0})
    b.close_position("AAPL")
    assert abs(b.get_account()["cash"] - (start + 200.0)) < 1e-6
    assert b.get_positions() == []


def test_cannot_spend_more_than_cash(tmp_path):
    b = broker(tmp_path)
    cash = b.get_account()["cash"]
    # Try to buy way more than we can afford — qty gets clamped, cash floors at ~0.
    b.submit_order(Order(symbol="AAPL", side=Side.BUY,
                         quantity=cash / 100.0 * 5, limit_price=100.0))
    assert b.get_account()["cash"] >= -1e-6


def test_slippage_makes_buys_cost_more(tmp_path):
    b = broker(tmp_path, slippage_bps=50.0)
    start = b.get_account()["cash"]
    b.submit_order(Order(symbol="AAPL", side=Side.BUY, quantity=10, limit_price=100.0))
    # Paid more than 1000 because of slippage.
    assert b.get_account()["cash"] < start - 1000.0


def test_order_status_and_fill_report(tmp_path):
    b = broker(tmp_path)
    o = Order(symbol="MSFT", side=Side.BUY, quantity=5, limit_price=200.0)
    b.submit_order(o)
    assert o.status == OrderStatus.FILLED and o.broker_order_id
    info = b.get_order(o.broker_order_id)
    assert info["status"] == "filled" and info["filled_avg_price"] == 200.0


def test_state_persists_across_instances(tmp_path):
    sp = tmp_path / "p.json"
    b1 = PaperBroker(slippage_bps=0.0, state_path=sp)
    b1.submit_order(Order(symbol="AAPL", side=Side.BUY, quantity=10, limit_price=100.0))
    # A fresh instance reads the saved state (survives restart).
    b2 = PaperBroker(slippage_bps=0.0, state_path=sp)
    assert len(b2.get_positions()) == 1
    assert b2.get_positions()[0].symbol == "AAPL"


def test_selling_unheld_is_rejected(tmp_path):
    b = broker(tmp_path)
    o = Order(symbol="NVDA", side=Side.SELL, quantity=5, limit_price=100.0)
    b.submit_order(o)
    assert o.status == OrderStatus.REJECTED
