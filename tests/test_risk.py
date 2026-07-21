"""Tests for RiskManager: position sizing, gating rules, circuit breakers."""

from datetime import datetime

from src.core.models import Position, Side, Signal, SignalAction
from src.risk.manager import RiskManager


def make_signal(action=SignalAction.BUY, symbol="AAPL", price=100.0, stop=95.0):
    return Signal(
        symbol=symbol, action=action, strategy="momentum", confidence=0.8,
        entry_price=price, stop_loss=stop, take_profit=110.0,
        reasoning={"atr": 2.0},
    )


def make_position(symbol="AAPL", qty=1.0, price=100.0):
    return Position(symbol=symbol, side=Side.BUY, quantity=qty,
                    entry_price=price, entry_time=datetime.utcnow(),
                    current_price=price)


def test_buy_produces_positive_sized_order():
    r = RiskManager()
    order = r.evaluate(make_signal(), positions=[], equity=100_000.0, cash=100_000.0)
    assert order is not None
    assert order.side == Side.BUY
    assert order.quantity > 0


def test_position_below_min_value_is_rejected():
    r = RiskManager()
    # With tiny equity the sized position can't meet min_position_value.
    order = r.evaluate(make_signal(price=100.0), positions=[], equity=50.0, cash=50.0)
    assert order is None


def test_reject_when_already_holding_symbol():
    r = RiskManager()
    order = r.evaluate(make_signal(symbol="AAPL"), [make_position("AAPL")],
                       100_000.0, 100_000.0)
    assert order is None


def test_reject_when_max_positions_reached():
    r = RiskManager()
    n = r.config.max_open_positions
    positions = [make_position(symbol=f"S{i}") for i in range(n)]
    order = r.evaluate(make_signal(symbol="NEW"), positions, 100_000.0, 100_000.0)
    assert order is None


def test_sell_without_position_returns_none():
    r = RiskManager()
    order = r.evaluate(make_signal(action=SignalAction.SELL, symbol="AAPL"),
                       [], 100_000.0, 100_000.0)
    assert order is None


def test_sell_with_position_sells_full_quantity():
    r = RiskManager()
    pos = make_position("AAPL", qty=7.5)
    order = r.evaluate(make_signal(action=SignalAction.SELL, symbol="AAPL"),
                       [pos], 100_000.0, 100_000.0)
    assert order is not None
    assert order.side == Side.SELL
    assert order.quantity == 7.5


def test_size_mult_scales_position():
    r = RiskManager()
    # Wide stop so the risk-based size (not the capital cap) is binding.
    full = make_signal(price=100.0, stop=80.0)
    full.reasoning = {"atr": 2.0, "size_mult": 1.0}
    half = make_signal(price=100.0, stop=80.0)
    half.reasoning = {"atr": 2.0, "size_mult": 0.5}
    o_full = r.evaluate(full, [], 100_000.0, 100_000.0)
    o_half = r.evaluate(half, [], 100_000.0, 100_000.0)
    assert o_full is not None and o_half is not None
    assert o_half.quantity < o_full.quantity
    assert abs(o_half.quantity * 2 - o_full.quantity) < 1e-6


def test_circuit_breaker_ignores_profit():
    r = RiskManager()
    eq = r.config.initial_capital
    r.update_daily_pnl(eq * 0.5)  # huge gain
    assert r.status["cooldown_active"] is False


def test_circuit_breaker_trips_on_daily_loss():
    r = RiskManager()
    eq = r.config.initial_capital
    r.update_daily_pnl(-eq * (r.config.max_daily_loss + 0.01))
    assert r.status["cooldown_active"] is True


def test_circuit_breaker_trips_on_consecutive_losses():
    r = RiskManager()
    for _ in range(r.config.max_consecutive_losses):
        r.update_daily_pnl(-1.0)
    assert r.status["cooldown_active"] is True


def test_no_orders_during_cooldown():
    r = RiskManager()
    r.update_daily_pnl(-r.config.initial_capital)  # trips the breaker
    order = r.evaluate(make_signal(), [], 100_000.0, 100_000.0)
    assert order is None
