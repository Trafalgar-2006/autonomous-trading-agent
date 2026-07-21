"""Tests for execution quality: slippage recording and the liquidity guard."""


from src.core.models import Signal, SignalAction
from src.data.store import DataStore
from src.risk.manager import RiskManager


def make_store(tmp_path):
    return DataStore(db_path=str(tmp_path / "exec.db"))


def make_signal(price=100.0, stop=80.0, reasoning=None):
    return Signal(symbol="AAPL", action=SignalAction.BUY, strategy="momentum",
                  confidence=0.8, entry_price=price, stop_loss=stop, take_profit=130.0,
                  reasoning=reasoning if reasoning is not None else {"atr": 2.0})


# ── Slippage ──────────────────────────────────────────────────────

def test_buy_slippage_positive_when_paying_more(tmp_path):
    store = make_store(tmp_path)
    # Expected 100, filled at 100.50 -> paid 50 bps more -> positive slippage
    bps = store.save_fill("AAPL", "buy", "o1", "b1", 100.0, 100.5, 10, "filled")
    assert 49 < bps < 51


def test_sell_slippage_positive_when_receiving_less(tmp_path):
    store = make_store(tmp_path)
    # Expected 100, sold at 99.50 -> received less -> also positive (worse)
    bps = store.save_fill("AAPL", "sell", "o2", "b2", 100.0, 99.5, 10, "filled")
    assert 49 < bps < 51


def test_favourable_fill_is_negative_slippage(tmp_path):
    store = make_store(tmp_path)
    bps = store.save_fill("AAPL", "buy", "o3", "b3", 100.0, 99.5, 10, "filled")
    assert bps < 0


def test_slippage_stats_aggregate(tmp_path):
    store = make_store(tmp_path)
    store.save_fill("AAPL", "buy", "o1", "b1", 100.0, 100.5, 10, "filled")
    store.save_fill("MSFT", "buy", "o2", "b2", 200.0, 201.0, 5, "filled")
    stats = store.get_slippage_stats()
    assert stats["n"] == 2
    assert stats["avg_bps"] > 0


def test_unfilled_order_is_still_recorded(tmp_path):
    store = make_store(tmp_path)
    store.save_fill("AAPL", "buy", "o9", "b9", 100.0, 0.0, 0, "pending_new")
    fills = store.get_fills()
    assert len(fills) == 1
    assert fills[0]["status"] == "pending_new"


# ── Liquidity guard ───────────────────────────────────────────────

def test_liquidity_guard_caps_position():
    r = RiskManager()
    # Thin name: $1M ADV, 1% cap -> at most $10k -> 100 shares at $100.
    thin = make_signal(reasoning={"atr": 2.0, "adv_notional": 1_000_000.0})
    order = r.evaluate(thin, [], equity=10_000_000.0, cash=10_000_000.0)
    assert order is not None
    assert order.quantity <= (1_000_000.0 * r.config.max_adv_pct) / 100.0 + 1e-6


def test_liquid_name_not_capped_by_liquidity():
    r = RiskManager()
    thin = make_signal(reasoning={"atr": 2.0, "adv_notional": 1_000_000.0})
    deep = make_signal(reasoning={"atr": 2.0, "adv_notional": 10_000_000_000.0})
    o_thin = r.evaluate(thin, [], 10_000_000.0, 10_000_000.0)
    o_deep = r.evaluate(deep, [], 10_000_000.0, 10_000_000.0)
    assert o_thin is not None and o_deep is not None
    assert o_deep.quantity > o_thin.quantity


def test_no_adv_means_no_liquidity_cap():
    r = RiskManager()
    # Without adv_notional the guard must not fire (and must not crash).
    order = r.evaluate(make_signal(), [], 100_000.0, 100_000.0)
    assert order is not None and order.quantity > 0
