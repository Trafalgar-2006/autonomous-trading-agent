"""Tests for trade persistence: open-trade round-trip and strategy stats."""

from datetime import datetime

from src.core.models import Side, Trade, TradeOutcome
from src.data.store import DataStore


def make_store(tmp_path):
    return DataStore(db_path=str(tmp_path / "test.db"))


def test_open_trade_roundtrip_preserves_context(tmp_path):
    store = make_store(tmp_path)
    t = Trade(
        symbol="ZZZ", side=Side.BUY, strategy="breakout",
        entry_time=datetime.utcnow(), entry_price=10.0, quantity=3.0,
        entry_reasoning={"atr": 0.5, "take_profit": 12.0, "stop_loss": 9.0},
    )
    store.save_trade(t)

    opens = [x for x in store.get_open_trades() if x.symbol == "ZZZ"]
    assert len(opens) == 1
    r = opens[0]
    assert r.strategy == "breakout"
    assert r.side == Side.BUY
    assert r.entry_price == 10.0
    assert r.entry_reasoning["take_profit"] == 12.0
    assert r.outcome == TradeOutcome.OPEN


def test_closed_trade_excluded_from_open(tmp_path):
    store = make_store(tmp_path)
    t = Trade(symbol="AAA", side=Side.BUY, strategy="momentum",
              entry_time=datetime.utcnow(), entry_price=10.0, quantity=1.0)
    store.save_trade(t)
    t.close(exit_price=11.0, exit_time=datetime.utcnow(), exit_reason="take_profit")
    store.save_trade(t)

    assert store.get_open_trades() == []


def test_strategy_performance_aggregates_wins(tmp_path):
    store = make_store(tmp_path)
    # One winning momentum trade
    win = Trade(symbol="AAA", side=Side.BUY, strategy="momentum",
                entry_time=datetime.utcnow(), entry_price=10.0, quantity=1.0)
    win.close(exit_price=12.0, exit_time=datetime.utcnow())
    store.save_trade(win)
    # One losing momentum trade
    loss = Trade(symbol="BBB", side=Side.BUY, strategy="momentum",
                 entry_time=datetime.utcnow(), entry_price=10.0, quantity=1.0)
    loss.close(exit_price=9.0, exit_time=datetime.utcnow())
    store.save_trade(loss)

    perf = store.get_strategy_performance()
    assert perf["momentum"]["trades"] == 2
    assert perf["momentum"]["wins"] == 1
    assert perf["momentum"]["losses"] == 1
    assert perf["momentum"]["win_rate"] == 0.5
