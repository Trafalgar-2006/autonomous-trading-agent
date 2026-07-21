"""Tests for the DecisionEngine (seb.ai-style memo funnel)."""

from datetime import datetime

from src.decision.engine import DecisionEngine, compute_risk_reward
from src.risk.manager import RiskManager
from src.core.models import (
    Signal, SignalAction, Position, Side, DecisionStatus,
)


def make_signal(action=SignalAction.BUY, symbol="AAPL", price=100.0,
                stop=95.0, target=110.0, confidence=0.8, reasoning=None):
    return Signal(
        symbol=symbol, action=action, strategy="momentum", confidence=confidence,
        entry_price=price, stop_loss=stop, take_profit=target,
        reasoning=reasoning if reasoning is not None else {"atr": 2.0},
    )


def make_position(symbol="AAPL", qty=1.0, price=100.0):
    return Position(symbol=symbol, side=Side.BUY, quantity=qty,
                    entry_price=price, entry_time=datetime.utcnow(),
                    current_price=price)


def test_compute_risk_reward():
    assert compute_risk_reward(100, 95, 110) == 2.0
    assert compute_risk_reward(100, 100, 110) is None
    assert compute_risk_reward(100, 95, None) is None


def test_strong_setup_is_approved():
    eng = DecisionEngine(RiskManager())
    # entry 100, stop 95, target 110 -> R:R 2.0; confidence 0.8 (strong)
    memo = eng.build(make_signal(), positions=[], equity=100_000.0, cash=100_000.0)
    assert memo.status == DecisionStatus.APPROVED
    assert memo.risk_reward == 2.0
    assert memo.quantity > 0
    assert memo.dollar_risk > 0


def test_low_rr_goes_to_watchlist():
    eng = DecisionEngine(RiskManager())
    # target only 1 above entry, risk 5 -> R:R 0.2 < 1.5 -> WATCHLIST
    memo = eng.build(make_signal(target=101.0), positions=[], equity=100_000.0, cash=100_000.0)
    assert memo.status == DecisionStatus.WATCHLIST
    assert any("R:R" in r for r in memo.reasons)


def test_low_confidence_goes_to_watchlist():
    eng = DecisionEngine(RiskManager())
    memo = eng.build(make_signal(confidence=0.5), positions=[], equity=100_000.0, cash=100_000.0)
    assert memo.status == DecisionStatus.WATCHLIST
    assert any("confidence" in r for r in memo.reasons)


def test_rr_exactly_at_threshold_is_approved():
    # R:R sitting exactly on the min (float-noise just below) must not be
    # demoted to WATCHLIST — momentum's natural R:R is exactly 1.5.
    eng = DecisionEngine(RiskManager())
    min_rr = eng.config.min_risk_reward  # 1.5
    # entry 100, stop 90 (risk 10), target = 100 + 10*min_rr -> R:R == min_rr
    target = 100 + 10 * min_rr
    memo = eng.build(make_signal(price=100.0, stop=90.0, target=target),
                     positions=[], equity=100_000.0, cash=100_000.0)
    assert memo.status == DecisionStatus.APPROVED


def test_already_holding_is_rejected():
    eng = DecisionEngine(RiskManager())
    memo = eng.build(make_signal(symbol="AAPL"), [make_position("AAPL")],
                     100_000.0, 100_000.0)
    assert memo.status == DecisionStatus.REJECTED
    assert any("already holding" in r for r in memo.reasons)


def test_circuit_breaker_rejects():
    rm = RiskManager()
    rm.update_daily_pnl(-rm.config.initial_capital)  # trip breaker
    eng = DecisionEngine(rm)
    memo = eng.build(make_signal(), [], 100_000.0, 100_000.0)
    assert memo.status == DecisionStatus.REJECTED
    assert any("circuit breaker" in r for r in memo.reasons)


def test_sell_with_position_is_approved_exit():
    eng = DecisionEngine(RiskManager())
    memo = eng.build(make_signal(action=SignalAction.SELL, symbol="AAPL"),
                     [make_position("AAPL", qty=3.0)], 100_000.0, 100_000.0)
    assert memo.status == DecisionStatus.APPROVED
    assert memo.quantity == 3.0


def test_sell_without_position_is_rejected():
    eng = DecisionEngine(RiskManager())
    memo = eng.build(make_signal(action=SignalAction.SELL, symbol="AAPL"),
                     [], 100_000.0, 100_000.0)
    assert memo.status == DecisionStatus.REJECTED


def test_invalidation_uses_donchian_low_for_breakout():
    eng = DecisionEngine(RiskManager())
    sig = make_signal(reasoning={"atr": 2.0, "donchian_low": 90.0})
    memo = eng.build(sig, [], 100_000.0, 100_000.0)
    assert memo.invalidation == 90.0


def test_market_filter_demotes_long_to_watchlist():
    eng = DecisionEngine(RiskManager())
    if not eng.config.market_filter:
        import pytest
        pytest.skip("market filter disabled in config")
    # Strong setup, but SPY below its SMA -> WATCHLIST, not APPROVED.
    memo = eng.build(make_signal(), positions=[], equity=100_000.0, cash=100_000.0,
                     market_ok=False)
    assert memo.status == DecisionStatus.WATCHLIST
    assert any("market filter" in r for r in memo.reasons)


def test_market_ok_true_still_approves():
    eng = DecisionEngine(RiskManager())
    memo = eng.build(make_signal(), positions=[], equity=100_000.0, cash=100_000.0,
                     market_ok=True)
    assert memo.status == DecisionStatus.APPROVED


def test_correlation_filter_demotes_overconcentrated_long():
    eng = DecisionEngine(RiskManager())
    # Hold a position (40% of equity) highly correlated with the new BUY.
    # Below the 80% total-exposure cap, but above the 30% correlated cap.
    held = make_position("QQQ", qty=100.0, price=400.0)  # market_value 40k
    corr = {"AAPL": {"QQQ": 0.95}}
    memo = eng.build(make_signal(symbol="AAPL"), [held],
                     equity=100_000.0, cash=100_000.0, correlations=corr)
    # correlated exposure 40k/100k = 40% >= 30% limit -> WATCHLIST
    assert memo.status == DecisionStatus.WATCHLIST
    assert any("correlated exposure" in r for r in memo.reasons)


def test_low_correlation_still_approves():
    eng = DecisionEngine(RiskManager())
    held = make_position("XOM", qty=100.0, price=100.0)
    corr = {"AAPL": {"XOM": 0.1}}  # uncorrelated
    memo = eng.build(make_signal(symbol="AAPL"), [held],
                     equity=100_000.0, cash=100_000.0, correlations=corr)
    assert memo.status == DecisionStatus.APPROVED


def test_memo_renders_without_error():
    eng = DecisionEngine(RiskManager())
    memo = eng.build(make_signal(), [], 100_000.0, 100_000.0)
    text = memo.render()
    assert "DECISION MEMO" in text
    assert "PAPER ONLY" in text
