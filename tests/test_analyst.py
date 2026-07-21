"""Tests for the AI Analyst (prompt builders + graceful degradation)."""

from datetime import datetime

from src.monitoring.analyst import (
    AIAnalyst, build_morning_prompt, build_eod_prompt, build_decision_prompt,
)
from src.core.models import Position, Side, DecisionMemo, DecisionStatus, SignalAction


def _pos(sym="AAPL", qty=10.0, price=100.0):
    return Position(symbol=sym, side=Side.BUY, quantity=qty, entry_price=price,
                    entry_time=datetime.utcnow(), current_price=price)


def test_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = AIAnalyst()
    assert a.enabled is False


def test_disabled_methods_return_none(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a = AIAnalyst()
    import asyncio
    assert asyncio.run(a.morning_brief({}, [], [])) is None
    assert asyncio.run(a.eod_narrative({}, [], {}, {})) is None


def test_morning_prompt_includes_positions_and_candidates():
    p = build_morning_prompt(
        {"equity": 100_000, "cash": 40_000},
        [_pos("AAPL"), _pos("MSFT")],
        [{"status": "approved", "action": "buy", "symbol": "NVDA",
          "strategy": "xs_momentum", "risk_reward": 1.8}],
    )
    assert "MORNING BRIEF" in p
    assert "AAPL" in p and "MSFT" in p
    assert "NVDA" in p


def test_eod_prompt_includes_key_numbers():
    p = build_eod_prompt(
        {"equity": 99_000}, [_pos()],
        {"daily_pnl": -500.0, "cooldown_active": False},
        {"total_trades": 10, "wins": 6, "total_pnl": 1200.0},
    )
    assert "END-OF-DAY" in p
    assert "99,000" in p


def test_decision_prompt_from_memo():
    memo = DecisionMemo(
        symbol="AMD", action=SignalAction.BUY, status=DecisionStatus.APPROVED,
        strategy="breakout", signal_strength=0.8, entry=100.0, stop=95.0,
        risk_reward=2.0, reasons=["passes risk, R:R and confidence gates"],
    )
    p = build_decision_prompt(memo)
    assert "AMD" in p and "APPROVED" in p and "breakout" in p
