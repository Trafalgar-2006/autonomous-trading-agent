"""Tests for skip-month momentum ranking and batch-level concentration."""

import asyncio

import numpy as np
import pandas as pd

from src.core.models import DecisionStatus, OrderStatus, SignalAction
from src.strategy.xs_momentum import CrossSectionalMomentum


def series(prices):
    idx = pd.date_range("2022-01-01", periods=len(prices), freq="B")
    return pd.DataFrame(
        {"open": prices, "high": [p + 1 for p in prices],
         "low": [p - 1 for p in prices], "close": prices,
         "volume": [1e6] * len(prices), "atr": [1.0] * len(prices)},
        index=idx,
    )


# ── Skip-month ranking ────────────────────────────────────────────

def test_skip_month_ignores_recent_reversal():
    # WINNER: strong 12m gain, then a sharp -20% last month (skip should ignore it).
    winner = list(np.linspace(50, 150, 260)) + list(np.linspace(150, 120, 21))
    # STEADY: modest, smooth gain the whole way, no late drop.
    steady = list(np.linspace(100, 118, 281))
    data = {"WIN": series(winner), "STEADY": series(steady)}

    skipped = CrossSectionalMomentum(lookback=252, skip=21).rank(data)
    # Skip-month looks past WIN's recent -20% and keeps it on top by its
    # strong 12-month trend.
    assert skipped[0][0] == "WIN"


def test_skip_requires_more_history():
    short = {"A": series(list(np.linspace(100, 110, 260)))}  # 260 bars
    # lookback 252 + skip 21 = 273 needed -> not enough -> no ranking
    assert CrossSectionalMomentum(lookback=252, skip=21).rank(short) == []
    # skip 0 needs only 253 -> ranks fine
    assert len(CrossSectionalMomentum(lookback=252, skip=0).rank(short)) == 1


# ── Batch-level concentration ─────────────────────────────────────

class _FakeBroker:
    def __init__(self, equity=100_000.0):
        self._equity = equity
        self.submitted = []

    def get_account(self):
        return {"equity": self._equity, "cash": self._equity, "status": "ACTIVE"}

    def get_positions(self):
        return []

    def submit_order(self, order):
        self.submitted.append(order)
        order.status = OrderStatus.SUBMITTED
        order.broker_order_id = f"f{len(self.submitted)}"
        return order

    def get_order(self, _):
        return {"status": "filled", "filled_qty": 1.0, "filled_avg_price": 100.0}


def _order_manager(tmp_path):
    from src.core.config import Config
    from src.core.event_bus import EventBus
    from src.data.store import DataStore
    from src.decision.engine import DecisionEngine
    from src.execution.order_manager import OrderManager
    from src.risk.manager import RiskManager

    om = OrderManager.__new__(OrderManager)
    om.config = Config()
    om.store = DataStore(db_path=str(tmp_path / "batch.db"))
    om.bus = EventBus()
    om.risk_manager = RiskManager()
    om.broker = _FakeBroker()
    om.decision_engine = DecisionEngine(risk_manager=om.risk_manager)
    om._active_trades = {}
    return om


def buy(symbol):
    from src.core.models import Signal
    return Signal(symbol=symbol, action=SignalAction.BUY, strategy="xs_momentum",
                  confidence=0.9, entry_price=100.0, stop_loss=95.0, take_profit=115.0,
                  reasoning={"atr": 2.0})


def test_batch_caps_max_positions_within_one_call(tmp_path):
    om = _order_manager(tmp_path)
    # 20 fresh BUYs, no held positions. max_open_positions is 8 -> at most 8
    # should be APPROVED even though they arrive in a single batch.
    signals = [buy(f"SYM{i}") for i in range(20)]
    memos = asyncio.run(om.process_signals(signals))
    approved = [m for m in memos if m.status == DecisionStatus.APPROVED]
    assert len(approved) <= om.config.max_open_positions


def test_batch_sector_cap_engages_within_call(tmp_path):
    om = _order_manager(tmp_path)
    # All financials: JPM, BAC, GS, MS, WFC, C -> should not all be approved,
    # the sector cap must engage as the batch accumulates.
    banks = ["JPM", "BAC", "GS", "MS", "WFC", "C", "AXP", "SCHW"]
    memos = asyncio.run(om.process_signals([buy(s) for s in banks]))
    approved = [m for m in memos if m.status == DecisionStatus.APPROVED]
    watchlisted = [m for m in memos if m.status == DecisionStatus.WATCHLIST]
    # At least one bank must be demoted for sector concentration.
    assert len(approved) < len(banks)
    assert any("sector" in r for m in watchlisted for r in m.reasons)
