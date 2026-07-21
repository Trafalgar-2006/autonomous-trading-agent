"""Tests for research metrics, fast backtest, and walk-forward harness."""


import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.research import metrics as M
from src.research.backtest import run_fast_backtest
from src.research.walkforward import WalkForward

# ── Metrics ──────────────────────────────────────────────────────

def test_metrics_basic():
    r = M.returns_from_equity([100, 110, 121])  # +10% twice
    m = M.metrics_from_returns(r)
    assert abs(m["total_return"] - 0.21) < 1e-9
    assert m["max_drawdown"] == 0.0
    assert m["win_days_pct"] == 1.0


def test_metrics_drawdown():
    r = M.returns_from_equity([100, 120, 60])  # up then -50%
    m = M.metrics_from_returns(r)
    assert m["max_drawdown"] == 0.5


def test_attribution_and_turnover():
    trades = [
        {"strategy": "momentum", "pnl": 100.0, "exit_price": 10.0, "qty": 5.0},
        {"strategy": "momentum", "pnl": -40.0, "exit_price": 10.0, "qty": 5.0},
        {"strategy": "breakout", "pnl": 20.0, "exit_price": 10.0, "qty": 5.0},
    ]
    attr = M.strategy_attribution(trades)
    assert attr["momentum"]["trades"] == 2
    assert attr["momentum"]["win_rate"] == 0.5
    assert M.turnover(trades, avg_equity=10_000.0, years=1.0) > 0


# ── Fast backtest ────────────────────────────────────────────────

def _trend(n=260, start=50.0, slope=0.25, seed=0):
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    close = start + slope * np.arange(n) + rng.randn(n) * 0.5
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def test_fast_backtest_runs():
    data = {"AAA": _trend(seed=1), "BBB": _trend(seed=2, slope=0.1)}
    res = run_fast_backtest(data, initial_capital=50_000.0)
    assert isinstance(res, BacktestResult)
    assert res.total_trades >= 0
    assert len(res.equity_dates) == len(res.equity_curve)


def test_trade_start_suppresses_early_trades():
    data = {"AAA": _trend(seed=3)}
    start = data["AAA"].index[200]
    res = run_fast_backtest(data, trade_start=start)
    for t in res.trades:
        if t.get("date") is not None:
            assert t["date"] >= start


# ── Walk-forward smoke ───────────────────────────────────────────

class _FakeFeed:
    def get_bars_multi(self, symbols, days=365, timeframe="1Day"):
        out = {}
        for i, s in enumerate(symbols):
            out[s] = _trend(n=420, seed=i + 1, slope=0.15 + 0.02 * i)
        return out


def test_walkforward_runs_end_to_end():
    wf = WalkForward(
        _FakeFeed(), ["AAA", "BBB"], initial_capital=50_000.0,
        train_days=120, test_days=60, warmup_buffer=60, benchmark="SPY",
    )
    result = wf.run(total_days=420)
    assert "strategy" in result
    assert "benchmark" in result
    assert isinstance(result["folds"], list)
    assert len(result["folds"]) >= 1
