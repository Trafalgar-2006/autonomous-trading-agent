"""Smoke + correctness tests for the backtest engine."""

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestEngine, BacktestResult


def trend_series(n=200, start=50.0, slope=0.2, seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    close = start + slope * np.arange(n) + rng.randn(n) * 0.5
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def test_backtest_runs_end_to_end():
    data = {"AAA": trend_series(seed=1, slope=0.3),
            "BBB": trend_series(seed=2, slope=-0.1)}
    engine = BacktestEngine(initial_capital=10_000.0, slippage_bps=5.0)
    result = engine.run(data)

    assert isinstance(result, BacktestResult)
    assert result.initial_capital == 10_000.0
    assert len(result.equity_curve) >= 1
    assert result.total_trades >= 0
    # metrics are finite numbers (not NaN)
    assert result.final_equity == result.final_equity
    assert result.max_drawdown_pct == result.max_drawdown_pct
    assert result.win_rate == result.win_rate


def test_backtest_insufficient_data_returns_flat_result():
    result = BacktestEngine().run({"AAA": trend_series(n=30)})
    assert result.total_trades == 0
    assert result.trades == []


def test_slippage_makes_costs_nonzero_direction():
    # Buy fill must be >= mid, sell fill <= mid (slippage direction sanity).
    engine = BacktestEngine(slippage_bps=10.0)
    assert engine._buy_fill(100.0) > 100.0
    assert engine._sell_fill(100.0) < 100.0
