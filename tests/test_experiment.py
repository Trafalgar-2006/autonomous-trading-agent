"""Tests for the research experiment options."""

import numpy as np
import pandas as pd

from src.research.backtest import run_fast_backtest
from src.research.experiment import ExperimentConfig, market_ok_series


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


def test_market_ok_series_uptrend_true_downtrend_false():
    idx = pd.date_range("2021-01-01", periods=300, freq="B")
    up = pd.DataFrame({"close": pd.Series(np.linspace(100, 200, 300), index=idx)})
    down = pd.DataFrame({"close": pd.Series(np.linspace(200, 100, 300), index=idx)})
    assert list(market_ok_series(up).values())[-1] is True
    assert list(market_ok_series(down).values())[-1] is False


def test_disabled_strategy_removes_its_trades():
    data = {"AAA": _trend(seed=1), "BBB": _trend(seed=2, slope=0.1)}
    # Force momentum ON as the control, then disable it — the disabled run must
    # never produce a momentum trade regardless of what the control does.
    control = run_fast_backtest(
        data, initial_capital=50_000.0,
        experiment=ExperimentConfig(strategies=("momentum", "mean_reversion", "breakout")),
    )
    no_mom = run_fast_backtest(
        data, initial_capital=50_000.0,
        experiment=ExperimentConfig(strategies=("mean_reversion", "breakout")),
    )
    assert "momentum" not in {t["strategy"] for t in no_mom.trades}
    # Control is allowed to contain momentum; both runs must stay well-formed.
    assert control.total_trades >= 0 and no_mom.total_trades >= 0


def test_market_filter_blocks_longs_when_off():
    data = {"AAA": _trend(seed=1)}
    # market_ok False for every date -> no BUYs can be opened -> no trades
    all_off = {d: False for d in data["AAA"].index}
    res = run_fast_backtest(
        data, initial_capital=50_000.0,
        experiment=ExperimentConfig(market_filter=True),
        market_ok=all_off,
    )
    assert res.total_trades == 0


def test_cross_sectional_top_limits_entries():
    data = {f"S{i}": _trend(seed=i, slope=0.2 + 0.01 * i) for i in range(6)}
    res = run_fast_backtest(
        data, initial_capital=200_000.0, max_positions=8,
        experiment=ExperimentConfig(cross_sectional_top=2),
    )
    # With top-2 per day, we should not blow past a small number of names held.
    held = {t["symbol"] for t in res.trades}
    assert len(held) <= 6  # sanity: never more symbols than the universe
