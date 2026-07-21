"""Tests for the live cross-sectional momentum strategy."""

import numpy as np
import pandas as pd

from src.strategy.xs_momentum import CrossSectionalMomentum
from src.core.models import SignalAction


def _series(n=300, slope=0.05, seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    c = np.maximum(50 + slope * np.arange(n) + rng.randn(n) * 0.3, 1.0)
    return pd.DataFrame(
        {"open": c, "high": c + 0.3, "low": c - 0.3, "close": c,
         "volume": 1e6, "atr": 1.0, "volatility_20d": 0.25},
        index=idx,
    )


def _universe():
    # S0 weakest ... S4 strongest momentum
    return {f"S{i}": _series(slope=0.05 * i, seed=i) for i in range(5)}


def test_rank_orders_by_trailing_return():
    xs = CrossSectionalMomentum(lookback=252, top_n=2)
    ranked = xs.rank(_universe())
    syms = [s for s, _ in ranked]
    assert syms[0] == "S4" and syms[1] == "S3"  # strongest first


def test_buys_top_n_not_held():
    xs = CrossSectionalMomentum(lookback=252, top_n=2)
    sigs = xs.build_signals(_universe(), held=set())
    buys = [s.symbol for s in sigs if s.action == SignalAction.BUY]
    assert buys == ["S4", "S3"]
    # BUYs carry a stop and target
    buy = next(s for s in sigs if s.action == SignalAction.BUY)
    assert buy.stop_loss < buy.entry_price < buy.take_profit


def test_rotation_sells_held_that_fell_out():
    xs = CrossSectionalMomentum(lookback=252, top_n=2)
    sigs = xs.build_signals(_universe(), held={"S0"})
    sells = [s.symbol for s in sigs if s.action == SignalAction.SELL]
    assert sells == ["S0"]  # weakest, not in top-2 -> rotate out


def test_held_top_name_not_rebought():
    xs = CrossSectionalMomentum(lookback=252, top_n=2)
    sigs = xs.build_signals(_universe(), held={"S4"})
    buys = [s.symbol for s in sigs if s.action == SignalAction.BUY]
    assert "S4" not in buys  # already held
    assert "S3" in buys


def test_vol_target_sets_size_mult():
    xs = CrossSectionalMomentum(lookback=252, top_n=2, vol_target=0.20)
    sigs = xs.build_signals(_universe(), held=set())
    buy = next(s for s in sigs if s.action == SignalAction.BUY)
    # vol 0.25 vs target 0.20 -> size_mult = 0.8
    assert abs(buy.reasoning.get("size_mult", 1.0) - 0.8) < 1e-9
