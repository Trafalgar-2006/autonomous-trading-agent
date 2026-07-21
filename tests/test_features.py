"""Tests for the technical-indicator FeatureEngine."""

import numpy as np
import pandas as pd

from src.data.features import FeatureEngine


def synth_ohlcv(n=250, seed=0):
    idx = pd.date_range("2022-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    close = 100 + np.cumsum(rng.randn(n))
    close = np.maximum(close, 1.0)
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def test_compute_all_adds_expected_columns():
    df = FeatureEngine().compute_all(synth_ohlcv())
    for col in ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
                "atr", "adx", "zscore", "donchian_high", "volatility_20d",
                "volume_ratio", "return_1d"]:
        assert col in df.columns, f"missing {col}"


def test_rsi_within_bounds():
    df = FeatureEngine().compute_all(synth_ohlcv())
    rsi = df["rsi_14"].dropna()
    assert not rsi.empty
    assert (rsi >= 0).all() and (rsi <= 100).all()


def test_empty_input_returns_empty():
    out = FeatureEngine().compute_all(pd.DataFrame())
    assert out.empty


def test_none_input_returns_none():
    assert FeatureEngine().compute_all(None) is None
