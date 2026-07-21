"""Tests for market-data quality checks and the research tearsheet."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestResult
from src.data.quality import filter_tradeable, validate_bars
from src.research.metrics import format_tearsheet, tearsheet


def good_bars(n=60, end=None):
    end = end or datetime.utcnow()
    idx = pd.date_range(end=end, periods=n, freq="B")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def codes(issues):
    return {i["code"] for i in issues}


# ── Data quality ──────────────────────────────────────────────────

def test_clean_data_has_no_issues():
    assert validate_bars(good_bars()) == []


def test_empty_data_is_critical():
    issues = validate_bars(pd.DataFrame())
    assert "empty" in codes(issues)
    assert issues[0]["severity"] == "critical"


def test_zero_price_is_critical():
    df = good_bars()
    df.iloc[5, df.columns.get_loc("close")] = 0.0
    issues = validate_bars(df)
    assert "non_positive_price" in codes(issues)


def test_high_below_low_is_critical():
    df = good_bars()
    df.iloc[3, df.columns.get_loc("high")] = 1.0  # below low
    assert "high_below_low" in codes(validate_bars(df))


def test_close_outside_range_is_critical():
    df = good_bars()
    df.iloc[4, df.columns.get_loc("close")] = 9_999.0
    assert "close_outside_range" in codes(validate_bars(df))


def test_stale_data_is_critical():
    df = good_bars(end=datetime.utcnow() - timedelta(days=30))
    assert "stale" in codes(validate_bars(df))


def _spike_bar(df, row=10, factor=3.0):
    """Scale a whole bar so it's a big jump but still valid OHLC."""
    df = df.copy()
    for col in ("open", "high", "low", "close"):
        df.iloc[row, df.columns.get_loc(col)] = df[col].iloc[row] * factor
    return df


def test_extreme_jump_is_warning_not_critical():
    issues = validate_bars(_spike_bar(good_bars()))
    assert "extreme_jump" in codes(issues)
    jump = next(i for i in issues if i["code"] == "extreme_jump")
    assert jump["severity"] == "warning"
    # A price spike alone must not be treated as unusable data.
    assert not any(i["severity"] == "critical" for i in issues)


def test_missing_columns_is_critical():
    assert "missing_columns" in codes(validate_bars(pd.DataFrame({"close": [1, 2]})))


def test_filter_tradeable_drops_only_critical():
    good = good_bars()
    stale = good_bars(end=datetime.utcnow() - timedelta(days=30))  # critical
    noisy = _spike_bar(good_bars())  # warning only (valid OHLC, big move)

    clean, issues = filter_tradeable({"GOOD": good, "STALE": stale, "NOISY": noisy})
    assert set(clean) == {"GOOD", "NOISY"}   # warnings survive, criticals dropped
    assert any(i["code"] == "stale" for i in issues)


# ── Tearsheet ─────────────────────────────────────────────────────

def _result():
    curve = [100_000 * (1.001 ** i) for i in range(260)]
    return BacktestResult(
        initial_capital=100_000.0,
        final_equity=curve[-1],
        equity_curve=curve,
        trades=[
            {"symbol": "AAA", "strategy": "momentum", "pnl": 500.0,
             "exit_price": 10.0, "qty": 10},
            {"symbol": "BBB", "strategy": "momentum", "pnl": -200.0,
             "exit_price": 10.0, "qty": 10},
            {"symbol": "CCC", "strategy": "breakout", "pnl": 300.0,
             "exit_price": 10.0, "qty": 10},
        ],
    )


def test_tearsheet_core_fields():
    sheet = tearsheet(_result())
    assert sheet["trades"]["total"] == 3
    assert sheet["trades"]["wins"] == 2
    assert sheet["trades"]["profit_factor"] > 1
    assert "momentum" in sheet["attribution"]
    assert sheet["metrics"]["cagr"] > 0


def test_tearsheet_benchmark_comparison():
    flat = np.zeros(260)                      # benchmark that goes nowhere
    sheet = tearsheet(_result(), benchmark_returns=flat)
    assert "benchmark" in sheet and "excess" in sheet
    assert sheet["beat_benchmark"] is True    # rising curve beats a flat one


def test_format_tearsheet_renders():
    text = format_tearsheet(tearsheet(_result()))
    assert "CAGR" in text and "Attribution" in text
