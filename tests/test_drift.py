"""Tests for forward-test drift detection."""

from datetime import datetime, timedelta

from src.research.drift import (
    MIN_DAYS_FOR_VERDICT,
    MIN_TRADES_FOR_VERDICT,
    compare_to_baseline,
    elapsed_trading_days,
    format_drift,
    live_metrics_from_snapshots,
    sharpe_standard_error,
)

BASELINE = {"cagr": 0.264, "sharpe": 0.78, "max_drawdown": 0.30}


def snapshots(n=400, daily_return=0.0005, start_equity=100_000.0):
    """Synthetic equity snapshots, one per day."""
    out = []
    equity = start_equity
    t0 = datetime.utcnow() - timedelta(days=n)
    for i in range(n):
        out.append({"timestamp": (t0 + timedelta(days=i)).isoformat(),
                    "equity": equity})
        equity *= (1 + daily_return)
    return out


# ── Sample-size honesty ───────────────────────────────────────────

def test_too_early_with_small_sample():
    r = compare_to_baseline({"sharpe": 3.0, "cagr": 1.0, "max_drawdown": 0.01},
                            BASELINE, n_days=10, n_trades=3)
    # Spectacular early numbers must NOT be read as success.
    assert r["verdict"] == "too_early"
    assert r["sufficient"] is False


def test_too_early_when_days_ok_but_trades_few():
    r = compare_to_baseline({"sharpe": 0.8, "cagr": 0.25, "max_drawdown": 0.2},
                            BASELINE, n_days=MIN_DAYS_FOR_VERDICT + 10, n_trades=5)
    assert r["verdict"] == "too_early"


def test_verdict_given_once_sample_is_sufficient():
    r = compare_to_baseline({"sharpe": 0.80, "cagr": 0.27, "max_drawdown": 0.28},
                            BASELINE, n_days=MIN_DAYS_FOR_VERDICT,
                            n_trades=MIN_TRADES_FOR_VERDICT)
    assert r["sufficient"] is True
    assert r["verdict"] == "on_track"


# ── Verdicts ──────────────────────────────────────────────────────

def test_severe_shortfall_is_diverged():
    r = compare_to_baseline({"sharpe": -1.5, "cagr": -0.3, "max_drawdown": 0.5},
                            BASELINE, n_days=500, n_trades=200)
    assert r["verdict"] == "diverged"
    assert r["sharpe_z"] < -2


def test_worse_drawdown_is_flagged():
    r = compare_to_baseline({"sharpe": 0.8, "cagr": 0.26, "max_drawdown": 0.60},
                            BASELINE, n_days=500, n_trades=200)
    assert any("Drawdown" in n for n in r["notes"])


# ── Standard error ────────────────────────────────────────────────

def test_standard_error_shrinks_with_more_data():
    assert sharpe_standard_error(1.0, 60) > sharpe_standard_error(1.0, 1000)


def test_standard_error_infinite_with_no_data():
    assert sharpe_standard_error(1.0, 0) == float("inf")


# ── Live metrics from snapshots ───────────────────────────────────

def test_live_metrics_from_rising_equity():
    m = live_metrics_from_snapshots(snapshots(daily_return=0.0005))
    assert m["total_return"] > 0
    assert m["max_drawdown"] == 0.0     # monotonically rising


def test_live_metrics_handles_empty():
    assert live_metrics_from_snapshots([])["n_days"] == 0


def test_elapsed_trading_days_reasonable():
    days = elapsed_trading_days(snapshots(n=365))
    assert 230 < days < 270            # ~252 trading days in a calendar year


def test_format_drift_renders():
    r = compare_to_baseline({"sharpe": 0.5, "cagr": 0.1, "max_drawdown": 0.2},
                            BASELINE, n_days=20, n_trades=2)
    text = format_drift(r)
    assert "TOO EARLY" in text and "Sharpe" in text
