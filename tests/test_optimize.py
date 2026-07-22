"""Tests for the parameter sweep + overfitting guard, and strategy diagnosis prompt."""

import numpy as np
import pandas as pd

from src.monitoring.analyst import build_strategy_diagnosis_prompt
from src.research.experiment import ExperimentConfig
from src.research.optimize import format_sweep, sweep


def _trend(n=320, seed=0, slope=0.2):
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    rng = np.random.RandomState(seed)
    close = np.maximum(50 + slope * np.arange(n) + rng.randn(n) * 0.5, 1.0)
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5,
         "close": close, "volume": 1e6},
        index=idx,
    )


class _FakeFeed:
    def get_bars_multi(self, symbols, days=365, timeframe="1Day"):
        return {s: _trend(seed=i + 1, slope=0.1 + 0.03 * i) for i, s in enumerate(symbols)}


def test_sweep_runs_and_ranks():
    base = ExperimentConfig(strategies=("mean_reversion", "breakout"),
                            xs_momentum=True, xs_top=4)
    out = sweep(
        _FakeFeed(), ["AAA", "BBB", "CCC", "DDD", "SPY"], base,
        {"xs_lookback": [60, 90, 120]},
        total_days=420, train_days=120, test_days=60,
    )
    assert len(out["results"]) == 3
    # Sorted by Sharpe descending.
    sharpes = [r["sharpe"] for r in out["results"]]
    assert sharpes == sorted(sharpes, reverse=True)
    assert out["recommended"] is not None
    assert out["best"] is not None


def test_sweep_reports_median_and_recommended():
    base = ExperimentConfig(strategies=("mean_reversion", "breakout"),
                            xs_momentum=True, xs_top=4)
    out = sweep(
        _FakeFeed(), ["AAA", "BBB", "CCC", "SPY"], base,
        {"xs_top": [2, 3, 4]},
        total_days=420, train_days=120, test_days=60,
    )
    text = format_sweep(out)
    assert "Parameter sweep" in text
    assert "median Sharpe" in text
    # Either a warning or the "fairly flat" note must be present.
    assert ("overfit" in text.lower()) or ("flat" in text.lower())


def test_diagnosis_prompt_shows_decay():
    perf = {"momentum": {"trades": 100, "win_rate": 0.55, "pnl": 5000, "profit_factor": 1.5},
            "breakout": {"trades": 40, "win_rate": 0.45, "pnl": -800, "profit_factor": 0.8}}
    recent = {"momentum": {"trades": 10, "win_rate": 0.30, "pnl": -400}}  # decaying
    prompt = build_strategy_diagnosis_prompt(perf, recent)
    assert "ALL-TIME" in prompt and "RECENT" in prompt
    assert "momentum" in prompt and "breakout" in prompt
