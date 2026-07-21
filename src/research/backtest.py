"""
Fast research backtest.

Same simulation as backtest/engine.py (next-bar-open fills, slippage,
commission, intrabar stops) but with signals PRECOMPUTED per symbol instead of
re-scanning the whole universe every bar. It injects a fast `signal_fn` into the
existing BacktestEngine, so the execution/accounting logic is shared (no
divergence) — only signal generation is accelerated.

Speedup comes from:
  * computing features once per symbol (not per bar),
  * computing the weekly-trend series and regime series once per symbol,
  * evaluating strategies on a bounded rolling window per bar.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..backtest.engine import BacktestEngine, BacktestResult
from ..data.features import FeatureEngine
from ..strategy.ensemble import SignalEnsemble

logger = logging.getLogger(__name__)


def build_fast_signal_fn(
    enriched: dict[str, pd.DataFrame],
    ensemble: SignalEnsemble,
    window: int = 320,
    trade_start=None,
):
    """
    Build a `signal_fn(date) -> list[Signal]` using precomputed context.

    Args:
        enriched:    symbol -> feature-enriched DataFrame.
        ensemble:    the SignalEnsemble (its classifier should already be trained).
        window:      bars of history handed to each strategy per bar (>= 200 for
                     the 200-day SMA; features are already computed on full history).
        trade_start: if set, no signals are emitted before this date (used to
                     confine a walk-forward fold to its test window).
    """
    weekly_map: dict[str, dict] = {}
    regime_map: dict[str, dict] = {}
    pos_index: dict[str, dict] = {}

    for sym, df in enriched.items():
        weekly_map[sym] = ensemble.compute_weekly_trend_series(df)
        regime_map[sym] = ensemble.classifier.classify_series(df)
        pos_index[sym] = {d: i for i, d in enumerate(df.index)}

    def signal_fn(date):
        if trade_start is not None and date < trade_start:
            return []
        out = []
        for sym, df in enriched.items():
            pos = pos_index[sym].get(date)
            if pos is None:
                continue
            start = max(0, pos - window + 1)
            win = df.iloc[start:pos + 1]
            if len(win) < 30:
                continue
            weekly = weekly_map[sym].get(date)
            regime = regime_map[sym].get(date)
            try:
                out.extend(ensemble.generate_signals(sym, win, weekly=weekly, regime=regime))
            except Exception as e:
                logger.debug(f"fast signal_fn failed for {sym} at {date}: {e}")
        out.sort(key=lambda s: s.confidence, reverse=True)
        return out

    return signal_fn


def run_fast_backtest(
    data: dict[str, pd.DataFrame],
    ensemble: Optional[SignalEnsemble] = None,
    initial_capital: float = 100_000.0,
    trade_start=None,
    slippage_bps: float = 5.0,
    commission_per_share: float = 0.0,
    max_risk_per_trade: float = 0.01,
    max_position_size: float = 0.10,
    max_positions: int = 8,
) -> BacktestResult:
    """Run the fast research backtest and return a BacktestResult."""
    features = FeatureEngine()
    enriched = {s: features.compute_all(df) for s, df in data.items()}

    ens = ensemble or SignalEnsemble()
    signal_fn = build_fast_signal_fn(enriched, ens, trade_start=trade_start)

    engine = BacktestEngine(
        initial_capital=initial_capital,
        max_risk_per_trade=max_risk_per_trade,
        max_position_size=max_position_size,
        max_positions=max_positions,
        slippage_bps=slippage_bps,
        commission_per_share=commission_per_share,
    )
    engine.ensemble = ens  # keep the (possibly trained) ensemble
    return engine.run(data, signal_fn=signal_fn)
