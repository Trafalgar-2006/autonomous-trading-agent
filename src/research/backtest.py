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

import pandas as pd

from ..backtest.engine import BacktestEngine, BacktestResult
from ..core.models import SignalAction
from ..data.features import FeatureEngine
from ..strategy.ensemble import SignalEnsemble
from .experiment import ExperimentConfig

logger = logging.getLogger(__name__)


def _xs_momentum_signals(enriched, pos_index, date, exp, longs_allowed):
    """
    Cross-sectional momentum: rank every symbol by trailing return over
    `xs_lookback` bars and long the top `xs_top`. Emits BUY for the top names
    (if longs allowed) and SELL for everything else (the engine only acts on a
    SELL when it actually holds that symbol), giving a monthly-ish rotation.
    """
    from ..core.models import Signal, SignalAction

    scores = {}
    rows = {}
    for sym, df in enriched.items():
        pos = pos_index[sym].get(date)
        if pos is None or pos < exp.xs_lookback:
            continue
        row = df.iloc[pos]
        past = df["close"].iloc[pos - exp.xs_lookback]
        price = float(row["close"])
        if past and past > 0 and price > 0:
            scores[sym] = price / float(past) - 1.0
            rows[sym] = row

    if not scores:
        return []

    ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
    top = set(ranked[: exp.xs_top])

    out = []
    for sym in ranked:
        row = rows[sym]
        price = float(row["close"])
        atr = float(row["atr"]) if "atr" in row and pd.notna(row["atr"]) else price * 0.02
        if sym in top:
            if not longs_allowed:
                continue
            reasoning = {"atr": atr, "xs_return": round(scores[sym], 4),
                         "entry_reason": "cross_sectional_momentum"}
            if exp.vol_target and "volatility_20d" in row and pd.notna(row["volatility_20d"]):
                vol = float(row["volatility_20d"])
                if vol > 0:
                    lo, hi = exp.vol_mult_bounds
                    reasoning["size_mult"] = float(min(hi, max(lo, exp.vol_target / vol)))
            out.append(Signal(
                symbol=sym, action=SignalAction.BUY, strategy="xs_momentum",
                confidence=0.8, entry_price=price,
                stop_loss=price - 2.0 * atr, take_profit=price + 3.0 * atr,
                reasoning=reasoning,
            ))
        else:
            out.append(Signal(
                symbol=sym, action=SignalAction.SELL, strategy="xs_momentum",
                confidence=1.0, entry_price=price,
                reasoning={"exit_reason": "fell_out_of_top"},
            ))
    # Exits first, then top-ranked BUYs.
    out.sort(key=lambda s: (s.action == SignalAction.BUY, s.confidence))
    return out


def build_fast_signal_fn(
    enriched: dict[str, pd.DataFrame],
    ensemble: SignalEnsemble,
    window: int = 320,
    trade_start=None,
    experiment: ExperimentConfig | None = None,
    market_ok: dict | None = None,
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
    exp = experiment or ExperimentConfig()

    weekly_map: dict[str, dict] = {}
    regime_map: dict[str, dict] = {}
    pos_index: dict[str, dict] = {}

    for sym, df in enriched.items():
        pos_index[sym] = {d: i for i, d in enumerate(df.index)}
        # Cross-sectional momentum ignores per-symbol strategy signals, so skip
        # the (expensive) weekly-trend and regime precompute for it.
        if not exp.xs_momentum:
            weekly_map[sym] = ensemble.compute_weekly_trend_series(df)
            regime_map[sym] = ensemble.classifier.classify_series(df)

    def signal_fn(date):
        if trade_start is not None and date < trade_start:
            return []

        # Market regime filter: block NEW longs when SPY is below its SMA.
        longs_allowed = True
        if exp.market_filter and market_ok is not None:
            longs_allowed = bool(market_ok.get(date, True))

        # Cross-sectional momentum: rank the universe by trailing return and long
        # the top names (ignores per-symbol strategy signals entirely).
        if exp.xs_momentum:
            return _xs_momentum_signals(
                enriched, pos_index, date, exp, longs_allowed)

        buys, sells = [], []
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
                sigs = ensemble.generate_signals(sym, win, weekly=weekly, regime=regime)
            except Exception as e:
                logger.debug(f"fast signal_fn failed for {sym} at {date}: {e}")
                continue

            for s in sigs:
                if s.action == SignalAction.BUY:
                    if not longs_allowed:
                        continue
                    # Volatility-target sizing: scale size toward a common vol.
                    if exp.vol_target and "volatility_20d" in df.columns:
                        vol = df.at[date, "volatility_20d"]
                        if pd.notna(vol) and vol > 0:
                            lo, hi = exp.vol_mult_bounds
                            s.reasoning["size_mult"] = float(
                                min(hi, max(lo, exp.vol_target / float(vol))))
                    buys.append(s)
                else:
                    sells.append(s)

        # Cross-sectional selection: keep only the top-N BUYs by conviction.
        buys.sort(key=lambda s: s.confidence, reverse=True)
        if exp.cross_sectional_top is not None:
            buys = buys[: exp.cross_sectional_top]

        # Exits first so held positions can always be closed.
        return sells + buys

    return signal_fn


def run_fast_backtest(
    data: dict[str, pd.DataFrame],
    ensemble: SignalEnsemble | None = None,
    initial_capital: float = 100_000.0,
    trade_start=None,
    slippage_bps: float = 5.0,
    commission_per_share: float = 0.0,
    max_risk_per_trade: float = 0.01,
    max_position_size: float = 0.10,
    max_positions: int = 8,
    experiment: ExperimentConfig | None = None,
    market_ok: dict | None = None,
) -> BacktestResult:
    """Run the fast research backtest and return a BacktestResult."""
    exp = experiment or ExperimentConfig()
    features = FeatureEngine()
    enriched = {s: features.compute_all(df) for s, df in data.items()}

    ens = ensemble or SignalEnsemble()
    if exp.strategies is not None:
        ens.set_active_strategies(exp.strategies)
    elif exp.disabled_strategies:
        ens.strategies = [s for s in ens.strategies if s.name not in exp.disabled_strategies]

    signal_fn = build_fast_signal_fn(
        enriched, ens, trade_start=trade_start, experiment=exp, market_ok=market_ok)

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
