"""
Experiment configuration for research — composable strategy improvements to
A/B test through the walk-forward harness.

Each option can be toggled independently so we can measure its marginal effect:
  * disabled_strategies : drop under-performing strategies (e.g. momentum)
  * market_filter       : only open longs when SPY is above its 200-day SMA
  * vol_target          : volatility-target position sizing (scale by target/vol)
  * cross_sectional_top : each day, keep only the top-N BUY candidates by conviction
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    strategies: tuple | None = None   # explicit set (overrides config `enabled`)
    disabled_strategies: tuple = ()
    market_filter: bool = False
    market_sma: int = 200
    vol_target: float | None = None          # annualized per-position vol target
    vol_mult_bounds: tuple = (0.5, 2.0)
    cross_sectional_top: int | None = None   # keep top-N BUYs per day
    # Cross-sectional momentum: ignore per-symbol strategy signals; instead rank
    # the whole universe by trailing return and long the top names.
    xs_momentum: bool = False
    xs_lookback: int = 120                       # bars of trailing return to rank on
    xs_skip: int = 0                             # skip the most recent N bars (12-1 momentum)
    xs_top: int = 6                              # number of top names to hold
    tags: dict = field(default_factory=dict)


def market_ok_series(spy_df: pd.DataFrame, sma: int = 200) -> dict:
    """
    date -> bool: is SPY above its `sma`-day SMA (i.e. long trades allowed)?

    Early bars without enough history to compute the SMA default to True
    (permissive — don't block when the trend is simply unknown).
    """
    if spy_df is None or spy_df.empty:
        return {}
    close = spy_df["close"]
    ma = close.rolling(sma).mean()
    ok = close > ma
    out = {}
    for date, val in ok.items():
        out[date] = True if pd.isna(val) else bool(val)
    return out
