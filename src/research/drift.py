"""
Forward-test drift detection — is live behaving like the backtest said it would?

A forward test only means something if you decide *in advance* what would count
as agreement or disagreement. This module compares realized live results to the
walk-forward baseline and, critically, refuses to render a verdict until the
sample is big enough to distinguish signal from noise.

The most common way to fool yourself with a paper account is to read three good
weeks as vindication. The `sufficient` flag exists to stop that.
"""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np

from .metrics import TRADING_DAYS, metrics_from_returns

# A Sharpe estimate's standard error is roughly sqrt((1 + S^2/2) / n) per year of
# data. Below ~1 year / ~30 trades the error bars are wider than any difference
# you'd care about, so we report "too early" rather than a verdict.
MIN_DAYS_FOR_VERDICT = 180
MIN_TRADES_FOR_VERDICT = 30


def sharpe_standard_error(sharpe: float, n_days: int,
                          periods_per_year: int = TRADING_DAYS) -> float:
    """Approximate standard error of an annualized Sharpe estimate."""
    if n_days <= 1:
        return float("inf")
    years = n_days / periods_per_year
    if years <= 0:
        return float("inf")
    return math.sqrt((1 + 0.5 * sharpe ** 2) / years)


def live_metrics_from_snapshots(snapshots: list[dict]) -> dict:
    """Realized performance from the agent's own equity snapshots."""
    equities = [float(s["equity"]) for s in snapshots
                if s.get("equity") not in (None, 0)]
    if len(equities) < 2:
        return metrics_from_returns(np.array([]))

    arr = np.asarray(equities, dtype=float)
    base = arr[:-1]
    returns = np.where(base > 0, np.diff(arr) / base, 0.0)

    # Snapshots are per cycle (often several a day), so annualize on the real
    # elapsed calendar time rather than assuming one bar per trading day.
    periods = TRADING_DAYS
    try:
        start = datetime.fromisoformat(snapshots[0]["timestamp"])
        end = datetime.fromisoformat(snapshots[-1]["timestamp"])
        days = max((end - start).total_seconds() / 86400.0, 1e-9)
        if days > 0.5:
            periods = int(max(1, len(returns) / (days / 365.25)))
    except Exception:
        pass

    return metrics_from_returns(returns, periods_per_year=periods)


def elapsed_trading_days(snapshots: list[dict]) -> int:
    if len(snapshots) < 2:
        return 0
    try:
        start = datetime.fromisoformat(snapshots[0]["timestamp"])
        end = datetime.fromisoformat(snapshots[-1]["timestamp"])
        return int((end - start).days * (TRADING_DAYS / 365.25))
    except Exception:
        return 0


def compare_to_baseline(live: dict, baseline: dict, n_days: int,
                        n_trades: int) -> dict:
    """
    Compare live metrics to the backtest baseline.

    Returns a verdict dict. `sufficient` is False until there's enough data to
    say anything honest — in which case `verdict` is "too_early".
    """
    live_sharpe = live.get("sharpe", 0.0)
    live_cagr = live.get("cagr", 0.0)
    live_dd = live.get("max_drawdown", 0.0)

    base_sharpe = baseline.get("sharpe", 0.0)
    base_cagr = baseline.get("cagr", 0.0)
    base_dd = baseline.get("max_drawdown", 0.0)

    se = sharpe_standard_error(live_sharpe, n_days)
    # How many standard errors below the backtest is the live Sharpe?
    z = (live_sharpe - base_sharpe) / se if se and math.isfinite(se) and se > 0 else 0.0

    sufficient = n_days >= MIN_DAYS_FOR_VERDICT and n_trades >= MIN_TRADES_FOR_VERDICT

    notes: list[str] = []
    if not sufficient:
        verdict = "too_early"
        notes.append(
            f"Need ~{MIN_DAYS_FOR_VERDICT} trading days and {MIN_TRADES_FOR_VERDICT} "
            f"trades before the numbers mean anything; have {n_days} and {n_trades}."
        )
    elif z < -2:
        verdict = "diverged"
        notes.append("Live Sharpe is more than 2 standard errors below the backtest — "
                     "the backtest's edge is not showing up.")
    elif z < -1:
        verdict = "underperforming"
        notes.append("Live is running below the backtest but still within noise. "
                     "Keep collecting data.")
    else:
        verdict = "on_track"
        notes.append("Live is consistent with the backtest within statistical error.")

    if base_dd and live_dd > base_dd * 1.25:
        notes.append(f"Drawdown ({live_dd:.1%}) is materially worse than backtest "
                     f"({base_dd:.1%}) — risk is behaving worse than modelled.")

    return {
        "verdict": verdict,
        "sufficient": sufficient,
        "n_days": n_days,
        "n_trades": n_trades,
        "sharpe_standard_error": se if math.isfinite(se) else None,
        "sharpe_z": z,
        "live": {"cagr": live_cagr, "sharpe": live_sharpe, "max_drawdown": live_dd},
        "baseline": {"cagr": base_cagr, "sharpe": base_sharpe, "max_drawdown": base_dd},
        "notes": notes,
    }


def format_drift(report: dict) -> str:
    """Render a drift report as readable text."""
    live, base = report["live"], report["baseline"]
    label = {
        "too_early": "TOO EARLY TO TELL",
        "on_track": "ON TRACK",
        "underperforming": "UNDERPERFORMING (within noise)",
        "diverged": "DIVERGED FROM BACKTEST",
    }.get(report["verdict"], report["verdict"])

    lines = [
        f"Forward test: {label}",
        f"  Sample: {report['n_days']} trading days, {report['n_trades']} closed trades",
        f"  {'metric':<10}{'live':>12}{'backtest':>12}",
        f"  {'CAGR':<10}{live['cagr']:>11.2%}{base['cagr']:>12.2%}",
        f"  {'Sharpe':<10}{live['sharpe']:>11.2f}{base['sharpe']:>12.2f}",
        f"  {'MaxDD':<10}{live['max_drawdown']:>11.2%}{base['max_drawdown']:>12.2%}",
    ]
    if report.get("sharpe_standard_error"):
        lines.append(f"  Sharpe std-error ±{report['sharpe_standard_error']:.2f} "
                     f"(z = {report['sharpe_z']:+.2f})")
    lines.extend(f"  - {n}" for n in report["notes"])
    return "\n".join(lines)
