"""
Performance metrics for research / walk-forward evaluation.

Everything is computed from a daily-return series so strategy and benchmark
(SPY buy-and-hold) are compared on identical footing.
"""

from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def returns_from_equity(equity: list[float]) -> np.ndarray:
    """Daily simple returns from an equity curve."""
    arr = np.asarray(equity, dtype=float)
    if len(arr) < 2:
        return np.array([])
    base = arr[:-1]
    return np.where(base > 0, np.diff(arr) / base, 0.0)


def metrics_from_returns(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> dict:
    """Compute a standard metrics bundle from a daily-return series."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {
            "total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "sortino": 0.0,
            "max_drawdown": 0.0, "calmar": 0.0, "volatility": 0.0,
            "win_days_pct": 0.0, "n_days": 0,
        }

    equity = np.cumprod(1 + r)
    total_return = float(equity[-1] - 1)
    years = len(r) / periods_per_year
    cagr = float(equity[-1] ** (1 / years) - 1) if years > 0 and equity[-1] > 0 else 0.0

    vol = float(r.std())
    sharpe = float(r.mean() / vol * np.sqrt(periods_per_year)) if vol > 0 else 0.0

    downside = r[r < 0]
    dstd = float(downside.std()) if len(downside) > 0 else 0.0
    sortino = float(r.mean() / dstd * np.sqrt(periods_per_year)) if dstd > 0 else 0.0

    peaks = np.maximum.accumulate(equity)
    dd = np.where(peaks > 0, (peaks - equity) / peaks, 0.0)
    max_dd = float(dd.max()) if len(dd) else 0.0
    calmar = float(cagr / max_dd) if max_dd > 0 else 0.0

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "calmar": calmar,
        "volatility": float(vol * np.sqrt(periods_per_year)),
        "win_days_pct": float((r > 0).mean()),
        "n_days": int(len(r)),
    }


def strategy_attribution(trades: list[dict]) -> dict:
    """Aggregate realized P&L per strategy from a list of trade dicts."""
    out: dict[str, dict] = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        d = out.setdefault(s, {"trades": 0, "wins": 0, "pnl": 0.0})
        d["trades"] += 1
        d["pnl"] += t.get("pnl", 0.0)
        if t.get("pnl", 0.0) > 0:
            d["wins"] += 1
    for s, d in out.items():
        d["win_rate"] = d["wins"] / d["trades"] if d["trades"] else 0.0
    return out


def turnover(trades: list[dict], avg_equity: float, years: float) -> float:
    """Annualized turnover = traded notional / equity / years (rough estimate)."""
    if avg_equity <= 0 or years <= 0:
        return 0.0
    notional = sum(abs(t.get("exit_price", 0.0) * t.get("qty", 0.0)) for t in trades)
    return float(notional / avg_equity / years)
