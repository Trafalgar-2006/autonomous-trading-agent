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
    for d in out.values():
        d["win_rate"] = d["wins"] / d["trades"] if d["trades"] else 0.0
    return out


def tearsheet(result, benchmark_returns=None, label: str = "Strategy") -> dict:
    """
    Full performance summary for a BacktestResult.

    Bundles the return/risk metrics, trade statistics, per-strategy attribution
    and (optionally) a benchmark comparison into one dict that can be printed,
    logged, or diffed between runs.
    """
    returns = returns_from_equity(result.equity_curve)
    m = metrics_from_returns(returns)

    trades = result.trades or []
    years = m["n_days"] / TRADING_DAYS if m["n_days"] else 0.0
    wins = [t for t in trades if t.get("pnl", 0) > 0]
    losses = [t for t in trades if t.get("pnl", 0) <= 0]
    gross_win = sum(t["pnl"] for t in wins) if wins else 0.0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

    sheet = {
        "label": label,
        "metrics": m,
        "trades": {
            "total": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else 0.0,
            "avg_win": gross_win / len(wins) if wins else 0.0,
            "avg_loss": gross_loss / len(losses) if losses else 0.0,
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else 0.0,
            "expectancy": (sum(t.get("pnl", 0) for t in trades) / len(trades)) if trades else 0.0,
        },
        "attribution": strategy_attribution(trades),
        "turnover": turnover(trades, result.initial_capital, years),
    }

    if benchmark_returns is not None and len(benchmark_returns) > 0:
        bm = metrics_from_returns(np.asarray(benchmark_returns, dtype=float))
        sheet["benchmark"] = bm
        sheet["excess"] = {
            "cagr": m["cagr"] - bm["cagr"],
            "sharpe": m["sharpe"] - bm["sharpe"],
            "max_drawdown": m["max_drawdown"] - bm["max_drawdown"],
        }
        sheet["beat_benchmark"] = (m["sharpe"] > bm["sharpe"] and m["cagr"] > bm["cagr"])

    return sheet


def format_tearsheet(sheet: dict) -> str:
    """Render a tearsheet as readable text."""
    m, t = sheet["metrics"], sheet["trades"]
    lines = [
        f"=== {sheet['label']} ===",
        f"  CAGR {m['cagr']:>8.2%}   Total {m['total_return']:>8.2%}",
        f"  Sharpe {m['sharpe']:>6.2f}   Sortino {m['sortino']:>6.2f}   Calmar {m['calmar']:>6.2f}",
        f"  MaxDD {m['max_drawdown']:>7.2%}   Vol {m['volatility']:>8.2%}   "
        f"WinDays {m['win_days_pct']:>6.1%}",
        f"  Trades {t['total']:>5}   WinRate {t['win_rate']:>6.1%}   "
        f"PF {t['profit_factor']:>5.2f}   Expectancy ${t['expectancy']:,.2f}",
        f"  Turnover {sheet['turnover']:.1f}x/yr",
    ]
    if sheet.get("attribution"):
        lines.append("  Attribution:")
        for name, d in sorted(sheet["attribution"].items(),
                              key=lambda kv: kv[1]["pnl"], reverse=True):
            lines.append(f"    {name:16} {d['trades']:>4} trades  "
                         f"{d['win_rate']:>5.0%} win  ${d['pnl']:>12,.2f}")
    if "benchmark" in sheet:
        b = sheet["benchmark"]
        verdict = "BEAT" if sheet.get("beat_benchmark") else "did NOT beat"
        lines.append(f"  Benchmark: CAGR {b['cagr']:.2%}, Sharpe {b['sharpe']:.2f} "
                     f"-> {verdict} the benchmark")
    return "\n".join(lines)


def turnover(trades: list[dict], avg_equity: float, years: float) -> float:
    """Annualized turnover = traded notional / equity / years (rough estimate)."""
    if avg_equity <= 0 or years <= 0:
        return 0.0
    notional = sum(abs(t.get("exit_price", 0.0) * t.get("qty", 0.0)) for t in trades)
    return float(notional / avg_equity / years)
