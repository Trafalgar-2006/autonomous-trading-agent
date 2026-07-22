"""
Parameter sweep for the walk-forward harness — WITH overfitting guards.

Systematic parameter search is the fastest way to fool yourself: run 50 configs,
pick the best Sharpe, and you've almost certainly selected noise. This module
runs the sweep but refuses to just hand you the winner. It reports the full
distribution and warns when the "best" is likely a lucky outlier rather than a
robust choice — and recommends the median-robust parameter, not the maximum.

Everything is out-of-sample (each config is scored on the same rolling
walk-forward). Even so: treat any sweep result as a hypothesis to forward-test,
never as a tuned answer.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import replace

import numpy as np

from .metrics import TRADING_DAYS
from .walkforward import WalkForward

logger = logging.getLogger(__name__)


def _sharpe_se(sharpe: float, n_days: int) -> float:
    if n_days <= 1:
        return float("inf")
    years = n_days / TRADING_DAYS
    return math.sqrt((1 + 0.5 * sharpe ** 2) / years) if years > 0 else float("inf")


def sweep(feed, symbols, base_experiment, param_grid: dict, total_days: int = 2600,
          **wf_kwargs) -> dict:
    """
    Grid-search `param_grid` (name -> list of values) over walk-forward.

    Returns {results, best, median, overfit_warning, recommended}. `results`
    is one row per config, sorted by Sharpe descending.
    """
    names = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[n] for n in names]))
    logger.info(f"Sweeping {len(combos)} configs over {len(names)} params")

    results = []
    for combo in combos:
        params = dict(zip(names, combo, strict=False))
        exp = replace(base_experiment, **params)
        wf = WalkForward(feed, symbols, experiment=exp, **wf_kwargs)
        r = wf.run(total_days=total_days, verbose=False)
        s = r.get("strategy", {})
        results.append({
            "params": params,
            "cagr": s.get("cagr", 0.0),
            "sharpe": s.get("sharpe", 0.0),
            "max_drawdown": s.get("max_drawdown", 0.0),
            "n_days": s.get("n_days", 0),
            "n_trades": r.get("n_trades", 0),
        })

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    sharpes = np.array([r["sharpe"] for r in results], dtype=float)
    best = results[0] if results else None
    median_sharpe = float(np.median(sharpes)) if len(sharpes) else 0.0

    # Overfitting heuristic: is the best more than ~1 standard error above the
    # median of the grid? If so, it's likely a lucky draw, not a real optimum.
    overfit_warning = None
    if best and len(results) >= 3:
        se = _sharpe_se(best["sharpe"], best["n_days"])
        gap = best["sharpe"] - median_sharpe
        if math.isfinite(se) and se > 0 and gap > se:
            overfit_warning = (
                f"The best config's Sharpe ({best['sharpe']:.2f}) is {gap / se:.1f} "
                f"standard errors above the grid median ({median_sharpe:.2f}) — "
                f"likely an overfit outlier. Prefer a parameter that is good "
                f"across neighbours, and forward-test before trusting it."
            )

    # Recommend the config nearest the median Sharpe (robust, not the peak).
    recommended = None
    if results:
        recommended = min(results, key=lambda r: abs(r["sharpe"] - median_sharpe))

    return {
        "results": results,
        "best": best,
        "median_sharpe": median_sharpe,
        "overfit_warning": overfit_warning,
        "recommended": recommended,
    }


def format_sweep(out: dict, top: int = 10) -> str:
    lines = ["=== Parameter sweep (walk-forward, out-of-sample) ==="]
    lines.append(f"{'params':40}{'CAGR':>8}{'Sharpe':>8}{'MaxDD':>8}{'trades':>8}")
    for r in out["results"][:top]:
        p = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        lines.append(f"{p[:40]:40}{r['cagr']:>7.1%}{r['sharpe']:>8.2f}"
                     f"{r['max_drawdown']:>7.1%}{r['n_trades']:>8}")
    lines.append(f"\nGrid median Sharpe: {out['median_sharpe']:.2f}")
    if out.get("recommended"):
        rp = ", ".join(f"{k}={v}" for k, v in out["recommended"]["params"].items())
        lines.append(f"Robust pick (nearest median): {rp}")
    if out.get("overfit_warning"):
        lines.append(f"\n[!] {out['overfit_warning']}")
    else:
        lines.append("\nNo single config stands out as an overfit outlier — the grid "
                     "is fairly flat (a good sign the parameter isn't fragile).")
    return "\n".join(lines)
