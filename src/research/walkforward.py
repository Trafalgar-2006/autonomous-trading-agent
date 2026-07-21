"""
Walk-forward evaluation harness.

This is the honest test of whether the strategies have any edge:

  * The timeline is split into rolling folds. For each fold the regime model is
    trained ONLY on the prior (in-sample) window, then the strategies are run on
    the next, unseen (out-of-sample) window.
  * The out-of-sample daily returns from every fold are concatenated into one
    continuous return stream — nothing is ever evaluated on data it was tuned on.
  * The result is compared head-to-head with simply buying and holding SPY over
    the same dates. If the system can't beat SPY buy-and-hold after costs, it has
    no demonstrated edge.

Nothing here trades real money — it only measures.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from ..data.features import FeatureEngine
from ..strategy.ensemble import SignalEnsemble
from .backtest import run_fast_backtest
from . import metrics as M

logger = logging.getLogger(__name__)
console = Console(force_terminal=True)


class WalkForward:
    """Rolling train/test evaluation with an SPY buy-and-hold benchmark."""

    def __init__(
        self,
        feed,
        symbols: list[str],
        initial_capital: float = 100_000.0,
        train_days: int = 504,       # ~2y in-sample training per fold
        test_days: int = 126,        # ~6m out-of-sample per fold
        warmup_buffer: int = 300,    # bars of history for indicators before a fold
        benchmark: str = "SPY",
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.0,
        max_risk_per_trade: float = 0.01,
        max_position_size: float = 0.10,
        max_positions: int = 8,
    ):
        self.feed = feed
        self.symbols = [s for s in symbols if s != benchmark]
        self.initial_capital = initial_capital
        self.train_days = train_days
        self.test_days = test_days
        self.warmup_buffer = warmup_buffer
        self.benchmark = benchmark
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.features = FeatureEngine()

    def run(self, total_days: int = 1260) -> dict:
        """Fetch data, run all folds, and return an aggregated result dict."""
        fetch = list(set(self.symbols) | {self.benchmark})
        data = self.feed.get_bars_multi(fetch, days=total_days)
        if not data:
            console.print("[red]No data available for walk-forward.[/red]")
            return {}

        bench_df = data.get(self.benchmark)
        if bench_df is None or bench_df.empty:
            console.print(f"[yellow]Benchmark {self.benchmark} unavailable — "
                          f"using union calendar, no benchmark comparison.[/yellow]")
            calendar = sorted(set().union(*[set(df.index) for df in data.values()]))
        else:
            calendar = list(bench_df.index)

        n = len(calendar)
        first_test = self.train_days
        if n < first_test + self.test_days:
            console.print(f"[red]Not enough history ({n} bars) for "
                          f"{self.train_days}+{self.test_days}.[/red]")
            return {}

        trade_data = {s: df for s, df in data.items() if s in self.symbols}

        combined_returns: list[float] = []
        oos_dates: list = []
        all_trades: list[dict] = []
        fold_rows: list[dict] = []

        k = first_test
        fold_no = 0
        while k + self.test_days <= n:
            fold_no += 1
            train_start = calendar[k - self.train_days]
            test_start = calendar[k]
            test_end = calendar[min(k + self.test_days - 1, n - 1)]
            buffer_start = calendar[max(0, k - self.warmup_buffer)]

            # Fresh ensemble per fold; train regime model on IN-SAMPLE only.
            ensemble = SignalEnsemble()
            train_slice = {
                s: df[(df.index >= train_start) & (df.index < test_start)]
                for s, df in trade_data.items()
            }
            enriched_train = {s: self.features.compute_all(df)
                              for s, df in train_slice.items() if len(df) > 60}
            if enriched_train:
                ensemble.train_classifier(enriched_train, save=False)

            # OUT-OF-SAMPLE test window (with warmup buffer for indicators).
            test_slice = {
                s: df[(df.index >= buffer_start) & (df.index <= test_end)]
                for s, df in trade_data.items()
            }
            test_slice = {s: df for s, df in test_slice.items() if len(df) >= 60}
            if not test_slice:
                k += self.test_days
                continue

            result = run_fast_backtest(
                test_slice,
                ensemble=ensemble,
                initial_capital=self.initial_capital,
                trade_start=test_start,
                slippage_bps=self.slippage_bps,
                commission_per_share=self.commission_per_share,
                max_risk_per_trade=self.max_risk_per_trade,
                max_position_size=self.max_position_size,
                max_positions=self.max_positions,
            )

            # OOS equity within [test_start, test_end] -> daily returns.
            fold_eq = [self.initial_capital]
            fold_dates = []
            for d, eq in zip(result.equity_dates, result.equity_curve):
                if d is not None and test_start <= d <= test_end:
                    fold_eq.append(eq)
                    fold_dates.append(d)
            fold_rets = M.returns_from_equity(fold_eq)
            combined_returns.extend(fold_rets.tolist())
            oos_dates.extend(fold_dates)
            all_trades.extend(result.trades)

            fold_m = M.metrics_from_returns(fold_rets)
            fold_rows.append({
                "fold": fold_no,
                "start": str(test_start.date()),
                "end": str(test_end.date()),
                "return": fold_m["total_return"],
                "sharpe": fold_m["sharpe"],
                "trades": result.total_trades,
            })

            k += self.test_days

        combined = np.asarray(combined_returns, dtype=float)
        strat_metrics = M.metrics_from_returns(combined)

        # Benchmark: SPY buy-and-hold over the same OOS dates.
        bench_metrics = {}
        if bench_df is not None and oos_dates:
            bser = bench_df["close"].reindex(sorted(set(oos_dates))).dropna()
            bench_rets = bser.pct_change().dropna().to_numpy()
            bench_metrics = M.metrics_from_returns(bench_rets)

        attribution = M.strategy_attribution(all_trades)
        years = len(combined) / M.TRADING_DAYS if len(combined) else 0.0
        to = M.turnover(all_trades, self.initial_capital, years)

        result = {
            "folds": fold_rows,
            "strategy": strat_metrics,
            "benchmark": bench_metrics,
            "attribution": attribution,
            "turnover": to,
            "n_trades": len(all_trades),
        }
        self._report(result)
        return result

    def _report(self, result: dict) -> None:
        s = result["strategy"]
        b = result.get("benchmark", {})

        # Per-fold table
        ftab = Table(title="Walk-Forward Folds (out-of-sample)", box=box.SIMPLE_HEAVY)
        for col in ["Fold", "Test Start", "Test End", "Return", "Sharpe", "Trades"]:
            ftab.add_column(col)
        for f in result["folds"]:
            ret_style = "green" if f["return"] >= 0 else "red"
            ftab.add_row(str(f["fold"]), f["start"], f["end"],
                         f"[{ret_style}]{f['return']:.2%}[/{ret_style}]",
                         f"{f['sharpe']:.2f}", str(f["trades"]))
        console.print(ftab)

        # Strategy vs benchmark
        cmp = Table(title="Strategy vs SPY Buy-and-Hold (aggregate OOS)", box=box.ROUNDED)
        cmp.add_column("Metric", style="bold")
        cmp.add_column("Strategy", justify="right")
        cmp.add_column("SPY B&H", justify="right")

        def row(label, key, pct=False, higher_better=True):
            sv = s.get(key, 0.0)
            bv = b.get(key, 0.0)
            fmt = (lambda v: f"{v:.2%}") if pct else (lambda v: f"{v:.2f}")
            sstyle = ""
            if b:
                win = sv >= bv if higher_better else sv <= bv
                sstyle = "green" if win else "red"
            cmp.add_row(label,
                        f"[{sstyle}]{fmt(sv)}[/{sstyle}]" if sstyle else fmt(sv),
                        fmt(bv) if b else "—")

        row("Total Return", "total_return", pct=True)
        row("CAGR", "cagr", pct=True)
        row("Sharpe", "sharpe")
        row("Sortino", "sortino")
        row("Max Drawdown", "max_drawdown", pct=True, higher_better=False)
        row("Calmar", "calmar")
        row("Volatility (ann.)", "volatility", pct=True, higher_better=False)
        row("Win days %", "win_days_pct", pct=True)
        console.print(cmp)

        console.print(f"\nTotal OOS trades: {result['n_trades']} | "
                      f"Annualized turnover: {result['turnover']:.1f}x")

        # Attribution
        if result["attribution"]:
            atab = Table(title="Per-Strategy Attribution (OOS)", box=box.SIMPLE)
            for col in ["Strategy", "Trades", "Win Rate", "Total P&L"]:
                atab.add_column(col)
            for name, d in sorted(result["attribution"].items(),
                                  key=lambda kv: kv[1]["pnl"], reverse=True):
                pnl_style = "green" if d["pnl"] >= 0 else "red"
                atab.add_row(name, str(d["trades"]), f"{d['win_rate']:.0%}",
                             f"[{pnl_style}]${d['pnl']:,.2f}[/{pnl_style}]")
            console.print(atab)

        # Verdict
        if b:
            beat = s.get("sharpe", 0) > b.get("sharpe", 0) and s.get("cagr", 0) > b.get("cagr", 0)
            if beat:
                console.print("\n[bold green]Strategy beat SPY buy-and-hold on both "
                              "Sharpe and CAGR out-of-sample.[/bold green]")
            else:
                console.print("\n[bold yellow]Strategy did NOT clearly beat SPY buy-and-hold "
                              "out-of-sample — no demonstrated edge yet.[/bold yellow]")
