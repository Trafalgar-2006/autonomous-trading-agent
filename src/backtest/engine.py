"""
Backtest Engine — simulates trading strategies on historical data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from ..core.models import SignalAction
from ..data.features import FeatureEngine
from ..strategy.ensemble import SignalEnsemble

logger = logging.getLogger(__name__)
console = Console(force_terminal=True)


@dataclass
class BacktestResult:
    """Stores results from a backtest run."""
    initial_capital: float = 0.0
    final_equity: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    equity_dates: list = field(default_factory=list)

    def print_summary(self):
        """Print a formatted backtest summary."""
        table = Table(title="Backtest Results", box=box.ROUNDED)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        ret_style = "green" if self.total_return >= 0 else "red"

        table.add_row("Initial Capital", f"${self.initial_capital:,.2f}")
        table.add_row("Final Equity", f"${self.final_equity:,.2f}")
        table.add_row("Total Return", f"[{ret_style}]${self.total_return:,.2f}[/{ret_style}]")
        table.add_row("Return %", f"[{ret_style}]{self.total_return_pct:.2%}[/{ret_style}]")
        table.add_row("", "")
        table.add_row("Total Trades", str(self.total_trades))
        table.add_row("Winning Trades", f"[green]{self.winning_trades}[/green]")
        table.add_row("Losing Trades", f"[red]{self.losing_trades}[/red]")
        table.add_row("Win Rate", f"{self.win_rate:.1%}")
        table.add_row("Avg Win", f"[green]${self.avg_win:.2f}[/green]")
        table.add_row("Avg Loss", f"[red]${self.avg_loss:.2f}[/red]")
        table.add_row("", "")
        table.add_row("Max Drawdown", f"[red]{self.max_drawdown_pct:.2%}[/red]")
        table.add_row("Sharpe Ratio", f"{self.sharpe_ratio:.2f}")
        table.add_row("Profit Factor", f"{self.profit_factor:.2f}")

        console.print(table)

        # Show last 10 trades
        if self.trades:
            trade_table = Table(title="Recent Trades", box=box.SIMPLE)
            trade_table.add_column("Symbol")
            trade_table.add_column("Action")
            trade_table.add_column("Entry")
            trade_table.add_column("Exit")
            trade_table.add_column("P&L")
            trade_table.add_column("Strategy")

            for t in self.trades[-10:]:
                pnl_style = "green" if t["pnl"] >= 0 else "red"
                trade_table.add_row(
                    t["symbol"],
                    t["action"],
                    f"${t['entry_price']:.2f}",
                    f"${t['exit_price']:.2f}",
                    f"[{pnl_style}]${t['pnl']:.2f}[/{pnl_style}]",
                    t["strategy"],
                )

            console.print(trade_table)


class BacktestEngine:
    """
    Event-driven backtest engine with realistic execution.

    Key properties (avoid look-ahead bias):
    - Signals are decided on bar `t` using data up to and including bar `t`'s CLOSE.
    - The resulting orders are filled on bar `t+1`'s OPEN (never the same close
      that produced the signal).
    - Slippage and commission are charged on every fill.
    - Stop-loss / take-profit are checked intrabar against the bar's high/low,
      with gap-through fills at the open when the market opens beyond the level.
    """

    def __init__(
        self,
        initial_capital: float = 20.0,
        max_risk_per_trade: float = 0.02,
        max_position_size: float = 0.15,
        max_positions: int = 5,
        slippage_bps: float = 5.0,          # 5 bps = 0.05% per fill
        commission_per_share: float = 0.0,  # Alpaca US equities are commission-free
        min_position_value: float = 1.0,
        cash_reserve_pct: float = 0.20,
    ):
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.min_position_value = min_position_value
        self.cash_reserve_pct = cash_reserve_pct

        self.features = FeatureEngine()
        self.ensemble = SignalEnsemble()

    # ── Cost helpers ──────────────────────────────────────────────
    def _buy_fill(self, price: float) -> float:
        """Price paid when buying (slippage pushes it up)."""
        return price * (1 + self.slippage_bps / 1e4)

    def _sell_fill(self, price: float) -> float:
        """Price received when selling (slippage pushes it down)."""
        return price * (1 - self.slippage_bps / 1e4)

    def _commission(self, qty: float) -> float:
        return qty * self.commission_per_share

    def _size_position(self, equity: float, cash: float, fill_price: float, atr: float) -> float:
        """Position size: min of risk-based, max-position, and available-cash caps."""
        if fill_price <= 0:
            return 0.0
        risk_value = equity * self.max_risk_per_trade
        risk_per_share = atr * 2 if atr > 0 else fill_price * 0.04
        qty_risk = risk_value / risk_per_share if risk_per_share > 0 else 0.0
        qty_max = (equity * self.max_position_size) / fill_price
        qty_cash = (cash * (1 - self.cash_reserve_pct)) / fill_price
        return max(0.0, min(qty_risk, qty_max, qty_cash))

    def _default_signal_fn(self, enriched: dict):
        """Signal provider used when none is injected: scan the universe each bar."""
        def fn(date):
            window_data = {}
            for symbol, df in enriched.items():
                window = df[df.index <= date].tail(365)
                if len(window) >= 30:
                    window_data[symbol] = window
            if not window_data:
                return []
            try:
                return self.ensemble.scan_universe(window_data)
            except Exception as e:
                logger.debug(f"Signal generation failed at {date}: {e}")
                return []
        return fn

    def run(self, data: dict[str, pd.DataFrame], signal_fn=None) -> BacktestResult:
        """
        Walk forward through history with next-bar-open execution.

        `signal_fn(date) -> list[Signal]` can be injected to supply signals
        (the research backtester passes a fast, precomputed provider). When
        omitted, the default per-bar universe scan is used.
        """
        cash = self.initial_capital
        equity = self.initial_capital
        positions: dict[str, dict] = {}      # symbol -> position dict
        pending: list[dict] = []             # orders queued to fill next bar's open
        trades: list[dict] = []
        equity_curve: list[float] = [equity]
        equity_dates: list = [None]          # aligns with equity_curve
        peak_equity = equity

        # Enrich all data with features
        enriched = {symbol: self.features.compute_all(df) for symbol, df in data.items()}

        # Common, sorted date axis across all symbols
        all_dates = set()
        for df in enriched.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) < 60:
            logger.warning("Not enough data for backtest")
            return BacktestResult(initial_capital=self.initial_capital)

        def bar(symbol: str, date) -> Optional[pd.Series]:
            df = enriched.get(symbol)
            if df is None or date not in df.index:
                return None
            return df.loc[date]

        if signal_fn is None:
            signal_fn = self._default_signal_fn(enriched)

        warmup = 60
        for i in range(warmup, len(dates)):
            date = dates[i]

            # ── 1. Fill orders queued on the previous bar at THIS bar's open ──
            for order in pending:
                symbol = order["symbol"]
                row = bar(symbol, date)
                if row is None:
                    continue  # symbol didn't trade this bar; drop the order
                open_price = float(row["open"])
                if open_price <= 0:
                    continue

                if order["action"] == "BUY":
                    if symbol in positions or len(positions) >= self.max_positions:
                        continue
                    fill = self._buy_fill(open_price)
                    qty = self._size_position(equity, cash, fill, order.get("atr", 0.0))
                    qty *= order.get("size_mult", 1.0)  # volatility-target scaling
                    if qty * fill < self.min_position_value:
                        continue
                    cost = qty * fill + self._commission(qty)
                    if cost > cash:
                        continue
                    cash -= cost
                    positions[symbol] = {
                        "qty": qty,
                        "entry_price": fill,
                        "strategy": order["strategy"],
                        "stop_loss": order.get("stop_loss"),
                        "take_profit": order.get("take_profit"),
                    }

                elif order["action"] == "SELL" and symbol in positions:
                    pos = positions[symbol]
                    fill = self._sell_fill(open_price)
                    cash += pos["qty"] * fill - self._commission(pos["qty"])
                    pnl = (fill - pos["entry_price"]) * pos["qty"] - self._commission(pos["qty"])
                    trades.append(self._record(symbol, "SELL", pos, fill, pnl, date))
                    del positions[symbol]

            pending = []  # queue consumed

            # ── 2. Intrabar stop-loss / take-profit against this bar's range ──
            for symbol in list(positions.keys()):
                row = bar(symbol, date)
                if row is None:
                    continue
                pos = positions[symbol]
                o, h, l = float(row["open"]), float(row["high"]), float(row["low"])
                exit_price = None
                action = None

                stop = pos.get("stop_loss")
                target = pos.get("take_profit")

                # Stop takes priority (conservative). Gap-through fills at the open.
                if stop and l <= stop:
                    exit_price = o if o < stop else stop
                    action = "STOP_LOSS"
                elif target and h >= target:
                    exit_price = o if o > target else target
                    action = "TAKE_PROFIT"

                if exit_price is not None:
                    fill = self._sell_fill(exit_price)
                    cash += pos["qty"] * fill - self._commission(pos["qty"])
                    pnl = (fill - pos["entry_price"]) * pos["qty"] - self._commission(pos["qty"])
                    trades.append(self._record(symbol, action, pos, fill, pnl, date))
                    del positions[symbol]

            # ── 3. Decide signals on this bar's close; queue for NEXT bar open ──
            signals = signal_fn(date)
            queued = {o["symbol"] for o in pending}
            for signal in signals:
                sym = signal.symbol
                if sym in queued:
                    continue
                if signal.action == SignalAction.BUY and sym not in positions:
                    if len(positions) + len(pending) >= self.max_positions:
                        continue
                    pending.append({
                        "symbol": sym,
                        "action": "BUY",
                        "strategy": signal.strategy,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                        "atr": signal.reasoning.get("atr", 0.0),
                        "size_mult": signal.reasoning.get("size_mult", 1.0),
                    })
                    queued.add(sym)
                elif signal.action == SignalAction.SELL and sym in positions:
                    pending.append({"symbol": sym, "action": "SELL", "strategy": signal.strategy})
                    queued.add(sym)

            # ── 4. Mark-to-market at this bar's close ─────────────────────
            positions_value = 0.0
            for symbol, pos in positions.items():
                row = bar(symbol, date)
                if row is not None:
                    positions_value += pos["qty"] * float(row["close"])
            equity = cash + positions_value
            equity_curve.append(equity)
            equity_dates.append(date)
            peak_equity = max(peak_equity, equity)

        # ── Close any remaining positions at the last available close ──────
        last_date = dates[-1]
        for symbol, pos in list(positions.items()):
            row = bar(symbol, last_date)
            if row is None:
                df = enriched.get(symbol)
                if df is None or df.empty:
                    continue
                last_price = float(df["close"].iloc[-1])
            else:
                last_price = float(row["close"])
            fill = self._sell_fill(last_price)
            cash += pos["qty"] * fill - self._commission(pos["qty"])
            pnl = (fill - pos["entry_price"]) * pos["qty"] - self._commission(pos["qty"])
            trades.append(self._record(symbol, "EOD_CLOSE", pos, fill, pnl, last_date))

        return self._build_result(cash, trades, equity_curve, equity_dates)

    @staticmethod
    def _record(symbol: str, action: str, pos: dict, exit_price: float,
                pnl: float, date=None) -> dict:
        return {
            "symbol": symbol,
            "action": action,
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "qty": pos["qty"],
            "pnl": pnl,
            "strategy": pos["strategy"],
            "date": date,
        }

    def _build_result(self, final_cash: float, trades: list[dict],
                      equity_curve: list[float], equity_dates: list = None) -> BacktestResult:
        equity = final_cash
        total_return = equity - self.initial_capital

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]

        equity_arr = np.array(equity_curve, dtype=float)
        peaks = np.maximum.accumulate(equity_arr)
        with np.errstate(divide="ignore", invalid="ignore"):
            drawdowns = np.where(peaks > 0, (peaks - equity_arr) / peaks, 0.0)
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

        if len(equity_arr) > 1:
            base = equity_arr[:-1]
            returns = np.where(base > 0, np.diff(equity_arr) / base, 0.0)
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0
        else:
            sharpe = 0.0

        gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=equity,
            total_return=total_return,
            total_return_pct=total_return / self.initial_capital if self.initial_capital > 0 else 0.0,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0.0,
            avg_win=gross_profit / len(wins) if wins else 0.0,
            avg_loss=gross_loss / len(losses) if losses else 0.0,
            max_drawdown=max_dd * self.initial_capital,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0.0,
            trades=trades,
            equity_curve=equity_curve,
            equity_dates=equity_dates if equity_dates is not None else [],
        )
