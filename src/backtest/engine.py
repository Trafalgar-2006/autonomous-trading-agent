"""
Backtest Engine — simulates trading strategies on historical data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

from ..core.config import Config
from ..core.models import Signal, SignalAction
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
    Event-driven backtest engine.
    
    Simulates trading through historical data bar-by-bar,
    generating signals and tracking portfolio performance.
    """

    def __init__(
        self,
        initial_capital: float = 20.0,
        max_risk_per_trade: float = 0.02,
        max_position_size: float = 0.15,
        max_positions: int = 5,
    ):
        self.initial_capital = initial_capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        
        self.features = FeatureEngine()
        self.ensemble = SignalEnsemble()

    def run(self, data: dict[str, pd.DataFrame]) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Strategy: Walk forward through bars, generating signals and
        simulating trades with position sizing and risk management.
        """
        equity = self.initial_capital
        cash = self.initial_capital
        positions: dict[str, dict] = {}  # symbol -> {qty, entry_price, strategy}
        trades: list[dict] = []
        equity_curve: list[float] = [equity]
        peak_equity = equity

        # Enrich all data with features
        enriched = {}
        for symbol, df in data.items():
            enriched[symbol] = self.features.compute_all(df)

        # Get common date range
        all_dates = set()
        for df in enriched.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        if len(dates) < 60:
            logger.warning("Not enough data for backtest")
            return BacktestResult(initial_capital=self.initial_capital)

        # Walk forward through dates (skip first 60 for warmup)
        warmup = 60
        for i in range(warmup, len(dates)):
            current_date = dates[i]

            # Create lookback window for each symbol
            window_data = {}
            for symbol, df in enriched.items():
                mask = df.index <= current_date
                window = df[mask].tail(365)
                if len(window) >= 30:
                    window_data[symbol] = window

            if not window_data:
                continue

            # Generate signals
            try:
                signals = self.ensemble.scan_universe(window_data)
            except Exception:
                continue

            # Process signals
            for signal in signals:
                if signal.action == SignalAction.BUY and signal.symbol not in positions:
                    if len(positions) >= self.max_positions:
                        continue

                    price = signal.entry_price or 0
                    if price <= 0:
                        continue

                    # Position sizing
                    max_value = equity * self.max_position_size
                    risk_value = equity * self.max_risk_per_trade
                    
                    atr = signal.reasoning.get("atr", price * 0.02)
                    risk_per_share = atr * 2 if atr > 0 else price * 0.04
                    
                    qty_risk = risk_value / risk_per_share if risk_per_share > 0 else 0
                    qty_max = max_value / price
                    qty_cash = cash * 0.8 / price  # Keep 20% reserve
                    
                    qty = min(qty_risk, qty_max, qty_cash)
                    if qty * price < 1.0:
                        continue

                    # Execute buy
                    cost = qty * price
                    cash -= cost
                    positions[signal.symbol] = {
                        "qty": qty,
                        "entry_price": price,
                        "strategy": signal.strategy,
                        "stop_loss": signal.stop_loss,
                        "take_profit": signal.take_profit,
                    }

                elif signal.action == SignalAction.SELL and signal.symbol in positions:
                    pos = positions[signal.symbol]
                    price = signal.entry_price or 0
                    if price <= 0:
                        continue

                    pnl = (price - pos["entry_price"]) * pos["qty"]
                    cash += pos["qty"] * price
                    
                    trades.append({
                        "symbol": signal.symbol,
                        "action": "SELL",
                        "entry_price": pos["entry_price"],
                        "exit_price": price,
                        "qty": pos["qty"],
                        "pnl": pnl,
                        "strategy": pos["strategy"],
                    })
                    
                    del positions[signal.symbol]

            # Check stop-losses on existing positions
            for symbol in list(positions.keys()):
                if symbol not in window_data:
                    continue
                
                pos = positions[symbol]
                current_price = window_data[symbol]["close"].iloc[-1]
                
                # Stop loss
                if pos.get("stop_loss") and current_price <= pos["stop_loss"]:
                    pnl = (current_price - pos["entry_price"]) * pos["qty"]
                    cash += pos["qty"] * current_price
                    trades.append({
                        "symbol": symbol,
                        "action": "STOP_LOSS",
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "qty": pos["qty"],
                        "pnl": pnl,
                        "strategy": pos["strategy"],
                    })
                    del positions[symbol]
                    continue

                # Take profit
                if pos.get("take_profit") and current_price >= pos["take_profit"]:
                    pnl = (current_price - pos["entry_price"]) * pos["qty"]
                    cash += pos["qty"] * current_price
                    trades.append({
                        "symbol": symbol,
                        "action": "TAKE_PROFIT",
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "qty": pos["qty"],
                        "pnl": pnl,
                        "strategy": pos["strategy"],
                    })
                    del positions[symbol]

            # Update equity
            positions_value = 0
            for symbol, pos in positions.items():
                if symbol in window_data:
                    current_price = window_data[symbol]["close"].iloc[-1]
                    positions_value += pos["qty"] * current_price

            equity = cash + positions_value
            equity_curve.append(equity)
            peak_equity = max(peak_equity, equity)

        # Close any remaining positions at last price
        for symbol, pos in list(positions.items()):
            if symbol in enriched:
                last_price = enriched[symbol]["close"].iloc[-1]
                pnl = (last_price - pos["entry_price"]) * pos["qty"]
                cash += pos["qty"] * last_price
                trades.append({
                    "symbol": symbol,
                    "action": "EOD_CLOSE",
                    "entry_price": pos["entry_price"],
                    "exit_price": last_price,
                    "qty": pos["qty"],
                    "pnl": pnl,
                    "strategy": pos["strategy"],
                })

        # Calculate final metrics
        equity = cash
        total_return = equity - self.initial_capital
        
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        
        # Drawdown
        equity_arr = np.array(equity_curve)
        peaks = np.maximum.accumulate(equity_arr)
        drawdowns = (peaks - equity_arr) / peaks
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = np.diff(equity_arr) / equity_arr[:-1]
            sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        else:
            sharpe = 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_equity=equity,
            total_return=total_return,
            total_return_pct=total_return / self.initial_capital if self.initial_capital > 0 else 0,
            total_trades=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=len(wins) / len(trades) if trades else 0,
            avg_win=gross_profit / len(wins) if wins else 0,
            avg_loss=gross_loss / len(losses) if losses else 0,
            max_drawdown=max_dd * self.initial_capital,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            trades=trades,
            equity_curve=equity_curve,
        )
