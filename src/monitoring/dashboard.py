"""
CLI Dashboard — rich terminal display for trading status.
"""

from __future__ import annotations

import logging

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..core.models import Position, Signal

logger = logging.getLogger(__name__)
console = Console(force_terminal=True)


class CLIDashboard:
    """Rich CLI dashboard for monitoring the trading agent."""

    def show_signals(self, signals: list[Signal]):
        """Display signals in a rich table."""
        if not signals:
            return

        table = Table(
            title="Trading Signals",
            box=box.ROUNDED,
            show_lines=True,
        )
        table.add_column("Symbol", style="bold cyan", width=8)
        table.add_column("Action", width=6)
        table.add_column("Strategy", style="dim", width=16)
        table.add_column("Confidence", justify="right", width=12)
        table.add_column("Price", justify="right", width=10)
        table.add_column("Regime", width=16)
        table.add_column("Weekly", width=8)

        for sig in signals:
            action_style = "green bold" if sig.action.value == "buy" else "red bold"
            conf_pct = f"{sig.confidence:.0%}"

            if sig.confidence >= 0.7:
                conf_style = "green"
            elif sig.confidence >= 0.5:
                conf_style = "yellow"
            else:
                conf_style = "dim"

            regime = sig.regime.value if sig.regime else "—"
            weekly = sig.reasoning.get("weekly_trend", "—") if sig.reasoning else "—"
            price = f"${sig.entry_price:.2f}" if sig.entry_price else "—"

            table.add_row(
                sig.symbol,
                Text(sig.action.value.upper(), style=action_style),
                sig.strategy,
                Text(conf_pct, style=conf_style),
                price,
                regime,
                weekly,
            )

        console.print(table)

    def show_decisions(self, memos: list):
        """Display decision memos (APPROVED / WATCHLIST / REJECTED) in a table."""
        if not memos:
            return

        table = Table(title="Decision Memos", box=box.ROUNDED, show_lines=True)
        table.add_column("Symbol", style="bold cyan", width=8)
        table.add_column("Action", width=6)
        table.add_column("Status", width=10)
        table.add_column("Strat", style="dim", width=14)
        table.add_column("Str", justify="right", width=5)
        table.add_column("Entry", justify="right", width=9)
        table.add_column("Stop", justify="right", width=9)
        table.add_column("Target", justify="right", width=9)
        table.add_column("R:R", justify="right", width=5)
        table.add_column("Why", width=28)

        style_for = {"approved": "green bold", "watchlist": "yellow", "rejected": "red"}
        for m in memos:
            st = m.status.value
            rr = f"{m.risk_reward:.2f}" if m.risk_reward else "—"
            table.add_row(
                m.symbol,
                Text(m.action.value.upper(), style="green" if m.action.value == "buy" else "red"),
                Text(st.upper(), style=style_for.get(st, "white")),
                m.strategy,
                f"{m.signal_strength:.0%}",
                f"${m.entry:.2f}" if m.entry else "—",
                f"${m.stop:.2f}" if m.stop else "—",
                f"${m.target:.2f}" if m.target else "—",
                rr,
                "; ".join(m.reasons)[:28] if m.reasons else "—",
            )
        console.print(table)

    def show_status(
        self,
        account: dict,
        positions: list[Position],
        risk_status: dict,
    ):
        """Display account status, positions, and risk info."""

        # ── Account Panel ─────────────────────────────────────
        equity = account.get("equity", 0)
        cash = account.get("cash", 0)
        buying_power = account.get("buying_power", 0)
        status = account.get("status", "UNKNOWN")

        account_text = (
            f"Status: {status}\n"
            f"Equity: ${equity:,.2f}\n"
            f"Cash: ${cash:,.2f}\n"
            f"Buying Power: ${buying_power:,.2f}\n"
            f"PDT: {'Yes' if account.get('pattern_day_trader') else 'No'}"
        )
        console.print(Panel(account_text, title="Account", border_style="green"))

        # ── Positions Table ───────────────────────────────────
        if positions:
            pos_table = Table(
                title=f"Open Positions ({len(positions)})",
                box=box.SIMPLE_HEAVY,
            )
            pos_table.add_column("Symbol", style="bold")
            pos_table.add_column("Qty", justify="right")
            pos_table.add_column("Entry", justify="right")
            pos_table.add_column("Current", justify="right")
            pos_table.add_column("P&L", justify="right")
            pos_table.add_column("P&L %", justify="right")
            pos_table.add_column("Value", justify="right")

            total_pnl = 0
            for p in positions:
                pnl = p.unrealized_pnl
                total_pnl += pnl
                pnl_pct = p.unrealized_pnl_pct * 100

                pnl_style = "green" if pnl >= 0 else "red"

                pos_table.add_row(
                    p.symbol,
                    f"{p.quantity:.4f}",
                    f"${p.entry_price:.2f}",
                    f"${p.current_price:.2f}",
                    Text(f"${pnl:.2f}", style=pnl_style),
                    Text(f"{pnl_pct:+.2f}%", style=pnl_style),
                    f"${p.market_value:.2f}",
                )

            console.print(pos_table)

            total_style = "green" if total_pnl >= 0 else "red"
            console.print(f"Total Unrealized P&L: [{total_style}]${total_pnl:.2f}[/{total_style}]")
        else:
            console.print("[dim]No open positions[/dim]")

        # ── Risk Status ───────────────────────────────────────
        daily_pnl = risk_status.get("daily_pnl", 0)
        weekly_pnl = risk_status.get("weekly_pnl", 0)
        consecutive = risk_status.get("consecutive_losses", 0)
        cooldown = risk_status.get("cooldown_active", False)

        risk_text = (
            f"Daily P&L: ${daily_pnl:.2f}\n"
            f"Weekly P&L: ${weekly_pnl:.2f}\n"
            f"Consecutive Losses: {consecutive}\n"
            f"Circuit Breaker: {'ACTIVE' if cooldown else 'OK'}"
        )

        border = "red" if cooldown else "blue"
        console.print(Panel(risk_text, title="Risk Status", border_style=border))

    def show_trade_result(self, symbol: str, pnl: float, pnl_pct: float, reason: str):
        """Display a trade result."""
        if pnl >= 0:
            style = "green"
            label = "WIN"
        else:
            style = "red"
            label = "LOSS"

        console.print(
            f"[{style} bold]{label}[/{style} bold] "
            f"{symbol}: P&L=${pnl:.2f} ({pnl_pct:.2%}) [{reason}]"
        )
