"""
Trading Agent — Main Entry Point & Orchestrator.

This is the central coordinator that ties all components together:
- Initializes all modules
- Runs scheduled market scans
- Processes signals through risk → execution pipeline
- Manages the lifecycle of the trading agent

Usage:
    # Scan markets and show signals (dry run)
    python -m src.main scan
    
    # Run backtest on historical data
    python -m src.main backtest
    
    # Start live/paper trading agent
    python -m src.main run
    
    # Show account status
    python -m src.main status
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

# Fix Windows console encoding for Unicode (emojis, special chars)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from rich.console import Console
from rich.logging import RichHandler
from rich import box

from .core.config import Config
from .core.event_bus import EventBus
from .core.models import DecisionStatus
from .data.feed import MarketDataFeed
from .data.features import FeatureEngine
from .data.store import DataStore
from .data.scanner import MarketScanner
from .strategy.ensemble import SignalEnsemble
from .execution.order_manager import OrderManager
from .monitoring.alerts import TelegramAlerts
from .monitoring.dashboard import CLIDashboard
from .monitoring.analyst import AIAnalyst

console = Console(force_terminal=True)
logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Configure rich logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            rich_tracebacks=True,
            markup=True,
            console=Console(force_terminal=True, stderr=True),
        )],
    )


class TradingAgent:
    """
    Main trading agent orchestrator.
    
    Coordinates data fetching, strategy analysis, risk management,
    order execution, and monitoring.
    """

    def __init__(self):
        self.config = Config()
        setup_logging(self.config.log_level)
        
        # Initialize components
        self.feed = MarketDataFeed()
        self.features = FeatureEngine()
        self.store = DataStore()
        self.ensemble = SignalEnsemble()
        self.order_manager = OrderManager()
        # Share the OrderManager's RiskManager so the dashboard reflects the
        # SAME state (daily P&L, consecutive losses, circuit breaker) that the
        # trade pipeline actually updates — not a second, always-empty instance.
        self.risk_manager = self.order_manager.risk_manager
        self.alerts = TelegramAlerts()
        self.dashboard = CLIDashboard()
        self.analyst = AIAnalyst()
        self.bus = EventBus()
        self.scanner = MarketScanner(self.feed)
        
        # Strategy mode: 'ensemble' (TA) or 'xs_momentum' (cross-sectional momentum)
        self.xs_strategy = None
        if self.config.strategy_mode == "xs_momentum":
            from .strategy.xs_momentum import CrossSectionalMomentum
            xs = self.config.xs
            self.xs_strategy = CrossSectionalMomentum(
                lookback=xs.get("lookback_days", 252),
                top_n=xs.get("top_n", 8),
                vol_target=xs.get("vol_target"),
            )

        # Dynamic universe (starts with core, expanded by scanner)
        if self.xs_strategy and self.config.xs.get("use_broad_universe", True):
            from .research.universe import BROAD_UNIVERSE
            self._active_symbols: list[str] = list(BROAD_UNIVERSE)
            # Optionally rank crypto alongside equities (24/7 market).
            if self.xs_strategy and self.config.xs.get("include_crypto", False):
                self._active_symbols += list(self.config.xs.get("crypto_symbols", []))
        else:
            self._active_symbols: list[str] = list(self.config.symbols)
        self._last_scan_date: str = ""
        self._last_daily_date: str = ""  # tracks daily EOD summary / perf refresh
        self._market_ok: bool = True     # SPY>SMA regime flag (set each scan)
        self._correlations: dict = {}    # pairwise return correlations (set each scan)
        
        self._running = False
        # Sync live equity from broker so position sizing is accurate
        account = self.order_manager.broker.get_account()
        live_equity = account.get("equity", 0)
        if live_equity > 0:
            self.config.set_live_equity(live_equity)
            logger.info(f"Live equity synced: ${live_equity:,.2f}")

        # Reconcile restored/open trades against what the broker actually holds
        # so every live position has stop-loss protection from the first cycle.
        self.order_manager.reconcile_open_trades()

        # Learn strategy weights from realized trade history (down-weight losers).
        self.ensemble.update_performance_weights(self.store)

        logger.info("Trading Agent initialized")

    async def scan(self):
        """
        Run a single market scan:
        1. Fetch latest data for all symbols
        2. Compute features
        3. Generate signals
        4. Display results
        """
        console.print("\n[bold cyan]Scanning market...[/bold cyan]\n")
        
        symbols = self._active_symbols
        logger.info(f"Scanning {len(symbols)} symbols ({self.config.strategy_mode} mode)")

        # Cross-sectional momentum needs enough history for its lookback.
        fetch_days = self.config.lookback_days
        if self.xs_strategy:
            fetch_days = max(fetch_days, self.xs_strategy.lookback + 120)

        # Fetch data
        data = self.feed.get_bars_multi(symbols, days=fetch_days)

        if not data:
            console.print("[red]No data available. Check your Alpaca API keys.[/red]")
            return []

        # Compute features
        enriched = {}
        for symbol, df in data.items():
            enriched[symbol] = self.features.compute_all(df)

        # Pairwise return correlations (for the correlation filter)
        self._correlations = self._compute_correlations(data)

        # Generate signals (mode-dependent)
        if self.xs_strategy:
            # Cross-sectional momentum bypasses the SPY market filter (evidence).
            self._market_ok = True
            held = {p.symbol for p in self.order_manager.broker.get_positions()}
            signals = self.xs_strategy.build_signals(enriched, held=held)
        else:
            # Market-regime filter flag: is SPY above its SMA? (ensemble only)
            self._market_ok = self._compute_market_ok(data)
            signals = self.ensemble.scan_universe(enriched)

        # Display
        self.dashboard.show_signals(signals)

        if signals:
            console.print(f"\n[green]Found {len(signals)} actionable signal(s)[/green]")
            for sig in signals:
                console.print(f"  * {sig.action.value.upper()} {sig.symbol} "
                            f"({sig.strategy}, confidence={sig.confidence:.0%})")
        else:
            console.print("\n[dim]No signals found in current scan[/dim]")

        return signals

    def _compute_correlations(self, data: dict, window: int = 60) -> dict:
        """Pairwise return correlations over the recent window: {sym: {sym: corr}}."""
        try:
            import pandas as pd
            closes = {s: df["close"] for s, df in data.items()
                      if df is not None and len(df) > 20}
            if len(closes) < 2:
                return {}
            rets = pd.DataFrame(closes).pct_change().tail(window)
            corr = rets.corr()
            return {s: corr[s].to_dict() for s in corr.columns}
        except Exception as e:
            logger.debug(f"Correlation computation failed: {e}")
            return {}

    def _compute_market_ok(self, data: dict) -> bool:
        """Is SPY above its configured SMA? Used by the market filter for longs."""
        if not self.config.market_filter:
            return True
        spy = data.get("SPY")
        if spy is None or spy.empty:
            return True  # permissive when unknown
        sma = self.config.market_filter_sma
        if len(spy) < sma:
            return True
        ma = spy["close"].rolling(sma).mean().iloc[-1]
        return bool(spy["close"].iloc[-1] > ma)

    async def plan(self):
        """
        Scan → build decision memos → print them. Never executes (paper preview).

        This is the seb.ai-style funnel as a one-shot: every signal becomes a
        structured APPROVED / WATCHLIST / REJECTED memo with a full trade plan,
        for you to review before deciding anything.
        """
        signals = await self.scan()
        if not signals:
            return []

        account = self.order_manager.broker.get_account()
        positions = self.order_manager.broker.get_positions()
        equity = account.get("equity", self.config.initial_capital)
        cash = account.get("cash", self.config.initial_capital)

        memos = []
        for signal in signals:
            memo = self.order_manager.decision_engine.build(
                signal, positions, equity, cash, market_ok=self._market_ok,
                correlations=self._correlations)
            self.store.save_decision(memo)
            memos.append(memo)

        self.dashboard.show_decisions(memos)
        console.print("\n[dim]Decision memos (paper preview only — nothing executed):[/dim]\n")
        for memo in memos:
            console.print(memo.render())
            console.print("")
        return memos

    async def run_cycle(self):
        """Run a single trading cycle: scan → risk check → execute."""
        try:
            signals = await self.scan()
            
            if signals:
                memos = await self.order_manager.process_signals(
                    signals, market_ok=self._market_ok, correlations=self._correlations)

                # Show decision memos and alert on actionable ones.
                self.dashboard.show_decisions(memos)
                explain = self.analyst.settings.get("explain_trades", False)
                for memo in memos:
                    if memo.status in (DecisionStatus.APPROVED, DecisionStatus.WATCHLIST):
                        await self.alerts.notify_decision(memo)
                    # Optional plain-English explanation for approved trades.
                    if explain and memo.status == DecisionStatus.APPROVED:
                        why = await self.analyst.explain_decision(memo)
                        if why:
                            await self.alerts.send(f"<i>{memo.symbol}: {why}</i>")

            # Check stop-losses on existing positions
            await self.order_manager.check_stop_losses()

            # Update dashboard
            account = self.order_manager.broker.get_account()
            positions = self.order_manager.broker.get_positions()
            risk_status = self.risk_manager.status
            self.dashboard.show_status(account, positions, risk_status)

        except Exception as e:
            logger.error(f"Error in trading cycle: {e}", exc_info=True)
            await self.alerts.notify_error(str(e))

    async def _run_scanner(self):
        """Run the market-wide scanner to discover new trading candidates."""
        from datetime import datetime
        try:
            console.print("\n[bold magenta]Running market-wide scanner...[/bold magenta]")
            new_universe = await self.scanner.scan()
            
            if new_universe:
                old_count = len(self._active_symbols)
                self._active_symbols = new_universe
                self._last_scan_date = datetime.now().strftime("%Y-%m-%d")
                
                new_symbols = [s for s in new_universe if s not in self.config.symbols]
                console.print(
                    f"[green]Scanner found {len(new_symbols)} new candidates. "
                    f"Active universe: {len(self._active_symbols)} symbols[/green]"
                )
                
                if new_symbols:
                    logger.info(f"New symbols from scanner: {new_symbols[:20]}{'...' if len(new_symbols) > 20 else ''}")
                    await self.alerts.send(
                        f"Scanner found {len(new_symbols)} new stocks! "
                        f"Total universe: {len(self._active_symbols)} symbols"
                    )
        except Exception as e:
            logger.error(f"Scanner error (using core symbols): {e}")
            # Fall back to core symbols on scanner failure
            self._active_symbols = list(self.config.symbols)

    async def _run_daily_tasks(self):
        """
        Once-per-calendar-day housekeeping:
        - Refresh strategy performance weights from realized P&L.
        - Send an end-of-day Telegram summary (for the day that just ended).
        """
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        if today == self._last_daily_date:
            return

        first_run = self._last_daily_date == ""
        self._last_daily_date = today

        # Learn from realized performance every day.
        self.ensemble.update_performance_weights(self.store)

        # Summarize the prior day on rollover (skip the very first startup cycle).
        if not first_run:
            await self._send_eod_summary()

        # Morning brief (the first of the "2 daily messages").
        await self._send_morning_brief()

    async def _send_eod_summary(self):
        """Send an end-of-day account + performance summary via Telegram."""
        try:
            account = self.order_manager.broker.get_account()
            positions = self.order_manager.broker.get_positions()
            risk_status = self.risk_manager.status
            stats = self.store.get_trade_stats()
            await self.alerts.notify_eod_summary(account, positions, risk_status, stats)

            # AI narrative (the second of the "2 daily messages").
            narrative = await self.analyst.eod_narrative(account, positions, risk_status, stats)
            if narrative:
                console.print(f"\n[bold cyan]AI Analyst — End of Day[/bold cyan]\n{narrative}\n")
                await self.alerts.send(f"<b>AI Analyst — End of Day</b>\n{narrative}")
        except Exception as e:
            logger.error(f"Failed to send EOD summary: {e}")

    async def _send_morning_brief(self):
        """Send a Claude-generated morning brief via Telegram (if enabled)."""
        if not self.analyst.enabled:
            return
        try:
            account = self.order_manager.broker.get_account()
            positions = self.order_manager.broker.get_positions()
            decisions = self.store.get_decisions(limit=20)
            brief = await self.analyst.morning_brief(account, positions, decisions)
            if brief:
                console.print(f"\n[bold cyan]AI Analyst — Morning Brief[/bold cyan]\n{brief}\n")
                await self.alerts.send(f"<b>AI Analyst — Morning Brief</b>\n{brief}")
        except Exception as e:
            logger.error(f"Failed to send morning brief: {e}")

    async def run(self):
        """
        Start the trading agent in continuous mode.
        Runs scans at configured intervals during market hours.
        """
        self._running = True
        scan_interval = self.config.settings.get("schedule", {}).get("scan_interval_minutes", 60)
        scanner_enabled = self.config.settings.get("scanner", {}).get("enabled", False)
        
        console.print(f"\n[bold green]Trading Agent started[/bold green]")
        console.print(f"   Mode: {'PAPER' if self.config.is_paper else 'LIVE'}")
        console.print(f"   Strategy: {self.config.strategy_mode.upper()}"
                      + (f" (12-mo XS momentum, top {self.xs_strategy.top_n})" if self.xs_strategy else ""))
        console.print(f"   Execution: {self.config.execution_mode.upper()} "
                      f"({'auto-trades APPROVED signals' if self.config.execution_mode == 'auto' else 'proposes only — human executes'})")
        console.print(f"   Core Symbols: {self.config.symbols}")
        console.print(f"   Market Scanner: {'ENABLED' if scanner_enabled else 'DISABLED'}")
        console.print(f"   Scan interval: {scan_interval} minutes")
        console.print(f"   Capital: ${self.config.initial_capital:.2f}")
        console.print(f"\n   Press Ctrl+C to stop\n")

        await self.alerts.send("Trading Agent started!")
        
        # Run market scanner on startup if enabled
        if scanner_enabled:
            await self._run_scanner()

        while self._running:
            try:
                # Daily housekeeping (perf weights refresh + EOD summary)
                await self._run_daily_tasks()

                # Check if market is open
                is_open = self.order_manager.broker.is_market_open()

                if is_open:
                    # Run daily scanner if we haven't today
                    if scanner_enabled:
                        from datetime import datetime
                        today = datetime.now().strftime("%Y-%m-%d")
                        if today != self._last_scan_date:
                            await self._run_scanner()
                    
                    await self.run_cycle()
                elif self.config.is_paper:
                    # Paper mode: still run cycles when market is closed
                    # so you can see the system work in real-time
                    logger.info("Market closed — running scan in PAPER mode anyway")
                    await self.run_cycle()
                else:
                    next_open = self.order_manager.broker.get_next_market_open()
                    logger.info(f"Market closed. Next open: {next_open}")

                # Wait for next cycle — chunked so shutdown (Ctrl+C sets
                # _running=False) is responsive instead of blocking a full interval.
                for _ in range(int(scan_interval * 60)):
                    if not self._running:
                        break
                    await asyncio.sleep(1)

            except (asyncio.CancelledError, KeyboardInterrupt):
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 min on error

        console.print("\n[yellow]Trading Agent stopped[/yellow]")
        await self.alerts.send("Trading Agent stopped")

    def stop(self):
        """Stop the agent gracefully."""
        self._running = False

    async def status(self):
        """Show current account status and positions."""
        account = self.order_manager.broker.get_account()
        positions = self.order_manager.broker.get_positions()
        risk_status = self.risk_manager.status
        self.dashboard.show_status(account, positions, risk_status)

    async def report(self):
        """
        Forward-test report: realized performance from the live/paper DB, so you
        can compare actual results to the backtest expectation over time.
        """
        from rich.table import Table

        stats = self.store.get_trade_stats()
        decisions = self.store.get_decisions(limit=500)
        account = self.order_manager.broker.get_account()

        total = stats.get("total_trades") or 0
        wins = stats.get("wins") or 0
        losses = stats.get("losses") or 0
        total_pnl = stats.get("total_pnl") or 0.0
        win_rate = (wins / total) if total else 0.0

        t = Table(title="Forward Paper-Test Report", box=box.ROUNDED)
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
        t.add_row("Account equity", f"${account.get('equity', 0):,.2f}")
        t.add_row("Closed trades", str(total))
        t.add_row("Win rate", f"{win_rate:.0%}")
        t.add_row("Wins / Losses", f"{wins} / {losses}")
        t.add_row("Realized P&L", f"${total_pnl:,.2f}")
        t.add_row("Best trade", f"${stats.get('best_trade') or 0:,.2f}")
        t.add_row("Worst trade", f"${stats.get('worst_trade') or 0:,.2f}")
        console.print(t)

        # Decision funnel counts
        from collections import Counter
        counts = Counter(d["status"] for d in decisions)
        console.print(f"\n[dim]Decisions logged: {len(decisions)} "
                      f"(APPROVED={counts.get('approved',0)}, "
                      f"WATCHLIST={counts.get('watchlist',0)}, "
                      f"REJECTED={counts.get('rejected',0)})[/dim]")
        if total == 0:
            console.print("[dim]No closed trades yet — let the paper run accumulate "
                          "a track record, then compare to the backtest.[/dim]")
        return stats

    async def doctor(self):
        """
        Health check: verify every external dependency the agent needs is
        working before you trust it to trade. Never raises — reports per check.
        """
        from pathlib import Path
        console.print("\n[bold cyan]Trading Agent — Health Check[/bold cyan]\n")
        checks: list[tuple[str, bool, str]] = []

        # 1. API keys present
        has_keys = bool(self.config.alpaca_api_key and
                        self.config.alpaca_api_key != "your_api_key_here")
        checks.append(("Alpaca API keys configured", has_keys,
                       "set ALPACA_API_KEY / ALPACA_SECRET_KEY in .env" if not has_keys else
                       f"mode={'PAPER' if self.config.is_paper else 'LIVE'}"))

        # 2. Broker reachable
        try:
            account = self.order_manager.broker.get_account()
            ok = bool(account) and account.get("status") not in (None, "NO_API_KEYS")
            detail = (f"equity=${account.get('equity', 0):,.2f}, status={account.get('status')}"
                      if ok else "could not fetch account")
            checks.append(("Broker reachable", ok, detail))
        except Exception as e:
            checks.append(("Broker reachable", False, str(e)))

        # 3. Market data feed
        try:
            probe = self.feed.get_bars(self.config.symbols[0] if self.config.symbols else "SPY", days=10)
            ok = probe is not None and not probe.empty
            checks.append(("Market data feed", ok,
                           f"fetched {len(probe)} bars" if ok else "no data returned"))
        except Exception as e:
            checks.append(("Market data feed", False, str(e)))

        # 4. Database writable
        try:
            self.store.conn.execute("SELECT 1")
            checks.append(("Database", True, self.config.db_path))
        except Exception as e:
            checks.append(("Database", False, str(e)))

        # 5. Telegram
        if self.alerts.enabled:
            try:
                import aiohttp
                url = f"https://api.telegram.org/bot{self.alerts.bot_token}/getMe"
                async with aiohttp.ClientSession() as s:
                    async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                        j = await r.json()
                ok = bool(j.get("ok"))
                detail = f"@{j['result']['username']}" if ok else str(j.get("description"))
                checks.append(("Telegram bot", ok, detail))
            except Exception as e:
                checks.append(("Telegram bot", False, str(e)))
        else:
            checks.append(("Telegram bot", True, "disabled (optional)"))

        # 6. Regime model
        model_exists = Path("data/regime_model.pkl").exists()
        checks.append(("ML regime model", True,
                       "trained model present" if model_exists else
                       "not trained yet — run `train` (heuristic fallback active)"))

        # 7. AI Analyst
        checks.append(("AI Analyst (Claude)", True,
                       f"enabled ({self.analyst.model})" if self.analyst.enabled else
                       "disabled (optional — set ANTHROPIC_API_KEY to enable)"))

        # Render
        from rich.table import Table
        table = Table(box=box.SIMPLE_HEAVY)
        table.add_column("Check", style="bold")
        table.add_column("Status")
        table.add_column("Detail", style="dim")
        all_ok = True
        for name, ok, detail in checks:
            if not ok:
                all_ok = False
            table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", str(detail))
        console.print(table)

        if all_ok:
            console.print("\n[bold green]All checks passed — agent is ready.[/bold green]")
        else:
            console.print("\n[bold red]Some checks failed — resolve the FAILs above before running.[/bold red]")
        return all_ok

    async def backtest(self, days: int = 365):
        """Run a backtest on historical data."""
        from .backtest.engine import BacktestEngine

        console.print(f"\n[bold cyan]Running backtest ({days} days)...[/bold cyan]\n")

        # Fetch historical data
        data = self.feed.get_bars_multi(self.config.symbols, days=days)
        
        if not data:
            console.print("[red]No data available for backtesting[/red]")
            return

        # Enrich with features first (needed for regime training)
        enriched = {}
        for symbol, df in data.items():
            enriched[symbol] = self.features.compute_all(df)

        # Train regime classifier on the historical data
        console.print("[dim]Training ML regime classifier...[/dim]")
        self.ensemble.train_classifier(enriched)

        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.config.initial_capital,
            max_risk_per_trade=self.config.max_risk_per_trade,
            max_position_size=self.config.max_position_size,
            max_positions=self.config.max_open_positions,
        )
        
        result = engine.run(data)
        result.print_summary()
        
        return result

    async def walkforward(self, days: int = 1260):
        """
        Run a walk-forward, out-of-sample evaluation vs SPY buy-and-hold.

        This is the honest 'is there any edge?' test. Uses the cached feed so
        repeat runs don't re-hit the API.
        """
        from .data.cache import CachedFeed
        from .research.walkforward import WalkForward

        days = max(days, 1000)  # need enough history for train + test folds
        console.print(f"\n[bold cyan]Walk-forward evaluation ({days} days of history)...[/bold cyan]")
        console.print("[dim]Training regime model in-sample per fold, testing out-of-sample.[/dim]\n")

        cached = CachedFeed(self.feed)
        wf = WalkForward(
            cached,
            self.config.symbols,
            initial_capital=self.config.initial_capital,
            max_risk_per_trade=self.config.max_risk_per_trade,
            max_position_size=self.config.max_position_size,
            max_positions=self.config.max_open_positions,
        )
        return wf.run(total_days=days)

    async def experiments(self, days: int = 1500):
        """
        Run a walk-forward comparison of strategy improvements vs SPY:
        baseline -> drop momentum -> +market filter -> +vol targeting -> +cross-sectional.
        """
        from .data.cache import CachedFeed
        from .research.walkforward import compare_experiments
        from .research.experiment import ExperimentConfig

        days = max(days, 2400)  # ~6.5y so folds span the 2022 bear, not just bull
        console.print(f"\n[bold cyan]Strategy experiment comparison ({days} days)...[/bold cyan]\n")

        base = ("mean_reversion", "breakout")  # momentum excluded (loses money)
        cached = CachedFeed(self.feed)
        exps = [
            ExperimentConfig(name="with_momentum",
                             strategies=("momentum", "mean_reversion", "breakout")),
            ExperimentConfig(name="no_momentum", strategies=base),
            ExperimentConfig(name="+market_filter_only", strategies=base, market_filter=True),
            ExperimentConfig(name="+vol_target_only", strategies=base, vol_target=0.20),
            ExperimentConfig(name="+xsectional_only", strategies=base, cross_sectional_top=5),
            ExperimentConfig(name="vol+market", strategies=base, market_filter=True, vol_target=0.20),
            ExperimentConfig(name="xs_momentum", strategies=base, xs_momentum=True,
                             xs_lookback=120, xs_top=6),
        ]
        return compare_experiments(
            cached, self.config.symbols, exps, total_days=days,
            initial_capital=self.config.initial_capital,
            max_risk_per_trade=self.config.max_risk_per_trade,
            max_position_size=self.config.max_position_size,
            max_positions=self.config.max_open_positions,
        )

    async def train(self, days: int = 365):
        """Train the ML regime classifier on historical data."""
        console.print(f"\n[bold cyan]Training ML regime classifier ({days} days)...[/bold cyan]\n")

        data = self.feed.get_bars_multi(self.config.symbols, days=days)
        
        if not data:
            console.print("[red]No data available for training[/red]")
            return

        enriched = {}
        for symbol, df in data.items():
            enriched[symbol] = self.features.compute_all(df)

        success = self.ensemble.train_classifier(enriched)
        
        if success:
            console.print("\n[bold green]Regime classifier trained and saved![/bold green]")
            
            # Show what regime each symbol is currently in
            from rich.table import Table
            from rich import box
            
            table = Table(title="Current Market Regimes", box=box.SIMPLE_HEAVY)
            table.add_column("Symbol", style="bold")
            table.add_column("Regime", style="cyan")
            table.add_column("Strategy Weights")
            
            for symbol, df in enriched.items():
                regime = self.ensemble.classifier.classify(df)
                weights = self.ensemble.classifier.get_strategy_weights(regime)
                weights_str = ", ".join([f"{k}={v:.1f}" for k, v in weights.items()])
                
                table.add_row(symbol, regime.value, weights_str)
            
            console.print(table)
        else:
            console.print("[yellow]Not enough data to train. Try with more history.[/yellow]")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Trading Agent")
    parser.add_argument(
        "command",
        choices=["scan", "plan", "run", "status", "report", "backtest", "train", "walkforward", "experiments", "doctor"],
        help="Command to execute",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Number of days for backtest/training (default: 365)",
    )

    args = parser.parse_args()

    agent = TradingAgent()

    if args.command == "scan":
        asyncio.run(agent.scan())
    elif args.command == "plan":
        asyncio.run(agent.plan())
    elif args.command == "run":
        # Graceful shutdown: Ctrl+C flips the running flag; the chunked sleep in
        # run() notices within a second and exits cleanly (sends "stopped" alert).
        def handle_signal(signum, frame):
            console.print("\n[yellow]Shutdown requested — finishing current cycle...[/yellow]")
            agent.stop()
        signal.signal(signal.SIGINT, handle_signal)
        asyncio.run(agent.run())
    elif args.command == "status":
        asyncio.run(agent.status())
    elif args.command == "report":
        asyncio.run(agent.report())
    elif args.command == "doctor":
        ok = asyncio.run(agent.doctor())
        sys.exit(0 if ok else 1)
    elif args.command == "backtest":
        asyncio.run(agent.backtest(days=args.days))
    elif args.command == "train":
        asyncio.run(agent.train(days=args.days))
    elif args.command == "walkforward":
        asyncio.run(agent.walkforward(days=args.days))
    elif args.command == "experiments":
        asyncio.run(agent.experiments(days=args.days))


if __name__ == "__main__":
    main()
