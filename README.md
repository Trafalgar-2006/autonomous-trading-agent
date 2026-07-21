# AI Trading Agent

[![CI](https://github.com/Trafalgar-2006/autonomous-trading-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Trafalgar-2006/autonomous-trading-agent/actions/workflows/ci.yml)

Autonomous AI-powered trading agent that uses an ensemble of technical strategies with ML regime detection to trade US equities via Alpaca.

## Architecture

```
src/
├── main.py                 # Orchestrator & CLI entry point
├── core/
│   ├── config.py           # YAML + env config loader (singleton)
│   ├── models.py           # Data models (Signal, Order, Trade, Position)
│   └── event_bus.py        # Async pub/sub event bus
├── data/
│   ├── feed.py             # Alpaca market data fetcher
│   ├── features.py         # Technical indicator engine
│   ├── store.py            # SQLite persistence
│   └── scanner.py          # Market-wide stock scanner
├── strategy/
│   ├── base.py             # Abstract strategy interface
│   ├── momentum.py         # RSI/MACD/EMA trend-following
│   ├── mean_reversion.py   # Bollinger Bands/RSI/Z-score
│   ├── breakout.py         # Donchian channel breakouts
│   ├── ml_classifier.py    # Random Forest regime classifier
│   └── ensemble.py         # Weighted signal aggregation
├── execution/
│   ├── broker.py           # Alpaca API wrapper
│   └── order_manager.py    # Signal → risk → execution pipeline
├── risk/
│   └── manager.py          # Position sizing, stops, circuit breakers
├── monitoring/
│   ├── alerts.py           # Telegram notifications
│   └── dashboard.py        # Rich CLI dashboard
└── backtest/
    └── engine.py           # Walk-forward backtester
```

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e .

# Configure
cp .env.example .env
# Edit .env with your Alpaca API keys
```

## Usage

```bash
# Health check — verify keys, broker, data feed, DB, Telegram, model
python -m src.main doctor

# One-shot market scan (no trades)
python -m src.main scan

# Decision memos: scan -> APPROVED / WATCHLIST / REJECTED plans (never executes)
python -m src.main plan

# Walk-forward, out-of-sample evaluation vs SPY buy-and-hold
python -m src.main walkforward --days 1500

# Start continuous paper trading (Ctrl+C for graceful shutdown)
python -m src.main run

# Check account status
python -m src.main status

# Run backtest
python -m src.main backtest --days 365

# Train ML regime classifier
python -m src.main train --days 365
```

## Recommended runbook

```bash
python -m src.main doctor          # 1. confirm everything is wired up
python -m src.main train --days 365 # 2. train the regime model
python -m src.main backtest --days 365  # 3. sanity-check on history
python -m src.main scan             # 4. see current signals (no trades)
python -m src.main run              # 5. go live on paper
```

## Testing

```bash
pip install -e ".[dev]"   # installs pytest
pytest                    # runs the full suite
```

Tests cover the money-critical paths: position sizing, circuit breakers,
trailing-stop math, trade persistence round-trip, feature computation, and a
backtest smoke test.

## Features

- **3 Strategies**: Momentum, Mean Reversion, Breakout
- **ML Regime Detection**: Random Forest classifies market into 5 regimes
- **Weekly Trend Filter**: Multi-timeframe analysis gates daily signals
- **Per-strategy feedback**: strategies are auto-weighted by realized P&L
- **Risk Management**: Position sizing, exposure limits, trailing stops, circuit breakers
- **Restart-safe**: open trades persist to SQLite and are restored + reconciled on startup
- **Paper Trading**: Alpaca paper trading with fractional shares
- **Telegram Alerts**: signal / error / start-stop pings + daily end-of-day summary
- **Backtesting**: next-bar-open execution (no look-ahead) with slippage + commission

## Decision memos & execution mode

Every actionable signal produces a structured **decision memo** — a full trade
plan (entry / target / stop / invalidation / timeframe / risk:reward) with a
verdict of **APPROVED**, **WATCHLIST** (a soft gate failed), or **REJECTED**
(a hard risk rule blocked it). Memos are persisted to SQLite for an audit trail.

`config/settings.yaml -> general.execution_mode`:
- `auto` — the agent trades APPROVED signals automatically (default)
- `propose` — the agent only produces memos; **a human executes** (seb.ai-style)

## Research & validation (`src/research/`)

`python -m src.main walkforward` runs a **walk-forward, out-of-sample** test: the
regime model is trained only on past data in each fold, the strategies are run on
the next unseen window, and the concatenated out-of-sample returns are compared
head-to-head with **SPY buy-and-hold**. A disk cache (`data/cache/`) avoids
re-fetching bars, and a precomputed-signal fast backtest keeps sweeps quick.

`python -m src.main experiments` runs a walk-forward **A/B comparison** of
strategy improvements (drop momentum, SPY>200SMA market filter, volatility-target
sizing, cross-sectional top-N) side-by-side against SPY.

> Walk-forward findings (multi-year out-of-sample):
> - **baseline** (3 strategies): −1.5% CAGR — momentum was actively losing money.
> - **momentum disabled**: +5.8% CAGR / 0.50 Sharpe, lower drawdown → now the default.
> - **volatility-target sizing** roughly halved max drawdown (12.8% → 6.8%).
> - Market filter / cross-sectional were mixed in a mostly-bull window (not adopted).
>
> Even improved, the system did **not** beat SPY buy-and-hold (~17% CAGR). Treat it
> as a research platform, not a money-maker, until a real edge is demonstrated.

## Notes on the backtest

The backtester decides signals on bar *t*'s close and fills them on bar *t+1*'s
**open** — never the same close — to avoid look-ahead bias. Every fill is charged
slippage (default 5 bps) and optional commission. Stop-loss / take-profit are
checked intrabar against each bar's high/low, with gap-through fills at the open.
Backtest results are an estimate, not a promise — validate out-of-sample before
trusting any strategy with real money.
