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

## Notes on the backtest

The backtester decides signals on bar *t*'s close and fills them on bar *t+1*'s
**open** — never the same close — to avoid look-ahead bias. Every fill is charged
slippage (default 5 bps) and optional commission. Stop-loss / take-profit are
checked intrabar against each bar's high/low, with gap-through fills at the open.
Backtest results are an estimate, not a promise — validate out-of-sample before
trusting any strategy with real money.
