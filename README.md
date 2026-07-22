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

# A/B comparison of strategy improvements (walk-forward vs SPY)
python -m src.main experiments --days 2600

# Forward paper-test report (realized performance vs backtest, from the DB)
python -m src.main report

# Live web dashboard (equity curve, positions, decisions, slippage, risk)
python -m src.main dashboard

# AI Analyst (needs ANTHROPIC_API_KEY)
python -m src.main review                              # LLM post-mortem on closed trades
python -m src.main diagnose                            # strategy contribution + decay analysis
python -m src.main ask --question "why did we skip NVDA?"   # NL query over the decision log

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
- **AI Analyst**: Claude-powered morning brief + end-of-day narrative (the "2 daily messages")
- **Risk Management**: sizing, exposure limits, trailing stops, circuit breakers, **correlation filter**
- **Multi-asset ready**: equities + optional crypto (BTC/ETH) via Alpaca
- **Restart-safe**: open trades persist to SQLite and are restored + reconciled on startup
- **Paper Trading**: Alpaca paper trading with fractional shares
- **Telegram Alerts**: signal / error / start-stop pings + daily end-of-day summary
- **Backtesting**: next-bar-open execution (no look-ahead) with slippage + commission

## Running it 24/7 (Docker)

```bash
cp .env.example .env      # add your keys
docker compose up -d      # agent + dashboard, restart on crash/reboot
docker compose logs -f agent
# dashboard: http://localhost:8501
```

**Liveness & safety**
- The agent writes `data/heartbeat.txt` each cycle; Docker's `HEALTHCHECK` uses it.
- `python -m src.ops.watchdog` (run hourly from cron/Task Scheduler) sends a
  Telegram alert if the agent stops completing cycles, and one when it recovers.
- **Kill switch:** `touch data/KILL` flattens every position and halts the agent.
- Config is validated at startup — it refuses to run on percent-typos
  (`0.10` vs `10`), unknown strategy modes, or incoherent risk limits.

## Brokers

The agent talks to a `BaseBroker` interface, so switching brokers is a config
change (`general.broker` in `config/settings.yaml`), not a rewrite:

- `alpaca` — real paper/live account via API keys (default)
- `paper` — a fully local simulated broker (`PaperBroker`) that needs **no
  account and no keys**. It fills orders at the signal price plus slippage,
  marks positions to market each cycle, and persists to `data/paper_state.json`.
  Great for demos, offline development, and running the whole loop with nothing
  configured. Future adapters (IBKR, Binance) implement the same interface.

## Security & key hygiene

- Secrets live only in `.env` (gitignored, never committed, never baked into
  the Docker image — it's injected via `env_file`).
- Use **paper** keys for anything that isn't a validated live system, and
  scope live keys to trading only. Rotate them if they ever appear in a log,
  screenshot, or shell history.
- The container runs as a **non-root** user and mounts `config/` read-only;
  the dashboard mounts `data/` read-only and never places orders.
- Built-in broker guards: exponential backoff on rate limits/5xx, a
  **Pattern Day Trader** guard (refuses new longs at 3+ day-trades on a
  sub-$25k account), and a block on trading if the broker flags the account.
- Going live from India involves LRS/overseas-investment rules and an
  Alpaca-eligibility question — settle that before flipping `ALPACA_PAPER`.

## Forward paper-test runbook

The active strategy is `xs_momentum` (the config that beat SPY out-of-sample).
To forward-test it on paper:

```bash
python -m src.main doctor    # confirm everything is wired
python -m src.main run       # start the loop (rotates into 12-month momentum leaders)
python -m src.main report    # check realized results vs the backtest over time
```

Watch `report` and the Telegram messages over weeks/months and compare to the
~27% CAGR / ~30% drawdown backtest. Switch `general.strategy_mode` to `ensemble`
in `config/settings.yaml` for the lower-drawdown TA config. Enable the AI Analyst
by adding `ANTHROPIC_API_KEY` to `.env`.

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

> Walk-forward findings (2019–2026 out-of-sample, includes the 2022 bear):
> - **momentum** loses money across the full cycle → **disabled** (config).
> - **market filter** (SPY > 200-day SMA for new longs) roughly **halved max
>   drawdown** (≈25% → 14%) for the best Sharpe, robust across 150/200/250-day
>   SMAs → **adopted live** (`risk.market_filter`).
> - **volatility-target sizing** further reduced drawdown → **adopted live**
>   (`risk.vol_target`); combined market+vol gave the lowest DD (≈11%).
> - **cross-sectional momentum** had the highest raw return (7.4% CAGR) but the
>   highest drawdown (34%) — kept as a research option, not adopted.
>
> The TA ensemble still does **not** beat SPY buy-and-hold — its adopted changes
> are about **not blowing up**.

### Cross-sectional momentum (`strategy_mode: xs_momentum`)

Ranking a broad (~70-name) universe by **12-month trailing return** and holding
the top 8 — the classic momentum factor — beats SPY buy-and-hold out-of-sample.
Tested two ways:

**Full cycle (2005–2026, incl. the 2008 crash — deep Yahoo history, 38 folds):**

| Config | CAGR | Sharpe | Max DD | Worst fold (2008) |
|---|---|---|---|---|
| SPY buy & hold | 10.5% | 0.61 | −55.2% | — |
| XS momentum (raw) | 13.6% | 0.79 | −40.6% | −29.6% |
| **XS momentum + vol target (deployed)** | **10.9%** | **0.83** | **−29.4%** | −21.0% |

> **This is the honest expectation: ~11–14% CAGR with 30–40% drawdowns**, beating
> SPY mostly on *risk-adjusted* terms (Sharpe 0.83 vs 0.61, roughly half the
> drawdown). Volatility targeting is adopted because it demonstrably softened the
> 2008 momentum crash.

The recent 2019–2026 sub-window ran much hotter (~27% CAGR) — a favorable regime,
**not** a sustainable expectation. The forward-test baseline uses the full-cycle
number so live results are judged against a bar that already includes a crash.

> **Honest caveats:** this is one ~7-year window that excludes the 2009 momentum
> crash; drawdown is ~30% (tameable to ~20% via `cross_sectional.vol_target`);
> momentum is high-turnover and crowded. Promising and evidence-backed, but
> **paper-trade it for months** (`python -m src.main report`) before trusting it.

## Notes on the backtest

The backtester decides signals on bar *t*'s close and fills them on bar *t+1*'s
**open** — never the same close — to avoid look-ahead bias. Every fill is charged
slippage (default 5 bps) and optional commission. Stop-loss / take-profit are
checked intrabar against each bar's high/low, with gap-through fills at the open.
Backtest results are an estimate, not a promise — validate out-of-sample before
trusting any strategy with real money.
