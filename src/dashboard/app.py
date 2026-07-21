"""
Live web dashboard (Streamlit).

Read-only view of everything the agent is doing: portfolio, equity curve,
open positions, the decision funnel, strategy performance, risk state, and
agent liveness. Reads the SQLite store and the broker — it never places orders.

Run:
    python -m src.main dashboard
    # or: streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make `src` importable when launched directly by streamlit.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.core.config import Config
from src.data.store import DataStore

st.set_page_config(page_title="Trading Agent", page_icon="📈", layout="wide")


# ── Data helpers ──────────────────────────────────────────────────

@st.cache_resource
def get_store() -> DataStore:
    return DataStore()


@st.cache_resource
def get_config() -> Config:
    return Config()


def load_broker_state():
    """Fetch live account/positions. Returns (account, positions, error)."""
    try:
        from src.execution.broker import AlpacaBroker
        broker = AlpacaBroker()
        return broker.get_account(), broker.get_positions(), None
    except Exception as e:
        return {}, [], str(e)


def heartbeat_age_seconds():
    hb = ROOT / "data" / "heartbeat.txt"
    if not hb.exists():
        return None
    try:
        ts = datetime.fromisoformat(hb.read_text().strip())
        return (datetime.utcnow() - ts).total_seconds()
    except Exception:
        return None


# ── Sidebar ───────────────────────────────────────────────────────

cfg = get_config()
store = get_store()

st.sidebar.title("Trading Agent")
st.sidebar.caption("Read-only monitor — this page never places orders.")

mode = cfg.settings.get("general", {}).get("strategy_mode", "ensemble")
st.sidebar.metric("Strategy mode", mode)
st.sidebar.metric("Execution", cfg.execution_mode.upper())
st.sidebar.metric("Trading mode", "PAPER" if cfg.is_paper else "LIVE")

age = heartbeat_age_seconds()
if age is None:
    st.sidebar.warning("Agent heartbeat: never seen")
elif age < 3600:
    st.sidebar.success(f"Agent alive ({int(age)}s ago)")
else:
    st.sidebar.error(f"Agent stale ({age / 3600:.1f}h ago)")

if st.sidebar.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()

auto = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto:
    st.sidebar.caption("Page reloads every 30 seconds.")


# ── Header metrics ────────────────────────────────────────────────

account, positions, err = load_broker_state()
if err:
    st.error(f"Broker unavailable: {err}")

equity = account.get("equity", 0) or 0
cash = account.get("cash", 0) or 0
open_pnl = sum(getattr(p, "unrealized_pnl", 0) or 0 for p in positions)

st.title("Portfolio")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Equity", f"${equity:,.2f}")
c2.metric("Cash", f"${cash:,.2f}")
c3.metric("Open positions", len(positions))
c4.metric("Unrealized P&L", f"${open_pnl:,.2f}", delta=f"{open_pnl:,.2f}")


# ── Equity curve ──────────────────────────────────────────────────

st.subheader("Equity curve")
snaps = store.get_snapshots(limit=5000)
if snaps:
    df = pd.DataFrame(snaps)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")
    st.line_chart(df[["equity"]], height=280)

    # Drawdown from the snapshot series
    eq = df["equity"].astype(float)
    peak = eq.cummax()
    dd = (peak - eq) / peak.replace(0, pd.NA)
    st.caption(f"Max drawdown since tracking began: {float(dd.max() or 0):.2%}")
    with st.expander("Drawdown chart"):
        st.area_chart(-dd.fillna(0), height=180)
else:
    st.info("No snapshots yet — they're recorded each cycle once `run` is going.")


# ── Positions ─────────────────────────────────────────────────────

st.subheader("Open positions")
if positions:
    st.dataframe(
        pd.DataFrame([{
            "Symbol": p.symbol,
            "Qty": round(p.quantity, 4),
            "Entry": round(p.entry_price, 2),
            "Current": round(p.current_price, 2),
            "P&L $": round(p.unrealized_pnl, 2),
            "P&L %": f"{(p.unrealized_pnl_pct or 0) * 100:.2f}%",
            "Value": round(p.market_value, 2),
        } for p in positions]),
        width="stretch", hide_index=True,
    )
else:
    st.caption("No open positions.")


# ── Decision funnel ───────────────────────────────────────────────

st.subheader("Decision funnel")
decisions = store.get_decisions(limit=200)
if decisions:
    ddf = pd.DataFrame(decisions)
    counts = ddf["status"].value_counts().to_dict()
    d1, d2, d3 = st.columns(3)
    d1.metric("Approved", counts.get("approved", 0))
    d2.metric("Watchlist", counts.get("watchlist", 0))
    d3.metric("Rejected", counts.get("rejected", 0))

    show = ddf[["timestamp", "symbol", "action", "status", "strategy",
                "signal_strength", "risk_reward", "reasons"]].head(50).copy()
    show["signal_strength"] = (show["signal_strength"].astype(float) * 100).round(0)
    st.dataframe(show, width="stretch", hide_index=True)
else:
    st.caption("No decisions logged yet — run `scan`, `plan`, or `run`.")


# ── Strategy performance & trades ─────────────────────────────────

left, right = st.columns(2)

with left:
    st.subheader("Strategy performance")
    perf = store.get_strategy_performance()
    if perf:
        st.dataframe(pd.DataFrame([
            {"Strategy": k, "Trades": v["trades"], "Win rate": f"{v['win_rate']:.0%}",
             "P&L": round(v["pnl"], 2)}
            for k, v in sorted(perf.items(), key=lambda kv: kv[1]["pnl"], reverse=True)
        ]), width="stretch", hide_index=True)
    else:
        st.caption("No closed trades yet.")

with right:
    st.subheader("Risk")
    stats = store.get_trade_stats()
    total = stats.get("total_trades") or 0
    wins = stats.get("wins") or 0
    st.metric("Closed trades", total)
    st.metric("Win rate", f"{(wins / total if total else 0):.0%}")
    st.metric("Realized P&L", f"${stats.get('total_pnl') or 0:,.2f}")
    st.caption(f"Max positions {cfg.max_open_positions} · "
               f"max exposure {cfg.max_total_exposure:.0%} · "
               f"risk/trade {cfg.max_risk_per_trade:.1%} · "
               f"market filter {'ON' if cfg.market_filter else 'OFF'}")

st.subheader("Forward test vs backtest")
_baseline = cfg.settings.get("forward_test", {}).get("baseline", {})
if _baseline:
    from src.research.drift import (
        compare_to_baseline,
        elapsed_trading_days,
        live_metrics_from_snapshots,
    )
    _stats = store.get_trade_stats()
    _live = live_metrics_from_snapshots(snaps or [])
    _drift = compare_to_baseline(
        _live, _baseline,
        n_days=elapsed_trading_days(snaps or []),
        n_trades=_stats.get("total_trades") or 0,
    )
    _label = {
        "too_early": ("Too early to tell", st.info),
        "on_track": ("On track", st.success),
        "underperforming": ("Underperforming (within noise)", st.warning),
        "diverged": ("Diverged from backtest", st.error),
    }.get(_drift["verdict"], (_drift["verdict"], st.info))
    _label[1](f"**{_label[0]}** — {_drift['n_days']} trading days, "
              f"{_drift['n_trades']} closed trades")
    st.dataframe(pd.DataFrame([
        {"Metric": "CAGR", "Live": f"{_drift['live']['cagr']:.2%}",
         "Backtest": f"{_drift['baseline']['cagr']:.2%}"},
        {"Metric": "Sharpe", "Live": f"{_drift['live']['sharpe']:.2f}",
         "Backtest": f"{_drift['baseline']['sharpe']:.2f}"},
        {"Metric": "Max drawdown", "Live": f"{_drift['live']['max_drawdown']:.2%}",
         "Backtest": f"{_drift['baseline']['max_drawdown']:.2%}"},
    ]), width="stretch", hide_index=True)
    for _n in _drift["notes"]:
        st.caption(_n)

st.subheader("Execution quality")
slip = store.get_slippage_stats()
fills = store.get_fills(limit=50)
if slip and (slip.get("n") or 0) > 0:
    s1, s2, s3 = st.columns(3)
    s1.metric("Fills recorded", int(slip.get("n") or 0))
    s2.metric("Avg slippage", f"{slip.get('avg_bps') or 0:.1f} bps")
    s3.metric("Worst slippage", f"{slip.get('worst_bps') or 0:.1f} bps")
    st.caption("Positive = worse than expected. Compare with the backtest's 5 bps assumption.")
    with st.expander("Recent fills"):
        st.dataframe(
            pd.DataFrame(fills)[["timestamp", "symbol", "side", "expected_price",
                                 "fill_price", "quantity", "slippage_bps", "status"]],
            width="stretch", hide_index=True,
        )
else:
    st.caption("No fills recorded yet — they appear once the agent executes an order.")

st.subheader("Recent trades")
trades = store.get_trades(limit=50)
if trades:
    tdf = pd.DataFrame(trades)[
        ["symbol", "side", "strategy", "entry_price", "exit_price", "pnl", "outcome", "exit_reason"]
    ]
    st.dataframe(tdf, width="stretch", hide_index=True)
else:
    st.caption("No trades recorded yet.")

st.caption("Paper trading. Not financial advice. Figures come from the agent's own "
           "database and the broker API.")

if auto:
    import time
    time.sleep(30)
    st.rerun()
