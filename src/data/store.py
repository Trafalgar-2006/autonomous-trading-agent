"""
Data Store — SQLite persistence for trades, signals, and portfolio snapshots.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from ..core.config import Config
from ..core.models import Signal, Trade, TradeOutcome

logger = logging.getLogger(__name__)


class DataStore:
    """SQLite-based persistence layer."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_db()
        return cls._instance

    def _init_db(self):
        """Initialize the database and create tables."""
        config = Config()
        db_path = Path(config.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                strategy TEXT,
                entry_time TEXT,
                entry_price REAL,
                quantity REAL,
                exit_time TEXT,
                exit_price REAL,
                exit_reason TEXT,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                outcome TEXT DEFAULT 'open',
                signal_id TEXT,
                entry_reasoning TEXT DEFAULT '{}',
                exit_reasoning TEXT DEFAULT '{}',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                strategy TEXT,
                confidence REAL,
                reasoning TEXT DEFAULT '{}',
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                regime TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cash REAL,
                equity REAL,
                positions_value REAL,
                open_positions INTEGER,
                daily_pnl REAL,
                daily_pnl_pct REAL,
                total_pnl REAL,
                total_pnl_pct REAL,
                drawdown REAL,
                max_drawdown REAL
            );

            CREATE TABLE IF NOT EXISTS daily_performance (
                date TEXT PRIMARY KEY,
                starting_equity REAL,
                ending_equity REAL,
                pnl REAL,
                pnl_pct REAL,
                trades_taken INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                max_drawdown REAL DEFAULT 0
            );
        """)
        self.conn.commit()
        logger.info(f"DataStore initialized at {db_path}")

    def save_signal(self, signal: Signal):
        """Save a signal to the database."""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO signals (id, timestamp, symbol, action, strategy, "
                "confidence, reasoning, entry_price, stop_loss, take_profit, regime) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    signal.id,
                    signal.timestamp.isoformat(),
                    signal.symbol,
                    signal.action.value,
                    signal.strategy,
                    signal.confidence,
                    json.dumps(signal.reasoning),
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.regime.value if signal.regime else None,
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving signal: {e}")

    def save_trade(self, trade: Trade):
        """Save or update a trade in the database."""
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO trades (id, symbol, side, strategy, entry_time, "
                "entry_price, quantity, exit_time, exit_price, exit_reason, pnl, pnl_pct, "
                "commission, outcome, signal_id, entry_reasoning, exit_reasoning) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trade.id,
                    trade.symbol,
                    trade.side.value,
                    trade.strategy,
                    trade.entry_time.isoformat() if trade.entry_time else None,
                    trade.entry_price,
                    trade.quantity,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.exit_price,
                    trade.exit_reason,
                    trade.pnl,
                    trade.pnl_pct,
                    trade.commission,
                    trade.outcome.value,
                    trade.signal_id,
                    json.dumps(trade.entry_reasoning),
                    json.dumps(trade.exit_reasoning),
                ),
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

    def get_trades(self, limit: int = 100) -> list[dict]:
        """Get recent trades."""
        cursor = self.conn.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_signals(self, limit: int = 100) -> list[dict]:
        """Get recent signals."""
        cursor = self.conn.execute(
            "SELECT * FROM signals ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_trade_stats(self) -> dict:
        """Get aggregate trade statistics."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_trades,
                SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades
            WHERE outcome != 'open'
        """)
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {}
