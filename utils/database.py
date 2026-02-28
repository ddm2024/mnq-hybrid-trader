"""
Database layer — SQLite for trade logs, session state, and performance tracking.
"""

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.db")


def get_connection():
    """Get SQLite connection, creating DB/tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _create_tables(conn)
    return conn


def _create_tables(conn):
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            symbol TEXT NOT NULL DEFAULT 'MNQ',
            strategy TEXT NOT NULL,
            direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
            entry_price REAL NOT NULL,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            quantity INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'OPEN' CHECK(status IN ('OPEN', 'CLOSED', 'CANCELLED')),
            pnl REAL DEFAULT 0.0,
            pnl_pct REAL DEFAULT 0.0,
            commission REAL DEFAULT 0.0,
            slippage REAL DEFAULT 0.0,
            exit_reason TEXT,
            notes TEXT,
            session_date TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            closed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS daily_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            gross_pnl REAL DEFAULT 0.0,
            net_pnl REAL DEFAULT 0.0,
            total_commission REAL DEFAULT 0.0,
            max_drawdown REAL DEFAULT 0.0,
            win_rate REAL DEFAULT 0.0,
            avg_rr REAL DEFAULT 0.0,
            sharpe REAL DEFAULT 0.0,
            account_balance REAL DEFAULT 0.0,
            strategy TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(session_date);
        CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
        CREATE INDEX IF NOT EXISTS idx_daily_date ON daily_summary(date);
    """)
    conn.commit()


def log_trade(trade: Dict) -> int:
    """Insert a new trade record. Returns the trade ID."""
    conn = get_connection()
    cursor = conn.execute("""
        INSERT INTO trades (timestamp, symbol, strategy, direction, entry_price, 
                           stop_loss, take_profit, quantity, status, session_date, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?)
    """, (
        trade.get("timestamp", datetime.now().isoformat()),
        trade.get("symbol", "MNQ"),
        trade.get("strategy", "hybrid1"),
        trade["direction"],
        trade["entry_price"],
        trade.get("stop_loss"),
        trade.get("take_profit"),
        trade.get("quantity", 1),
        trade.get("session_date", datetime.now().strftime("%Y-%m-%d")),
        trade.get("notes", ""),
    ))
    conn.commit()
    trade_id = cursor.lastrowid
    conn.close()
    return trade_id


def close_trade(trade_id: int, exit_price: float, exit_reason: str, 
                pnl: float, commission: float = 0.0, slippage: float = 0.0):
    """Close an existing trade with exit details."""
    conn = get_connection()
    conn.execute("""
        UPDATE trades SET 
            exit_price = ?, status = 'CLOSED', pnl = ?, pnl_pct = ?,
            commission = ?, slippage = ?, exit_reason = ?, closed_at = ?
        WHERE id = ?
    """, (
        exit_price, pnl, 0.0, commission, slippage, exit_reason,
        datetime.now().isoformat(), trade_id
    ))
    conn.commit()
    conn.close()


def get_open_trades() -> List[Dict]:
    """Get all currently open trades."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM trades WHERE status = 'OPEN' ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_trades_today(date_str: Optional[str] = None) -> List[Dict]:
    """Get all trades for a given date (default: today)."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM trades WHERE session_date = ? ORDER BY created_at", (date_str,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_all_trades(limit: int = 500) -> pd.DataFrame:
    """Get trade history as DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY closed_at DESC LIMIT ?",
        conn, params=(limit,)
    )
    conn.close()
    return df


def get_daily_pnl(date_str: Optional[str] = None) -> float:
    """Get total P&L for a date."""
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")
    conn = get_connection()
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE session_date = ? AND status = 'CLOSED'",
        (date_str,)
    ).fetchone()
    conn.close()
    return row["total"] if row else 0.0


def save_daily_summary(summary: Dict):
    """Upsert daily performance summary."""
    conn = get_connection()
    conn.execute("""
        INSERT OR REPLACE INTO daily_summary 
            (date, total_trades, winning_trades, losing_trades, gross_pnl, net_pnl,
             total_commission, max_drawdown, win_rate, avg_rr, sharpe, account_balance, strategy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        summary["date"], summary.get("total_trades", 0),
        summary.get("winning_trades", 0), summary.get("losing_trades", 0),
        summary.get("gross_pnl", 0), summary.get("net_pnl", 0),
        summary.get("total_commission", 0), summary.get("max_drawdown", 0),
        summary.get("win_rate", 0), summary.get("avg_rr", 0),
        summary.get("sharpe", 0), summary.get("account_balance", 0),
        summary.get("strategy", ""),
    ))
    conn.commit()
    conn.close()


def get_performance_history(days: int = 90) -> pd.DataFrame:
    """Get daily summary history."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?",
        conn, params=(days,)
    )
    conn.close()
    return df


def save_setting(key: str, value: str):
    """Save a key-value setting."""
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
        (key, value, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_setting(key: str, default: str = "") -> str:
    """Get a setting value."""
    conn = get_connection()
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default
