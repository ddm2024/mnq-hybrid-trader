"""
Utility helpers — shared functions used across the application.
"""

import hashlib
import hmac
import os
import pytz
from datetime import datetime, time, timedelta
from typing import Optional, Tuple
import numpy as np
import pandas as pd


# ─── Time Utilities ───

CT = pytz.timezone("America/Chicago")
ET = pytz.timezone("America/New_York")


def now_ct() -> datetime:
    """Current time in Central Time."""
    return datetime.now(CT)


def now_et() -> datetime:
    """Current time in Eastern Time."""
    return datetime.now(ET)


def is_within_rth(dt: Optional[datetime] = None, 
                  start: str = "08:30", end: str = "15:00") -> bool:
    """Check if a datetime falls within Regular Trading Hours (CT)."""
    if dt is None:
        dt = now_ct()
    elif dt.tzinfo is None:
        dt = CT.localize(dt)
    else:
        dt = dt.astimezone(CT)
    
    start_h, start_m = map(int, start.split(":"))
    end_h, end_m = map(int, end.split(":"))
    
    start_time = time(start_h, start_m)
    end_time = time(end_h, end_m)
    current_time = dt.time()
    
    # Also check it's a weekday
    if dt.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    
    return start_time <= current_time <= end_time


def is_trading_day(dt: Optional[datetime] = None) -> bool:
    """Check if today is a weekday (basic check; doesn't account for holidays)."""
    if dt is None:
        dt = now_ct()
    return dt.weekday() < 5


def get_session_date() -> str:
    """Get today's session date string."""
    return now_ct().strftime("%Y-%m-%d")


# ─── Math & Stats Utilities ───

def calculate_position_size(account_size: float, risk_pct: float, 
                            stop_distance_points: float,
                            point_value: float = 2.0) -> int:
    """
    Vector Algorithmics-style position sizing.
    Position = (Account × Risk%) / (Stop Distance × Point Value)
    
    Returns number of contracts (minimum 1 if risk allows, 0 if not).
    """
    if stop_distance_points <= 0 or account_size <= 0:
        return 0
    
    risk_amount = account_size * risk_pct
    cost_per_contract = stop_distance_points * point_value
    
    contracts = int(risk_amount / cost_per_contract)
    return max(0, contracts)


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05,
                           periods_per_year: int = 252) -> float:
    """
    Annualized Sharpe ratio.
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    return float(np.sqrt(periods_per_year) * excess_returns.mean() / returns.std())


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
    """
    Calculate maximum drawdown from an equity curve.
    Returns (max_drawdown_amount, max_drawdown_pct).
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0
    
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = equity_curve - peak
    max_dd = drawdown.min()
    
    # Percentage
    dd_pct = (drawdown / peak).min()
    
    return float(abs(max_dd)), float(abs(dd_pct))


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """Calculate win rate from trades DataFrame."""
    if len(trades) == 0:
        return 0.0
    winners = (trades["pnl"] > 0).sum()
    return float(winners / len(trades))


def calculate_avg_rr(trades: pd.DataFrame) -> float:
    """Calculate average reward-to-risk ratio from trades."""
    if len(trades) == 0:
        return 0.0
    
    winners = trades[trades["pnl"] > 0]["pnl"]
    losers = trades[trades["pnl"] < 0]["pnl"]
    
    if len(losers) == 0 or losers.mean() == 0:
        return float(winners.mean()) if len(winners) > 0 else 0.0
    
    avg_win = winners.mean() if len(winners) > 0 else 0.0
    avg_loss = abs(losers.mean())
    
    return float(avg_win / avg_loss) if avg_loss > 0 else 0.0


def calculate_profit_factor(trades: pd.DataFrame) -> float:
    """Profit factor = gross profits / gross losses."""
    if len(trades) == 0:
        return 0.0
    
    gross_profit = trades[trades["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades[trades["pnl"] < 0]["pnl"].sum())
    
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


# ─── Formatting Utilities ───

def fmt_currency(value: float) -> str:
    """Format as currency."""
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    """Format as percentage."""
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a number with commas."""
    return f"{value:,.{decimals}f}"


# ─── Security Utilities ───

def hash_password(password: str) -> str:
    """Simple password hashing for demo auth."""
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    import bcrypt
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ─── Data Validation ───

def validate_config_ranges(config) -> list:
    """Validate strategy config values are within safe ranges."""
    warnings = []
    
    if config.risk_per_trade_pct > 0.02:
        warnings.append("⚠️ Risk per trade above 2% is extremely aggressive for futures.")
    
    if config.risk_per_trade_pct > 0.01:
        warnings.append("⚡ Risk per trade above 1% is above recommended levels. Consider 0.25-0.5%.")
    
    if config.max_daily_loss_pct > 0.05:
        warnings.append("⚠️ Max daily loss above 5% risks blowing the account. Recommended: 2%.")
    
    if config.reward_risk_ratio < 1.0:
        warnings.append("⚡ R:R below 1.0 means you need >50% win rate to be profitable. Aim for 1.5+.")
    
    if config.max_trades_per_session > 15:
        warnings.append("⚠️ More than 15 trades/session suggests overtrading. Quality > quantity.")
    
    if config.stop_loss_points < 10:
        warnings.append("⚡ Stop loss under 10 points is very tight for MNQ. Risk frequent stop-outs.")
    
    if config.stop_loss_points > 50:
        warnings.append("⚠️ Stop loss over 50 points means large dollar risk per contract.")
    
    return warnings
