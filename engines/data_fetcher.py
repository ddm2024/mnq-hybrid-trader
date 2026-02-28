"""
Data Fetching Layer
Provides market data for MNQ/NQ futures using yfinance (primary) with fallback generation.
Handles 5-minute intraday data and historical daily data for backtesting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


def fetch_mnq_data(symbol: str = "NQ=F", period: str = "5d", 
                   interval: str = "5m") -> pd.DataFrame:
    """
    Fetch MNQ/NQ futures data via yfinance.
    
    Note: yfinance uses NQ=F for E-mini Nasdaq futures. 
    MNQ (Micro) tracks the same index — price action is identical, 
    just different contract multiplier.
    
    For 5m data, yfinance supports up to 60 days of history.
    """
    try:
        import yfinance as yf
        
        # Map common symbols to yfinance tickers
        symbol_map = {
            "MNQ": "NQ=F",
            "NQ": "NQ=F",
            "MES": "ES=F",
            "ES": "ES=F",
            "MYM": "YM=F",
            "YM": "YM=F",
            "MNQ1!": "NQ=F",
            "NQ1!": "NQ=F",
        }
        
        yf_symbol = symbol_map.get(symbol.upper(), symbol)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            return _generate_synthetic_data(interval=interval, periods=500)
        
        # Standardize columns
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure we have required columns
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                if col == "volume":
                    df[col] = 1000  # Default volume
                else:
                    return _generate_synthetic_data(interval=interval, periods=500)
        
        df = df[required].copy()
        df.dropna(inplace=True)
        
        return df
        
    except Exception as e:
        print(f"yfinance fetch failed ({e}), using synthetic data")
        return _generate_synthetic_data(interval=interval, periods=500)


def fetch_historical_daily(symbol: str = "NQ=F", period: str = "1y") -> pd.DataFrame:
    """Fetch daily historical data for backtesting."""
    try:
        import yfinance as yf
        
        symbol_map = {"MNQ": "NQ=F", "NQ": "NQ=F", "MES": "ES=F", "ES": "ES=F"}
        yf_symbol = symbol_map.get(symbol.upper(), symbol)
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval="1d")
        
        if df.empty:
            return _generate_synthetic_data(interval="1d", periods=252)
        
        df.columns = [c.lower() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        df = df[[c for c in required if c in df.columns]].copy()
        if "volume" not in df.columns:
            df["volume"] = np.random.randint(50000, 200000, len(df))
        df.dropna(inplace=True)
        
        return df
        
    except Exception:
        return _generate_synthetic_data(interval="1d", periods=252)


def _generate_synthetic_data(interval: str = "5m", periods: int = 500, 
                             base_price: float = 21500.0) -> pd.DataFrame:
    """
    Generate realistic synthetic MNQ price data for demo/testing.
    Uses geometric Brownian motion with mean-reverting volatility.
    """
    np.random.seed(42)
    
    # Parameters calibrated to MNQ 5m behavior
    if interval == "5m":
        dt = 5 / (60 * 24 * 252)  # 5 min as fraction of year
        vol = 0.20  # ~20% annualized vol
        freq = "5min"
    elif interval == "1h":
        dt = 1 / (24 * 252)
        vol = 0.18
        freq = "1h"
    else:  # daily
        dt = 1 / 252
        vol = 0.22
        freq = "1D"
    
    # Generate returns with slight mean reversion
    drift = 0.0001
    returns = np.random.normal(drift * dt, vol * np.sqrt(dt), periods)
    
    # Add some autocorrelation (trending behavior)
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    # Build price series
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = []
    for i, close in enumerate(prices):
        noise = close * 0.001  # 0.1% noise for OHLC spread
        open_p = close + np.random.normal(0, noise)
        high_p = max(open_p, close) + abs(np.random.normal(0, noise * 2))
        low_p = min(open_p, close) - abs(np.random.normal(0, noise * 2))
        volume = int(np.random.lognormal(8, 1))  # Lognormal volume
        
        data.append({
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close, 2),
            "volume": volume,
        })
    
    # Create datetime index
    end = datetime.now()
    if interval in ("5m", "5min"):
        dates = pd.date_range(end=end, periods=periods, freq="5min")
    elif interval == "1h":
        dates = pd.date_range(end=end, periods=periods, freq="1h")
    else:
        dates = pd.date_range(end=end, periods=periods, freq="1D")
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = "datetime"
    
    return df


def resample_data(df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
    """Resample OHLCV data to a different timeframe."""
    interval_map = {
        "1m": "1min", "5m": "5min", "15m": "15min", 
        "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D",
    }
    
    freq = interval_map.get(target_interval, target_interval)
    
    resampled = df.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    
    return resampled
