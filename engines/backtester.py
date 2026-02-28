"""
Backtesting Engine
Walk-forward backtesting with realistic simulation (slippage, commissions).
Designed to avoid overfitting via out-of-sample validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from strategies.strategy_engine import (
    compute_indicators, compute_opening_range,
    generate_signal_hybrid1, generate_signal_hybrid2, Signal
)
from engines.risk_manager import RiskManager
from utils.config import StrategyConfig, MNQ_POINT_VALUE
from utils.helpers import (
    calculate_sharpe_ratio, calculate_max_drawdown,
    calculate_win_rate, calculate_avg_rr, calculate_profit_factor
)


class BacktestResult:
    """Holds backtesting results and performance metrics."""
    
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        self.metrics: Dict = {}
        self.signals_generated: int = 0
        self.signals_filtered: int = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame(self.trades)
    
    def summary(self) -> str:
        m = self.metrics
        lines = [
            "═" * 50,
            "  BACKTEST RESULTS",
            "═" * 50,
            f"  Period:          {m.get('start_date', 'N/A')} → {m.get('end_date', 'N/A')}",
            f"  Total Trades:    {m.get('total_trades', 0)}",
            f"  Win Rate:        {m.get('win_rate', 0):.1%}",
            f"  Avg R:R:         {m.get('avg_rr', 0):.2f}",
            f"  Profit Factor:   {m.get('profit_factor', 0):.2f}",
            f"  Net P&L:         ${m.get('net_pnl', 0):,.2f}",
            f"  Max Drawdown:    ${m.get('max_drawdown', 0):,.2f} ({m.get('max_drawdown_pct', 0):.1%})",
            f"  Sharpe Ratio:    {m.get('sharpe', 0):.2f}",
            f"  Total Commission:${m.get('total_commission', 0):,.2f}",
            f"  Signals Gen:     {self.signals_generated}",
            f"  Signals Filtered:{self.signals_filtered}",
            "═" * 50,
        ]
        return "\n".join(lines)


def run_backtest(df: pd.DataFrame, config: StrategyConfig, 
                 walk_forward: bool = True,
                 in_sample_pct: float = 0.7) -> BacktestResult:
    """
    Run a full backtest on historical data.
    
    If walk_forward=True, splits data into in-sample (training) and 
    out-of-sample (validation) periods to reduce overfitting risk.
    
    Args:
        df: OHLCV DataFrame
        config: Strategy configuration
        walk_forward: Enable walk-forward validation
        in_sample_pct: Fraction of data for in-sample (0.7 = 70%)
    
    Returns:
        BacktestResult with trades, equity curve, and metrics
    """
    result = BacktestResult()
    
    if len(df) < config.trend_ema_period + 20:
        result.metrics = {"error": "Insufficient data for backtest"}
        return result
    
    # Walk-forward: only test on out-of-sample data
    if walk_forward:
        split_idx = int(len(df) * in_sample_pct)
        # Compute indicators on full dataset (indicators need history)
        df_full = compute_indicators(df, config)
        # But only generate signals on out-of-sample portion
        test_start = split_idx
    else:
        df_full = compute_indicators(df, config)
        test_start = config.trend_ema_period + 5  # Skip warmup period
    
    # Initialize
    risk_mgr = RiskManager(config)
    account = config.account_size
    equity = [account]
    trades = []
    current_trade = None
    trade_id_counter = 0
    
    # Compute ORB if needed (simplified for backtest: use first 3 bars of each "session")
    orb_high, orb_low = None, None
    
    for i in range(test_start, len(df_full)):
        row = df_full.iloc[i]
        
        # Reset ORB at "session start" (simplified: every 78 bars ≈ 6.5 hours of 5m)
        if i % 78 == 0:
            orb_slice = df_full.iloc[i:i+3]
            if len(orb_slice) >= 2:
                orb_high = orb_slice["high"].max()
                orb_low = orb_slice["low"].min()
            risk_mgr.reset_daily()
        
        # ─── Check exits for open position ───
        if current_trade is not None:
            exit_check = risk_mgr.check_exit_conditions(
                current_trade["id"], row["high"], row["low"], row["close"]
            )
            
            if exit_check:
                exit_price, exit_reason = exit_check
                result_info = risk_mgr.process_exit(current_trade["id"], exit_price, exit_reason)
                
                account += result_info["net_pnl"]
                
                trade_record = {
                    **current_trade,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "pnl": result_info["net_pnl"],
                    "gross_pnl": result_info["gross_pnl"],
                    "commission": result_info["commission"],
                    "exit_bar": i,
                    "bars_held": i - current_trade["entry_bar"],
                }
                trades.append(trade_record)
                current_trade = None
        
        # ─── Check for new signals (only if no position) ───
        if current_trade is None:
            can_trade, reason = risk_mgr.can_trade()
            
            if can_trade and i > test_start + 3:  # Skip ORB period
                # Generate signal based on strategy mode
                if config.strategy_mode == "hybrid1":
                    signal = generate_signal_hybrid1(df_full, config, idx=i)
                elif config.strategy_mode == "hybrid2":
                    signal = generate_signal_hybrid2(df_full, config, orb_high, orb_low, idx=i)
                else:
                    sig1 = generate_signal_hybrid1(df_full, config, idx=i)
                    sig2 = generate_signal_hybrid2(df_full, config, orb_high, orb_low, idx=i)
                    signal = sig1 if sig1 and (not sig2 or sig1.confidence >= sig2.confidence) else sig2
                
                if signal and signal.signal != Signal.FLAT:
                    result.signals_generated += 1
                    
                    # Position sizing
                    stop_dist = abs(signal.entry_price - signal.stop_loss)
                    qty = risk_mgr.calculate_position_size(stop_dist)
                    
                    if qty > 0 and signal.confidence >= 0.5:
                        # Apply slippage to entry
                        slippage = signal.entry_price * config.slippage_pct
                        if signal.signal == Signal.LONG:
                            adj_entry = signal.entry_price + slippage
                        else:
                            adj_entry = signal.entry_price - slippage
                        
                        trade_id_counter += 1
                        
                        risk_mgr.process_entry(
                            trade_id_counter, signal.signal.value,
                            adj_entry, signal.stop_loss, signal.take_profit, qty
                        )
                        
                        current_trade = {
                            "id": trade_id_counter,
                            "direction": signal.signal.value,
                            "entry_price": round(adj_entry, 2),
                            "stop_loss": signal.stop_loss,
                            "take_profit": signal.take_profit,
                            "quantity": qty,
                            "strategy": signal.strategy,
                            "reason": signal.reason,
                            "confidence": signal.confidence,
                            "entry_bar": i,
                        }
                    else:
                        result.signals_filtered += 1
        
        equity.append(account)
    
    # ─── Close any remaining position at last price ───
    if current_trade is not None:
        last_price = df_full.iloc[-1]["close"]
        result_info = risk_mgr.process_exit(current_trade["id"], last_price, "Session End")
        account += result_info["net_pnl"]
        trade_record = {
            **current_trade,
            "exit_price": last_price,
            "exit_reason": "Session End",
            "pnl": result_info["net_pnl"],
            "gross_pnl": result_info["gross_pnl"],
            "commission": result_info["commission"],
            "exit_bar": len(df_full) - 1,
            "bars_held": len(df_full) - 1 - current_trade["entry_bar"],
        }
        trades.append(trade_record)
        equity.append(account)
    
    # ─── Compute Metrics ───
    result.trades = trades
    result.equity_curve = equity
    
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    equity_series = pd.Series(equity)
    
    if len(trades_df) > 0:
        returns = equity_series.pct_change().dropna()
        dd_amt, dd_pct = calculate_max_drawdown(equity_series)
        total_commission = trades_df["commission"].sum() if "commission" in trades_df.columns else 0
        
        result.metrics = {
            "start_date": str(df.index[test_start]) if hasattr(df.index[test_start], 'strftime') else str(test_start),
            "end_date": str(df.index[-1]) if hasattr(df.index[-1], 'strftime') else str(len(df)),
            "total_trades": len(trades_df),
            "winning_trades": int((trades_df["pnl"] > 0).sum()),
            "losing_trades": int((trades_df["pnl"] <= 0).sum()),
            "win_rate": calculate_win_rate(trades_df),
            "avg_rr": calculate_avg_rr(trades_df),
            "profit_factor": calculate_profit_factor(trades_df),
            "net_pnl": round(account - config.account_size, 2),
            "gross_pnl": round(trades_df["gross_pnl"].sum(), 2) if "gross_pnl" in trades_df.columns else 0,
            "total_commission": round(total_commission, 2),
            "max_drawdown": round(dd_amt, 2),
            "max_drawdown_pct": round(dd_pct, 4),
            "sharpe": calculate_sharpe_ratio(returns) if len(returns) > 10 else 0.0,
            "final_balance": round(account, 2),
            "return_pct": round((account - config.account_size) / config.account_size, 4),
            "avg_trade_pnl": round(trades_df["pnl"].mean(), 2),
            "best_trade": round(trades_df["pnl"].max(), 2),
            "worst_trade": round(trades_df["pnl"].min(), 2),
            "avg_bars_held": round(trades_df["bars_held"].mean(), 1) if "bars_held" in trades_df.columns else 0,
            "walk_forward": walk_forward,
            "in_sample_pct": in_sample_pct if walk_forward else 1.0,
        }
    else:
        result.metrics = {
            "total_trades": 0,
            "note": "No trades generated. Try adjusting parameters or using more data.",
            "walk_forward": walk_forward,
        }
    
    return result
