"""
Risk Management Engine
Combines QuantVue trailing stop logic + Vector Algorithmics strict sizing & daily limits.

Key features:
- Position sizing: account × risk% / stop distance (Vector)
- Trailing stops that lock in profits (QuantVue)
- Max daily loss auto-shutdown
- Per-session trade limits
- Slippage & commission modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from utils.config import StrategyConfig, MNQ_POINT_VALUE


class RiskManager:
    """
    Centralized risk management for all trading operations.
    Enforces position limits, daily loss limits, and trailing stop logic.
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.daily_pnl: float = 0.0
        self.trades_today: int = 0
        self.is_shutdown: bool = False
        self.shutdown_reason: str = ""
        self.open_positions: Dict[int, Dict] = {}  # trade_id -> position info
        self.trade_log: List[Dict] = []
    
    def reset_daily(self):
        """Reset daily counters (call at session start)."""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.is_shutdown = False
        self.shutdown_reason = ""
    
    # ─── Pre-Trade Checks ───
    
    def can_trade(self) -> Tuple[bool, str]:
        """
        Check all pre-trade risk conditions. Returns (allowed, reason).
        """
        if self.is_shutdown:
            return False, f"Trading halted: {self.shutdown_reason}"
        
        # Max daily loss check
        max_loss = self.config.account_size * self.config.max_daily_loss_pct
        if self.daily_pnl <= -max_loss:
            self.is_shutdown = True
            self.shutdown_reason = f"Max daily loss reached (${max_loss:.2f})"
            return False, self.shutdown_reason
        
        # Max trades per session
        if self.trades_today >= self.config.max_trades_per_session:
            return False, f"Max trades per session reached ({self.config.max_trades_per_session})"
        
        # Check if any position is already open (one-at-a-time for safety)
        if len(self.open_positions) > 0:
            return False, "Position already open — wait for exit"
        
        return True, "OK"
    
    # ─── Position Sizing (Vector Algorithmics) ───
    
    def calculate_position_size(self, stop_distance_points: float) -> int:
        """
        Vector-style position sizing:
        Contracts = (Account × Risk%) / (Stop Distance × Point Value)
        """
        if stop_distance_points <= 0:
            return 0
        
        risk_amount = self.config.account_size * self.config.risk_per_trade_pct
        cost_per_contract = stop_distance_points * MNQ_POINT_VALUE
        
        contracts = int(risk_amount / cost_per_contract)
        
        # Minimum 1 contract if risk allows at least half a contract
        if contracts == 0 and (risk_amount / cost_per_contract) >= 0.5:
            contracts = 1
        
        # Safety cap: never more than account can handle
        max_by_margin = int(self.config.account_size / max(self.config.account_size * 0.01, 1))
        contracts = min(contracts, max(1, max_by_margin))
        
        return max(0, contracts)
    
    def calculate_risk_amount(self, stop_distance_points: float, quantity: int) -> float:
        """Calculate dollar risk for a given position."""
        return stop_distance_points * MNQ_POINT_VALUE * quantity
    
    # ─── Entry Processing ───
    
    def process_entry(self, trade_id: int, direction: str, entry_price: float,
                      stop_loss: float, take_profit: float, quantity: int):
        """Register a new position with risk manager."""
        self.open_positions[trade_id] = {
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "quantity": quantity,
            "highest_price": entry_price,  # For trailing (longs)
            "lowest_price": entry_price,   # For trailing (shorts)
            "entry_time": datetime.now().isoformat(),
        }
        self.trades_today += 1
    
    # ─── Exit Processing ───
    
    def process_exit(self, trade_id: int, exit_price: float, reason: str) -> Dict:
        """
        Process a position exit. Returns P&L details.
        """
        if trade_id not in self.open_positions:
            return {"pnl": 0, "commission": 0, "slippage": 0, "net_pnl": 0}
        
        pos = self.open_positions[trade_id]
        direction = pos["direction"]
        entry = pos["entry_price"]
        qty = pos["quantity"]
        
        # Apply slippage
        slippage_amount = exit_price * self.config.slippage_pct
        if direction == "LONG":
            adj_exit = exit_price - slippage_amount
            pnl_points = adj_exit - entry
        else:
            adj_exit = exit_price + slippage_amount
            pnl_points = entry - adj_exit
        
        # P&L in dollars
        gross_pnl = pnl_points * MNQ_POINT_VALUE * qty
        
        # Commission (per side × 2 for round trip)
        commission = self.config.commission_per_contract * qty * 2
        slippage_cost = slippage_amount * MNQ_POINT_VALUE * qty
        
        net_pnl = gross_pnl - commission
        
        # Update daily P&L
        self.daily_pnl += net_pnl
        
        # Check daily loss limit after this trade
        max_loss = self.config.account_size * self.config.max_daily_loss_pct
        if self.daily_pnl <= -max_loss:
            self.is_shutdown = True
            self.shutdown_reason = f"Max daily loss reached after trade (${abs(self.daily_pnl):.2f})"
        
        result = {
            "trade_id": trade_id,
            "direction": direction,
            "entry_price": entry,
            "exit_price": exit_price,
            "adj_exit_price": adj_exit,
            "pnl_points": round(pnl_points, 2),
            "gross_pnl": round(gross_pnl, 2),
            "commission": round(commission, 2),
            "slippage_cost": round(slippage_cost, 2),
            "net_pnl": round(net_pnl, 2),
            "quantity": qty,
            "exit_reason": reason,
        }
        
        self.trade_log.append(result)
        del self.open_positions[trade_id]
        
        return result
    
    # ─── Trailing Stop (QuantVue-inspired) ───
    
    def update_trailing_stop(self, trade_id: int, current_price: float) -> Optional[float]:
        """
        Update trailing stop for an open position.
        Returns new stop level if adjusted, None otherwise.
        
        QuantVue trailing logic: once in profit, trail by a percentage of
        the accumulated profit to lock in gains.
        """
        if trade_id not in self.open_positions:
            return None
        
        pos = self.open_positions[trade_id]
        direction = pos["direction"]
        entry = pos["entry_price"]
        current_stop = pos["stop_loss"]
        
        if direction == "LONG":
            # Track highest price
            if current_price > pos["highest_price"]:
                pos["highest_price"] = current_price
            
            # Only trail if in profit
            profit = current_price - entry
            if profit <= 0:
                return None
            
            # Trail: stop = highest_price - (trailing_pct × profit from entry)
            trail_amount = profit * self.config.trailing_stop_pct
            new_stop = pos["highest_price"] - trail_amount
            
            # Only move stop UP, never down
            if new_stop > current_stop:
                pos["stop_loss"] = round(new_stop, 2)
                return pos["stop_loss"]
        
        else:  # SHORT
            if current_price < pos["lowest_price"]:
                pos["lowest_price"] = current_price
            
            profit = entry - current_price
            if profit <= 0:
                return None
            
            trail_amount = profit * self.config.trailing_stop_pct
            new_stop = pos["lowest_price"] + trail_amount
            
            # Only move stop DOWN, never up
            if new_stop < current_stop:
                pos["stop_loss"] = round(new_stop, 2)
                return pos["stop_loss"]
        
        return None
    
    # ─── Price Check for Stops/Targets ───
    
    def check_exit_conditions(self, trade_id: int, current_high: float, 
                              current_low: float, current_close: float) -> Optional[Tuple[float, str]]:
        """
        Check if stop loss or take profit has been hit.
        Returns (exit_price, reason) or None.
        """
        if trade_id not in self.open_positions:
            return None
        
        pos = self.open_positions[trade_id]
        direction = pos["direction"]
        stop = pos["stop_loss"]
        target = pos["take_profit"]
        
        if direction == "LONG":
            # Stop loss hit (check low)
            if current_low <= stop:
                return (stop, "Stop Loss")
            # Take profit hit (check high)
            if current_high >= target:
                return (target, "Take Profit")
        
        else:  # SHORT
            # Stop loss hit (check high)
            if current_high >= stop:
                return (stop, "Stop Loss")
            # Take profit hit (check low)
            if current_low <= target:
                return (target, "Take Profit")
        
        # Update trailing stop
        self.update_trailing_stop(trade_id, current_close)
        
        return None
    
    # ─── Status & Reporting ───
    
    def get_status(self) -> Dict:
        """Get current risk manager status."""
        max_loss = self.config.account_size * self.config.max_daily_loss_pct
        remaining_loss_budget = max_loss + self.daily_pnl  # daily_pnl is negative when losing
        
        return {
            "daily_pnl": round(self.daily_pnl, 2),
            "trades_today": self.trades_today,
            "max_trades": self.config.max_trades_per_session,
            "trades_remaining": max(0, self.config.max_trades_per_session - self.trades_today),
            "is_shutdown": self.is_shutdown,
            "shutdown_reason": self.shutdown_reason,
            "max_daily_loss": round(max_loss, 2),
            "remaining_loss_budget": round(max(0, remaining_loss_budget), 2),
            "open_positions": len(self.open_positions),
            "loss_budget_pct": round(remaining_loss_budget / max_loss * 100, 1) if max_loss > 0 else 0,
        }
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get all processed trades as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_log)
