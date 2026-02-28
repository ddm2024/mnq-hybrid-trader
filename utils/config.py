"""
MNQ Hybrid Algo Trader — Core Configuration & Constants
Blends QuantVue momentum automation + Vector Algorithmics market-neutral logic.
Primary focus: 5-minute MNQ (Micro E-mini Nasdaq-100) futures.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─── Risk Disclaimer ───
RISK_DISCLAIMER = """
⚠️ **IMPORTANT RISK WARNING** ⚠️

Trading futures involves substantial risk of loss and is not suitable for all investors.
Past performance is not indicative of future results. You could lose more than your
initial investment. This software is provided for EDUCATIONAL and SIMULATION purposes.

• Always start with PAPER/DEMO trading before risking real capital.
• Never trade with money you cannot afford to lose.
• This is NOT financial advice. Consult a licensed financial advisor.
• The developers assume no liability for trading losses.

By using this application, you acknowledge these risks.
"""

# ─── Session & Trading Hours (Central Time) ───
RTH_START_CT = "08:30"  # Regular Trading Hours start (CT) = 9:30 ET
RTH_END_CT = "15:00"    # Regular Trading Hours end (CT) = 16:00 ET
TIMEZONE = "America/Chicago"

# ─── MNQ Contract Specifications ───
MNQ_TICK_SIZE = 0.25       # Minimum price increment
MNQ_TICK_VALUE = 0.50      # Dollar value per tick
MNQ_POINT_VALUE = 2.00     # Dollar value per point (1 point = 4 ticks)
MNQ_MARGIN_INTRADAY = 50   # Approximate intraday margin per contract (varies by broker)

# ─── Default Strategy Parameters ───
@dataclass
class StrategyConfig:
    """Configuration for the hybrid MNQ strategy."""
    # Account
    account_size: float = float(os.getenv("DEFAULT_ACCOUNT_SIZE", 25000))
    risk_per_trade_pct: float = float(os.getenv("DEFAULT_RISK_PCT", 0.0025))  # 0.25%
    max_daily_loss_pct: float = float(os.getenv("DEFAULT_MAX_DAILY_LOSS_PCT", 0.02))  # 2%

    # Instrument
    symbol: str = os.getenv("DEFAULT_SYMBOL", "MNQ")
    timeframe: str = os.getenv("DEFAULT_TIMEFRAME", "5m")
    
    # Trend Filter (QuantVue-inspired)
    trend_ema_period: int = 50          # 50 EMA for trend bias
    fast_ema_period: int = 9            # Fast EMA for entry signal
    slow_ema_period: int = 21           # Slow EMA for entry signal
    
    # Volume Filter (QuantVue)
    volume_sma_period: int = 20         # Volume SMA for confirmation
    volume_multiplier: float = 1.2      # Volume must be 1.2x average
    
    # ATR / Volatility (Vector-inspired)
    atr_period: int = 14                # ATR lookback
    atr_stop_multiplier: float = 1.5    # Stop = ATR × multiplier
    atr_breakout_multiplier: float = 1.0  # Breakout confirmation threshold
    
    # Entry & Exit
    stop_loss_points: float = 25.0      # Hard stop in MNQ points (fallback)
    reward_risk_ratio: float = 1.75     # Take profit = stop × R:R
    trailing_stop_pct: float = 0.5      # Trailing stop as % of profit (QuantVue)
    use_atr_stops: bool = True          # Use ATR-based stops (Vector) vs fixed
    
    # Session & Limits
    max_trades_per_session: int = 6     # Max trades per day
    session_start: str = RTH_START_CT   # Only trade during RTH
    session_end: str = RTH_END_CT
    
    # ORB (Opening Range Breakout) — Hybrid 2
    orb_period_minutes: int = 15        # First 15 min for opening range
    orb_atr_filter: bool = True         # Require ATR expansion for ORB
    
    # Strategy Bias
    strategy_mode: str = "hybrid1"      # "hybrid1" (Momentum-Vol) or "hybrid2" (ORB/Pullback)
    trend_vs_scalp_bias: float = 0.7    # 0=pure scalp, 1=pure trend-follow (0.7 = trend-leaning)
    
    # Execution
    slippage_pct: float = 0.001         # 0.1% simulated slippage
    commission_per_contract: float = 0.62  # Typical MNQ commission per side
    
    # Paper Mode
    paper_mode: bool = True


@dataclass 
class AppConfig:
    """Application-level configuration."""
    secret_key: str = os.getenv("APP_SECRET_KEY", "dev-secret-change-me")
    demo_mode: bool = os.getenv("DEMO_MODE", "true").lower() == "true"
    db_path: str = "data/trades.db"
    log_level: str = "INFO"
    
    # Alpaca
    alpaca_api_key: str = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret_key: str = os.getenv("ALPACA_SECRET_KEY", "")
    alpaca_base_url: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Twilio (optional alerts)
    twilio_sid: str = os.getenv("TWILIO_SID", "")
    twilio_auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    twilio_from: str = os.getenv("TWILIO_FROM", "")
    twilio_to: str = os.getenv("TWILIO_TO", "")


# ─── Strategy Descriptions (for UI) ───
STRATEGY_INFO = {
    "hybrid1": {
        "name": "Momentum-Volatility Fusion",
        "short": "QuantVue EMA crossover + Vector ATR breakout",
        "description": (
            "Combines QuantVue-style momentum entries (9/21 EMA crossover with volume "
            "confirmation) and a 50 EMA trend filter, with Vector Algorithmics' ATR-based "
            "breakout confirmation and strict position sizing. Designed for trending 5m MNQ "
            "sessions with volatility expansion."
        ),
        "best_for": "Trending markets with clear directional momentum",
        "risk_profile": "Moderate — 0.25-0.5% per trade, ATR-adjusted stops",
    },
    "hybrid2": {
        "name": "5m MNQ ORB/Pullback",
        "short": "Opening range breakout + EMA pullback in trend",
        "description": (
            "Trades the opening range breakout (first 15 minutes of RTH) or pullback entries "
            "to fast EMA in the direction of the 50 EMA trend. Hard stops at 20-30 MNQ points "
            "with 1.5-2R targets. Strictly RTH only with max trades per session cap. "
            "Blends QuantVue's disciplined entries with Vector's session-only risk control."
        ),
        "best_for": "High-volume opens and clear intraday trends",
        "risk_profile": "Conservative — fixed stops, session-only, max daily loss cutoff",
    },
}

# ─── Tooltips & Help Text ───
TOOLTIPS = {
    "ema_crossover": "When the fast EMA (9) crosses above the slow EMA (21), it signals bullish momentum. Below = bearish.",
    "trend_filter": "The 50 EMA acts as a directional bias filter. Only take longs above it, shorts below it.",
    "atr": "Average True Range measures volatility. Higher ATR = wider stops and targets to account for bigger swings.",
    "volume_filter": "Confirms entries only when volume exceeds its 20-period average × 1.2. Avoids low-conviction moves.",
    "orb": "Opening Range Breakout: defines a high/low range from the first 15 min of RTH, then trades the breakout.",
    "risk_per_trade": "How much of your account you risk per trade. 0.25% = conservative, 0.5% = moderate. Never exceed 1%.",
    "max_daily_loss": "Auto-stops trading if daily losses hit this threshold. Standard: 2%. Protects against tilt/bad sessions.",
    "reward_risk": "Target profit as multiple of risk. 1.5R means if you risk $100, target is $150. Aim for 1.5-2.0R.",
    "trailing_stop": "Locks in profits as price moves favorably. QuantVue-style: trails by a % of accumulated profit.",
    "paper_mode": "Simulated trading with no real money at risk. ALWAYS start here. Test for at least 30 days.",
    "slippage": "Real-world cost: your fill price is slightly worse than expected. We simulate 0.1% to be realistic.",
    "sharpe_ratio": "Risk-adjusted return. Above 1.0 = acceptable, above 2.0 = very good. Below 0.5 = concerning.",
    "max_drawdown": "Largest peak-to-trough decline. Keep under 10% for futures. Above 15% = strategy needs review.",
    "position_sizing": "Vector-style: Position = (Account × Risk%) / Stop Distance. Never risk more than the math allows.",
}
