"""
MNQ Hybrid Algo Trader — Main Streamlit Dashboard
==================================================
Dark-themed financial trading dashboard for the MNQ Hybrid Algo Trader.

Architecture (TradingView → Tradovate flow):
  1. This app generates Pine Script strategies (via pine_generator)
  2. User pastes Pine Script into TradingView Pine Editor
  3. TradingView fires webhook alerts on signal / order fill
  4. Webhook receiver forwards JSON payload to Tradovate for execution

Pages (sidebar navigation):
  1. 📊 Dashboard      — market status, live chart, risk status, positions
  2. ⚙️  Strategy Config — all strategy parameters with tooltips & validation
  3. 🔗 TV Integration  — Pine Script generator, webhook setup, alert guide
  4. 🧪 Backtesting     — walk-forward backtest with equity curve & metrics
  5. 📋 Trade History   — database trade log, daily P&L chart, CSV export
  6. 🛠️  Settings        — paper mode, broker connection, database management

Run with:
    streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────────
# Standard library & third-party imports
# ─────────────────────────────────────────────────────────────────────────────────
import sys
import os
import io
import json
import time
import datetime
import warnings
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────────
# Local module imports — ensure we resolve from the project root
# ─────────────────────────────────────────────────────────────────────────────────
# Add project root to sys.path so all modules resolve correctly
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Config & Constants ───────────────────────────────────────────────────────────────────────
from utils.config import (
    StrategyConfig, AppConfig,
    RISK_DISCLAIMER, STRATEGY_INFO, TOOLTIPS,
)

# ── Helpers ───────────────────────────────────────────────────────────────────────────────────
from utils.helpers import (
    now_ct, is_within_rth,
    calculate_position_size, fmt_currency, fmt_pct,
    validate_config_ranges, calculate_sharpe_ratio,
    calculate_max_drawdown, calculate_win_rate, calculate_avg_rr,
)

# ── Database ──────────────────────────────────────────────────────────────────────────────────
from utils.database import (
    log_trade, close_trade,
    get_open_trades, get_trades_today, get_all_trades,
    get_daily_pnl, save_daily_summary, get_performance_history,
)

# ── Strategy Engine ───────────────────────────────────────────────────────────────────────────
try:
    from strategies.strategy_engine import (
        compute_indicators, compute_opening_range,
        run_strategy, Signal, TradeSignal,
    )
    _STRATEGY_ENGINE_OK = True
except Exception as _e:
    _STRATEGY_ENGINE_OK = False
    _STRATEGY_ENGINE_ERR = str(_e)

# ── Risk Manager ──────────────────────────────────────────────────────────────────────────────
from engines.risk_manager import RiskManager

# ── Data Fetcher ──────────────────────────────────────────────────────────────────────────────
from engines.data_fetcher import fetch_mnq_data, fetch_historical_daily

# ── Backtester ──────────────────────────────────────────────────────────────────────────────────
from engines.backtester import run_backtest, BacktestResult

# ── Pine Script Generator ───────────────────────────────────────────────────────────────────────────
from engines.pine_generator import (
    generate_pine_script,
    generate_webhook_json_template,
    generate_alert_setup_instructions,
)

# ── Tradovate Client ────────────────────────────────────────────────────────────────────────────
try:
    from engines.tradovate_client import (
        TradovateClientSync, TradovateWebhookProcessor,
        create_client_from_env,
    )
    _TRADOVATE_OK = True
except Exception as _te:
    _TRADOVATE_OK = False
    _TRADOVATE_ERR = str(_te)


# ─────────────────────────────────────────────────────────────────────────────────
# Streamlit page config — must be the first st call
# ─────────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNQ Hybrid Algo Trader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "MNQ Hybrid Algo Trader v1.0\n\n"
            "Combines QuantVue momentum automation with Vector Algorithmics "
            "market-neutral risk control for 5-minute MNQ futures.\n\n"
            "⚠️ For educational purposes only. Always start with paper trading."
        ),
    },
)

# ─────────────────────────────────────────────────────────────────────────────────
# Custom CSS — polish the dark theme, fix spacing quirks
# ─────────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Metric delta colours ── */
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Status pill helpers ── */
.status-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}
.pill-green  { background: #166534; color: #86efac; }
.pill-red    { background: #7f1d1d; color: #fca5a5; }
.pill-yellow { background: #713f12; color: #fde68a; }
.pill-blue   { background: #1e3a5f; color: #93c5fd; }

/* ── Info card ── */
.info-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.9rem;
}

/* ── Flow step box ── */
.flow-box {
    background: #0f2744;
    border: 1px solid #2563EB;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 4px 0;
    font-size: 0.88rem;
    color: #93c5fd;
}

/* ── Risk warning box ── */
.risk-warning {
    background: #450a0a;
    border: 1px solid #dc2626;
    border-left: 4px solid #dc2626;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    color: #fca5a5;
}

/* ── Paper mode warning box ── */
.paper-mode-box {
    background: #1a2e05;
    border: 1px solid #4ade80;
    border-left: 4px solid #4ade80;
    border-radius: 6px;
    padding: 8px 14px;
    color: #86efac;
    font-size: 0.88rem;
}

/* ── Live mode warning box ── */
.live-mode-box {
    background: #450a0a;
    border: 1px solid #ef4444;
    border-left: 4px solid #ef4444;
    border-radius: 6px;
    padding: 8px 14px;
    color: #fca5a5;
    font-size: 0.88rem;
}

/* ── Sidebar spacing ── */
.stSidebarNav { padding-top: 0.5rem; }

/* ── Hide the default plotly toolbar clutter ── */
.modebar-container { opacity: 0.4; }
.modebar-container:hover { opacity: 1; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────────
# Session-state initialisation helpers
# All persistent state lives in st.session_state to survive re-runs.
# ─────────────────────────────────────────────────────────────────────────────────

def _init_session_state():
    """Initialise all session-state keys with sensible defaults on first run."""

    # ── Strategy config (shared across all pages) ─────────────────────────────────
    if "config" not in st.session_state:
        st.session_state.config = StrategyConfig()

    # ── App config ───────────────────────────────────────────────────────────────────────
    if "app_config" not in st.session_state:
        st.session_state.app_config = AppConfig()

    # ── Risk manager (re-created whenever config changes) ─────────────────────────
    if "risk_manager" not in st.session_state:
        st.session_state.risk_manager = RiskManager(st.session_state.config)

    # ── Cached market data ───────────────────────────────────────────────────────────
    if "market_data" not in st.session_state:
        st.session_state.market_data = None
    if "market_data_ts" not in st.session_state:
        st.session_state.market_data_ts = None  # timestamp of last fetch

    # ── Backtest results ───────────────────────────────────────────────────────────
    if "backtest_result" not in st.session_state:
        st.session_state.backtest_result = None

    # ── Tradovate connection state ────────────────────────────────────────────────
    if "tradovate_connected" not in st.session_state:
        st.session_state.tradovate_connected = False
    if "tradovate_credentials" not in st.session_state:
        st.session_state.tradovate_credentials = {
            "username": "", "password": "", "app_id": "",
            "cid": "", "device_id": "", "secret": "",
            "demo_mode": True,
        }

    # ── Webhook log (in-memory last N lines) ─────────────────────────────────────────
    if "webhook_log" not in st.session_state:
        st.session_state.webhook_log = []

    # ── Risk disclaimer acknowledgement ──────────────────────────────────────────
    if "disclaimer_shown" not in st.session_state:
        st.session_state.disclaimer_shown = False

    # ── Paper/live mode (mirrors config.paper_mode for easy sidebar access) ───
    if "paper_mode" not in st.session_state:
        st.session_state.paper_mode = st.session_state.config.paper_mode


_init_session_state()


# ─────────────────────────────────────────────────────────────────────────────────
# Shared Plotly theme
# ─────────────────────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    template="plotly_dark",
    paper_bgcolor="#0F172A",
    plot_bgcolor="#0F172A",
    font_color="#E2E8F0",
)


# ─────────────────────────────────────────────────────────────────────────────────
# Utility functions used across pages
# ─────────────────────────────────────────────────────────────────────────────────

def _get_market_data(force_refresh: bool = False) -> pd.DataFrame:
    """
    Return cached 5-minute MNQ data, refreshing if older than 5 minutes
    or if force_refresh is True.  Falls back to synthetic data on failure.
    """
    now = time.time()
    stale = (
        st.session_state.market_data is None
        or st.session_state.market_data_ts is None
        or (now - st.session_state.market_data_ts) > 300  # 5-min cache
        or force_refresh
    )

    if stale:
        with st.spinner("Fetching MNQ market data…"):
            try:
                df = fetch_mnq_data(period="5d", interval="5m")
                st.session_state.market_data = df
                st.session_state.market_data_ts = now
            except Exception as e:
                st.warning(f"Data fetch failed ({e}). Using synthetic data.")
                # Fallback — synthetic data already returned by fetcher
                st.session_state.market_data = fetch_mnq_data(period="5d", interval="5m")
                st.session_state.market_data_ts = now

    return st.session_state.market_data


def _build_candlestick_chart(df: pd.DataFrame, config: StrategyConfig) -> go.Figure:
    """
    Build a Plotly candlestick chart with EMA 9/21/50 overlays,
    ATR bands, and a volume subplot. Uses the dark theme.
    """
    if df is None or len(df) < 20:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font_size=16,
                           font_color="#94a3b8")
        fig.update_layout(**PLOTLY_DARK, height=400)
        return fig

    # ── Compute indicators ────────────────────────────────────────────────────────────────────
    close = df["close"]
    ema9  = close.ewm(span=config.fast_ema_period,  adjust=False).mean()
    ema21 = close.ewm(span=config.slow_ema_period,  adjust=False).mean()
    ema50 = close.ewm(span=config.trend_ema_period, adjust=False).mean()

    # ATR (simplified Wilder's)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"]  - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=config.atr_period, adjust=False).mean()
    atr_upper = close + atr * config.atr_stop_multiplier
    atr_lower = close - atr * config.atr_stop_multiplier

    # Show last 100 bars to keep chart readable
    tail = 100
    idx = df.index[-tail:]

    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.78, 0.22],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # ── Candlesticks ───────────────────────────────────────────────────────────────────
    fig.add_trace(go.Candlestick(
        x=idx,
        open=df["open"].iloc[-tail:],
        high=df["high"].iloc[-tail:],
        low=df["low"].iloc[-tail:],
        close=close.iloc[-tail:],
        name="MNQ",
        increasing_line_color="#4ade80",
        decreasing_line_color="#f87171",
        increasing_fillcolor="#166534",
        decreasing_fillcolor="#7f1d1d",
        line_width=1,
        showlegend=False,
    ), row=1, col=1)

    # ── EMAs ──────────────────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=idx, y=ema9.iloc[-tail:], name=f"EMA {config.fast_ema_period}",
        line=dict(color="#facc15", width=1), opacity=0.9,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=ema21.iloc[-tail:], name=f"EMA {config.slow_ema_period}",
        line=dict(color="#fb923c", width=1.5), opacity=0.9,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=ema50.iloc[-tail:], name=f"EMA {config.trend_ema_period}",
        line=dict(color="#60a5fa", width=2), opacity=0.9,
    ), row=1, col=1)

    # ── ATR bands (filled) ────────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=list(idx) + list(idx[::-1]),
        y=list(atr_upper.iloc[-tail:]) + list(atr_lower.iloc[-tail:][::-1]),
        fill="toself", fillcolor="rgba(96,165,250,0.06)",
        line=dict(color="rgba(0,0,0,0)"), name="ATR Band",
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=atr_upper.iloc[-tail:], name="ATR Upper",
        line=dict(color="#f87171", width=1, dash="dot"), opacity=0.5,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=atr_lower.iloc[-tail:], name="ATR Lower",
        line=dict(color="#4ade80", width=1, dash="dot"), opacity=0.5,
    ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────────────────────
    vol = df["volume"].iloc[-tail:]
    vol_colors = [
        "#166534" if df["close"].iloc[-tail + i] >= df["open"].iloc[-tail + i]
        else "#7f1d1d"
        for i in range(len(vol))
    ]
    fig.add_trace(go.Bar(
        x=idx, y=vol, name="Volume",
        marker_color=vol_colors, opacity=0.6, showlegend=False,
    ), row=2, col=1)

    # ── Layout ──────────────────────────────────────────────────────────────────────────────
    fig.update_layout(
        **PLOTLY_DARK,
        height=480,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
            bgcolor="rgba(15,23,42,0.7)", font_size=11,
        ),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor="#1E293B")
    fig.update_yaxes(title_text="Vol",   row=2, col=1, gridcolor="#1E293B")
    fig.update_xaxes(gridcolor="#1E293B")

    return fig


def _pnl_color(value: float) -> str:
    """Return green/red CSS color string based on P&L sign."""
    return "#4ade80" if value >= 0 else "#f87171"


# ─────────────────────────────────────────────────────────────────────────────────
# Sidebar — navigation & global status indicators
# ─────────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    """Render the sidebar: logo, mode badge, nav, and global status strip."""
    with st.sidebar:
        # ── Logo / title ──────────────────────────────────────────────────────────────────
        st.markdown("## 📈 MNQ Hybrid Trader")
        st.markdown("<div style='color:#64748b;font-size:0.78rem;margin-top:-8px;'>Micro E-mini Nasdaq-100 Algo</div>", unsafe_allow_html=True)
        st.markdown("---")

        # ── Paper / Live badge ───────────────────────────────────────────────────────────────
        if st.session_state.config.paper_mode:
            st.markdown('<span class="status-pill pill-green">📄 PAPER MODE</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill pill-red">🔴 LIVE MODE</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Navigation ────────────────────────────────────────────────────────────────────────
        page = st.radio(
            "Navigation",
            options=[
                "📊 Dashboard",
                "⚙️ Strategy Config",
                "🔗 TV Integration",
                "🧪 Backtesting",
                "📋 Trade History",
                "🛠️ Settings",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")

        # ── Quick status strip ───────────────────────────────────────────────────────────────
        now = now_ct()
        rth_open = is_within_rth()
        st.markdown(
            f"**Time (CT):** {now.strftime('%H:%M:%S')}<br>"
            f"**RTH:** {'🟢 Open' if rth_open else '🔴 Closed'}<br>"
            f"**Strategy:** {STRATEGY_INFO[st.session_state.config.strategy_mode]['name'][:18]}…",
            unsafe_allow_html=True,
        )

        # Daily P&L from database
        try:
            daily_pnl = get_daily_pnl()
            pnl_color = _pnl_color(daily_pnl)
            st.markdown(
                f"**Daily P&L:** <span style='color:{pnl_color}'>{fmt_currency(daily_pnl)}</span>",
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        st.markdown("---")
        st.markdown(
            "<div style='color:#475569;font-size:0.72rem;'>v1.0 · Educational use only<br>Not financial advice</div>",
            unsafe_allow_html=True,
        )

    return page


# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Dashboard (Home)
# ─────────────────────────────────────────────────────────────────────────────────

def page_dashboard():
    """
    The main landing page. Shows market status, candlestick chart,
    risk manager status, open positions, and quick performance metrics.
    """
    config = st.session_state.config
    risk_mgr = st.session_state.risk_manager

    # ── Risk Disclaimer (shown first-visit, collapsible) ────────────────────────
    if not st.session_state.disclaimer_shown:
        with st.expander("⚠️  RISK WARNING — Read Before Trading (click to dismiss)", expanded=True):
            st.warning(RISK_DISCLAIMER)
            if st.button("I understand the risks — dismiss this warning"):
                st.session_state.disclaimer_shown = True
                st.rerun()
    else:
        # Compact reminder — always one click away
        with st.expander("⚠️ Risk Warning"):
            st.warning(RISK_DISCLAIMER)

    # ── Header row: market status + strategy card ────────────────────────────────────
    col_mkt, col_strat = st.columns([1, 2])

    with col_mkt:
        st.subheader("📡 Market Status")
        now = now_ct()
        rth = is_within_rth()

        status_html = (
            '<span class="status-pill pill-green">🟢 RTH OPEN</span>'
            if rth else
            '<span class="status-pill pill-red">🔴 RTH CLOSED</span>'
        )
        st.markdown(status_html, unsafe_allow_html=True)
        st.markdown(f"**{now.strftime('%A, %B %-d · %H:%M:%S CT')}**")
        st.markdown(
            f"Regular Trading Hours: **08:30 – 15:00 CT**<br>"
            f"*(9:30 – 16:00 ET · CME Globex)*",
            unsafe_allow_html=True,
        )
        if not rth:
            st.info("Market is outside RTH. Strategy only trades 08:30–15:00 CT.")

    with col_strat:
        st.subheader("📋 Active Strategy")
        mode = config.strategy_mode
        info = STRATEGY_INFO[mode]
        badge_color = "#1d4ed8" if mode == "hybrid1" else "#7c3aed"
        st.markdown(
            f'<div class="info-card">'
            f'<b style="color:{badge_color};">{info["name"]}</b><br>'
            f'<span style="color:#94a3b8;font-size:0.85rem;">{info["short"]}</span><br><br>'
            f'{info["description"][:220]}…<br><br>'
            f'<b>Best for:</b> {info["best_for"]}<br>'
            f'<b>Risk profile:</b> {info["risk_profile"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Live chart ──────────────────────────────────────────────────────────────────────────
    st.subheader("📊 MNQ 5-Minute Chart")
    chart_col, refresh_col = st.columns([5, 1])

    with refresh_col:
        if st.button("🔄 Refresh", use_container_width=True):
            _get_market_data(force_refresh=True)
            st.rerun()
        if st.session_state.market_data_ts:
            age = int(time.time() - st.session_state.market_data_ts)
            st.caption(f"Data age: {age}s")

    df = _get_market_data()

    with chart_col:
        if df is not None and len(df) > 0:
            current_price = df["close"].iloc[-1]
            prev_price    = df["close"].iloc[-2] if len(df) > 1 else current_price
            delta         = current_price - prev_price
            delta_pct     = delta / prev_price * 100 if prev_price else 0

            price_col, delta_col, atr_col = st.columns(3)
            with price_col:
                st.metric(
                    "Last Price",
                    f"{current_price:,.2f}",
                    f"{delta:+.2f} ({delta_pct:+.2f}%)",
                )
            with delta_col:
                # Compute ATR on recent data
                tr = pd.concat([
                    df["high"] - df["low"],
                    (df["high"] - df["close"].shift()).abs(),
                    (df["low"]  - df["close"].shift()).abs(),
                ], axis=1).max(axis=1)
                atr_val = tr.ewm(span=config.atr_period, adjust=False).mean().iloc[-1]
                st.metric("ATR (5m)", f"{atr_val:.1f} pts", help=TOOLTIPS["atr"])
            with atr_col:
                volume_now = df["volume"].iloc[-1]
                vol_sma = df["volume"].rolling(config.volume_sma_period).mean().iloc[-1]
                vol_ratio = volume_now / vol_sma if vol_sma else 1
                vol_label = f"{vol_ratio:.2f}× avg"
                st.metric("Volume Ratio", vol_label,
                          delta="✓ Elevated" if vol_ratio > config.volume_multiplier else "Low",
                          help=TOOLTIPS["volume_filter"])

    fig = _build_candlestick_chart(df, config)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Risk manager status ───────────────────────────────────────────────────────────────
    st.subheader("🛡️ Risk Manager Status")
    status = risk_mgr.get_status()

    if status["is_shutdown"]:
        st.markdown(
            f'<div class="live-mode-box">🚨 <b>TRADING HALTED</b> — {status["shutdown_reason"]}</div>',
            unsafe_allow_html=True,
        )
    elif status["daily_pnl"] < -status["max_daily_loss"] * 0.7:
        st.markdown(
            '<div class="risk-warning">⚡ Approaching max daily loss threshold. Trade with caution.</div>',
            unsafe_allow_html=True,
        )

    r1, r2, r3, r4, r5 = st.columns(5)

    with r1:
        pnl = status["daily_pnl"]
        st.metric("Daily P&L", fmt_currency(pnl),
                  delta=f"{pnl/config.account_size*100:+.2f}% of account")

    with r2:
        st.metric(
            "Trades Today",
            f"{status['trades_today']} / {status['max_trades']}",
            help="Number of trades taken today vs. session maximum.",
        )

    with r3:
        budget = status["remaining_loss_budget"]
        st.metric(
            "Loss Budget Left",
            fmt_currency(budget),
            help="Daily loss budget remaining before auto-shutdown.",
        )

    with r4:
        open_pos = status["open_positions"]
        st.metric("Open Positions", open_pos)

    with r5:
        budget_pct = status["loss_budget_pct"]
        st.metric(
            "Budget Used",
            f"{100 - budget_pct:.0f}%",
            delta=f"{budget_pct:.0f}% remaining",
        )

    # Loss budget progress bar
    budget_pct_val = max(0, min(100, status["loss_budget_pct"]))
    bar_color = "#4ade80" if budget_pct_val > 50 else ("#fb923c" if budget_pct_val > 20 else "#ef4444")
    st.markdown(f"""
    <div style="background:#1E293B;border-radius:6px;height:14px;overflow:hidden;margin:4px 0 12px 0;">
        <div style="width:{budget_pct_val:.1f}%;background:{bar_color};height:100%;transition:width 0.3s ease;border-radius:6px;"></div>
    </div>
    <span style="font-size:0.78rem;color:#94a3b8;">
        Loss budget: {fmt_currency(budget)} remaining of {fmt_currency(status['max_daily_loss'])} max daily loss
    </span>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Open positions table ───────────────────────────────────────────────────────────────
    st.subheader("🔓 Open Positions")
    try:
        open_trades = get_open_trades()
        if open_trades:
            df_open = pd.DataFrame(open_trades)
            display_cols = ["id", "direction", "entry_price", "stop_loss",
                            "take_profit", "quantity", "strategy", "timestamp"]
            display_cols = [c for c in display_cols if c in df_open.columns]
            st.dataframe(df_open[display_cols], use_container_width=True, hide_index=True)
        else:
            st.info("No open positions. Market orders from TradingView webhooks will appear here.")
    except Exception as e:
        st.warning(f"Could not load open positions: {e}")

    st.markdown("---")

    # ── Quick performance metrics ──────────────────────────────────────────────────────────
    st.subheader("📈 Session Performance")

    try:
        all_trades_df = get_all_trades(limit=200)
        trades_today  = get_trades_today()
        today_df      = pd.DataFrame(trades_today) if trades_today else pd.DataFrame()

        if len(all_trades_df) == 0:
            st.info(
                "No closed trades yet. Run a backtest or connect TradingView webhooks "
                "to start seeing performance data. Try the **🧪 Backtesting** page first!"
            )
        else:
            m1, m2, m3, m4 = st.columns(4)

            with m1:
                wr = calculate_win_rate(all_trades_df) if len(all_trades_df) > 0 else 0
                wr_delta = "🟢 Good" if wr >= 0.5 else "🔴 Below 50%"
                st.metric("Win Rate", fmt_pct(wr), wr_delta,
                          help=TOOLTIPS["sharpe_ratio"])

            with m2:
                avg_rr = calculate_avg_rr(all_trades_df) if len(all_trades_df) > 0 else 0
                st.metric("Avg R:R", f"{avg_rr:.2f}",
                          "✓ Above 1.5" if avg_rr >= 1.5 else "Below target",
                          help=TOOLTIPS["reward_risk"])

            with m3:
                if len(all_trades_df) > 10 and "pnl" in all_trades_df.columns:
                    equity = config.account_size + all_trades_df["pnl"].cumsum()
                    returns = equity.pct_change().dropna()
                    sharpe = calculate_sharpe_ratio(returns)
                else:
                    sharpe = 0.0
                st.metric("Sharpe Ratio", f"{sharpe:.2f}",
                          "🟢 > 1.0" if sharpe >= 1.0 else "< 1.0",
                          help=TOOLTIPS["sharpe_ratio"])

            with m4:
                if len(all_trades_df) > 1 and "pnl" in all_trades_df.columns:
                    equity = config.account_size + all_trades_df["pnl"].cumsum()
                    _, dd_pct = calculate_max_drawdown(equity)
                    st.metric("Max Drawdown", fmt_pct(dd_pct),
                              "⚠️ High" if dd_pct > 0.1 else "OK",
                              help=TOOLTIPS["max_drawdown"])
                else:
                    st.metric("Max Drawdown", "—")

    except Exception as e:
        st.warning(f"Could not compute performance metrics: {e}")


# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 2 — Strategy Configuration
# ─────────────────────────────────────────────────────────────────────────────────

def page_strategy_config():
    """
    Full strategy parameter editor. All parameters use st.slider() / st.number_input()
    with tooltips from the TOOLTIPS dict. Config is validated in real-time
    and saved to session_state on button press.
    """
    st.title("⚙️ Strategy Configuration")
    st.markdown(
        "Adjust all parameters below. Changes only take effect after clicking "
        "**Save Configuration**. Warnings are shown when values exceed safe ranges."
    )

    config = st.session_state.config  # current saved config

    # ── Strategy selector ───────────────────────────────────────────────────────────────
    st.subheader("Strategy Selector")
    strategy_options = {
        "hybrid1": f"Hybrid 1 — {STRATEGY_INFO['hybrid1']['name']}",
        "hybrid2": f"Hybrid 2 — {STRATEGY_INFO['hybrid2']['name']}",
    }
    selected_mode = st.radio(
        "Choose Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda k: strategy_options[k],
        index=0 if config.strategy_mode == "hybrid1" else 1,
        horizontal=True,
    )

    # Show strategy description
    info = STRATEGY_INFO[selected_mode]
    st.markdown(
        f'<div class="info-card">'
        f'<b>{info["name"]}</b><br>'
        f'<span style="color:#94a3b8;">{info["description"]}</span><br><br>'
        f'<b>Best for:</b> {info["best_for"]}<br>'
        f'<b>Risk profile:</b> {info["risk_profile"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── All parameter inputs — stored in local variables, not session state yet ──
    # This way nothing is committed until "Save Configuration" is pressed.

    tab_risk, tab_ema, tab_atr, tab_entry, tab_session, tab_advanced = st.tabs([
        "💰 Account & Risk", "📉 EMA Settings", "📊 ATR & Volatility",
        "🎯 Entry & Exit", "🕐 Session & Limits", "🔧 Advanced",
    ])

    # ── TAB 1: Account & Risk ──────────────────────────────────────────────────────────────
    with tab_risk:
        st.markdown("### Account & Risk Parameters")
        st.caption("These are the most critical settings. Start conservative.")

        col1, col2 = st.columns(2)

        with col1:
            account_size = st.number_input(
                "Account Size ($)",
                min_value=1000.0,
                max_value=500_000.0,
                value=float(config.account_size),
                step=1000.0,
                help="Your total trading account balance in USD. Used for position sizing.",
            )
            risk_per_trade_pct_ui = st.slider(
                "Risk Per Trade (%)",
                min_value=0.1,
                max_value=2.0,
                value=float(config.risk_per_trade_pct * 100),
                step=0.05,
                format="%.2f%%",
                help=TOOLTIPS["risk_per_trade"],
            )
            risk_per_trade_pct = risk_per_trade_pct_ui / 100.0

        with col2:
            max_daily_loss_pct_ui = st.slider(
                "Max Daily Loss (%)",
                min_value=0.5,
                max_value=5.0,
                value=float(config.max_daily_loss_pct * 100),
                step=0.25,
                format="%.2f%%",
                help=TOOLTIPS["max_daily_loss"],
            )
            max_daily_loss_pct = max_daily_loss_pct_ui / 100.0

            # Show dollar values in context
            st.markdown(
                f'<div class="info-card">'
                f'Risk per trade: <b>${account_size * risk_per_trade_pct:,.2f}</b><br>'
                f'Max daily loss: <b>${account_size * max_daily_loss_pct:,.2f}</b>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── TAB 2: EMA Settings ────────────────────────────────────────────────────────────────
    with tab_ema:
        st.markdown("### EMA (Exponential Moving Average) Settings")
        st.caption(
            "The 9/21 EMA crossover generates entries. "
            "The 50 EMA acts as a trend filter — only longs above it, shorts below."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            fast_ema_period = st.slider(
                f"Fast EMA Period (default: {config.fast_ema_period})",
                min_value=5, max_value=20,
                value=config.fast_ema_period, step=1,
                help=TOOLTIPS["ema_crossover"],
            )
        with col2:
            slow_ema_period = st.slider(
                f"Slow EMA Period (default: {config.slow_ema_period})",
                min_value=15, max_value=50,
                value=config.slow_ema_period, step=1,
                help=TOOLTIPS["ema_crossover"],
            )
        with col3:
            trend_ema_period = st.slider(
                f"Trend EMA Period (default: {config.trend_ema_period})",
                min_value=20, max_value=200,
                value=config.trend_ema_period, step=5,
                help=TOOLTIPS["trend_filter"],
            )

        # Validation warning
        if fast_ema_period >= slow_ema_period:
            st.warning("⚡ Fast EMA period should be smaller than Slow EMA period.")

        volume_sma_period = st.slider(
            "Volume SMA Period",
            min_value=5, max_value=50,
            value=config.volume_sma_period, step=1,
            help=TOOLTIPS["volume_filter"],
        )
        volume_multiplier = st.slider(
            "Volume Multiplier (× SMA)",
            min_value=0.5, max_value=3.0,
            value=float(config.volume_multiplier), step=0.1,
            format="%.1f×",
            help=TOOLTIPS["volume_filter"],
        )

    # ── TAB 3: ATR & Volatility ──────────────────────────────────────────────────────────────
    with tab_atr:
        st.markdown("### ATR (Average True Range) & Volatility Settings")
        st.caption("ATR measures market volatility and dynamically adjusts stop distances and position sizing.")

        col1, col2 = st.columns(2)

        with col1:
            atr_period = st.slider(
                "ATR Period",
                min_value=5, max_value=30,
                value=config.atr_period, step=1,
                help=TOOLTIPS["atr"],
            )
            atr_stop_multiplier = st.slider(
                "ATR Stop Multiplier",
                min_value=0.5, max_value=3.0,
                value=float(config.atr_stop_multiplier), step=0.25,
                format="%.2f×",
                help="Stop loss = ATR × this multiplier. Higher = wider stops, fewer stop-outs.",
            )
        with col2:
            atr_breakout_multiplier = st.slider(
                "ATR Breakout Multiplier",
                min_value=0.25, max_value=3.0,
                value=float(config.atr_breakout_multiplier), step=0.25,
                format="%.2f×",
                help="Price must exceed ATR × this value to confirm a breakout entry.",
            )
            trend_vs_scalp_bias = st.slider(
                "Trend vs. Scalp Bias",
                min_value=0.0, max_value=1.0,
                value=float(config.trend_vs_scalp_bias), step=0.05,
                help="0 = pure scalp (tight targets, many trades). "
                     "1 = pure trend-follow (let winners run). 0.7 recommended.",
            )
        st.info(
            f"With ATR Period = {atr_period} and multiplier = {atr_stop_multiplier}×, "
            f"a typical MNQ ATR of ~20 pts would produce a ~{20*atr_stop_multiplier:.1f}-point stop "
            f"(≈ ${20*atr_stop_multiplier*2:.0f}/contract)."
        )

    # ── TAB 4: Entry & Exit ───────────────────────────────────────────────────────────────
    with tab_entry:
        st.markdown("### Entry & Exit Parameters")

        col1, col2 = st.columns(2)

        with col1:
            stop_loss_points = st.slider(
                "Stop Loss (MNQ points)",
                min_value=10, max_value=60,
                value=int(config.stop_loss_points), step=1,
                help="Hard stop in MNQ points. Used by Hybrid 2. "
                     "1 point = $2/contract. 25 points = $50/contract.",
            )
            reward_risk_ratio = st.slider(
                "Reward:Risk Ratio",
                min_value=1.0, max_value=4.0,
                value=float(config.reward_risk_ratio), step=0.25,
                format="%.2f R",
                help=TOOLTIPS["reward_risk"],
            )
        with col2:
            trailing_stop_pct = st.slider(
                "Trailing Stop (%)",
                min_value=0.1, max_value=1.0,
                value=float(config.trailing_stop_pct), step=0.1,
                format="%.1f%%",
                help=TOOLTIPS["trailing_stop"],
            )
            use_atr_stops = st.toggle(
                "Use ATR-Based Stops",
                value=config.use_atr_stops,
                help="If ON, uses ATR × multiplier for stop distance (Hybrid 1). "
                     "If OFF, uses fixed Stop Loss Points (Hybrid 2).",
            )

        # Show calculated target in dollars
        stop_dollar = stop_loss_points * 2  # $2/point for MNQ
        target_dollar = stop_dollar * reward_risk_ratio
        st.markdown(
            f'<div class="info-card">'
            f'Stop: <b>{stop_loss_points} pts = ${stop_dollar:.0f}/contract</b> &nbsp;|&nbsp; '
            f'Target: <b>{stop_loss_points * reward_risk_ratio:.1f} pts = ${target_dollar:.0f}/contract</b> '
            f'({reward_risk_ratio:.2f}R)'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── TAB 5: Session & Limits ──────────────────────────────────────────────────────────────
    with tab_session:
        st.markdown("### Session & Trade Limits")
        st.caption("These settings protect against overtrading and overnight risk.")

        col1, col2 = st.columns(2)

        with col1:
            max_trades_per_session = st.slider(
                "Max Trades Per Session",
                min_value=1, max_value=20,
                value=config.max_trades_per_session, step=1,
                help="Hard cap on entries per RTH session. 4–6 is typical for disciplined trading.",
            )
            orb_period_minutes = st.slider(
                "ORB Period (minutes)",
                min_value=5, max_value=30,
                value=config.orb_period_minutes, step=5,
                help=TOOLTIPS["orb"],
            ) if selected_mode == "hybrid2" else config.orb_period_minutes

        with col2:
            orb_atr_filter = st.toggle(
                "Require ATR Expansion for ORB",
                value=config.orb_atr_filter,
                help="Only take ORB breakouts when ATR is expanding. Filters low-volatility fakeouts.",
            ) if selected_mode == "hybrid2" else config.orb_atr_filter

        st.info(
            "The strategy is RTH-only (08:30–15:00 CT). "
            "All positions are auto-closed at session end by the Pine Script to prevent overnight holds."
        )

    # ── TAB 6: Advanced ──────────────────────────────────────────────────────────────────
    with tab_advanced:
        st.markdown("### Advanced / Execution Settings")
        st.caption("These rarely need adjustment. Modify only if you understand the impact.")

        col1, col2 = st.columns(2)

        with col1:
            slippage_pct_ui = st.number_input(
                "Slippage (%)",
                min_value=0.0,
                max_value=0.5,
                value=float(config.slippage_pct * 100),
                step=0.01,
                format="%.3f",
                help=TOOLTIPS["slippage"],
            )
            slippage_pct = slippage_pct_ui / 100.0

        with col2:
            commission_per_contract = st.number_input(
                "Commission Per Contract ($)",
                min_value=0.0,
                max_value=5.0,
                value=float(config.commission_per_contract),
                step=0.01,
                format="%.2f",
                help="Per-side commission in USD. Typical: $0.62 for MNQ at major brokers.",
            )

        paper_mode = st.toggle(
            "Paper Mode (Simulated Trading)",
            value=config.paper_mode,
            help=TOOLTIPS["paper_mode"],
        )
        if not paper_mode:
            st.markdown(
                '<div class="live-mode-box">🚨 <b>LIVE MODE WARNING:</b> Real money will be at risk. '
                'Only enable this after extensive paper trading. Never risk money you cannot afford to lose.</div>',
                unsafe_allow_html=True,
            )

    # ── Build tentative config from current UI state ────────────────────────────────
    # This is used for validation preview and position size calculator.
    tentative_config = StrategyConfig(
        account_size=account_size,
        risk_per_trade_pct=risk_per_trade_pct,
        max_daily_loss_pct=max_daily_loss_pct,
        fast_ema_period=fast_ema_period,
        slow_ema_period=slow_ema_period,
        trend_ema_period=trend_ema_period,
        volume_sma_period=volume_sma_period,
        volume_multiplier=volume_multiplier,
        atr_period=atr_period,
        atr_stop_multiplier=atr_stop_multiplier,
        atr_breakout_multiplier=atr_breakout_multiplier,
        stop_loss_points=stop_loss_points,
        reward_risk_ratio=reward_risk_ratio,
        trailing_stop_pct=trailing_stop_pct,
        use_atr_stops=use_atr_stops,
        max_trades_per_session=max_trades_per_session,
        orb_period_minutes=orb_period_minutes,
        orb_atr_filter=orb_atr_filter,
        strategy_mode=selected_mode,
        trend_vs_scalp_bias=trend_vs_scalp_bias,
        slippage_pct=slippage_pct,
        commission_per_contract=commission_per_contract,
        paper_mode=paper_mode,
    )

    st.markdown("---")

    # ── Validation warnings ───────────────────────────────────────────────────────────────
    warnings_list = validate_config_ranges(tentative_config)
    if warnings_list:
        st.subheader("⚠️ Configuration Warnings")
        for w in warnings_list:
            st.warning(w)
    else:
        st.success("✅ Configuration is within recommended safe ranges.")

    # ── Position size calculator preview ──────────────────────────────────────────────
    st.subheader("📐 Position Size Calculator Preview")
    st.caption("Based on your current settings, here's how many contracts you would trade per signal:")

    ps_col1, ps_col2, ps_col3 = st.columns(3)
    stop_dist = stop_loss_points  # use fixed stop for preview

    with ps_col1:
        contracts = calculate_position_size(account_size, risk_per_trade_pct, stop_dist)
        st.metric("Contracts (at fixed stop)", str(contracts),
                  help=TOOLTIPS["position_sizing"])

    with ps_col2:
        risk_amt = account_size * risk_per_trade_pct
        st.metric("Dollar Risk Per Trade", fmt_currency(risk_amt))

    with ps_col3:
        max_loss_amt = account_size * max_daily_loss_pct
        st.metric("Max Daily Loss Budget", fmt_currency(max_loss_amt))

    # ── Save button ──────────────────────────────────────────────────────────────────────
    st.markdown("---")
    save_col, _ = st.columns([1, 3])
    with save_col:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            st.session_state.config = tentative_config
            # Rebuild risk manager with new config
            st.session_state.risk_manager = RiskManager(tentative_config)
            st.session_state.paper_mode = tentative_config.paper_mode
            # Clear cached market data so chart refreshes with new EMA periods
            st.session_state.market_data = None
            st.success("✅ Configuration saved! Risk manager updated.")
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 3 — TradingView Integration
# ─────────────────────────────────────────────────────────────────────────────────

def page_tv_integration():
    """
    The key integration page. Step-by-step guide for the full
    TradingView → Tradovate execution pipeline.
    """
    config = st.session_state.config

    st.title("🔗 TradingView Integration")

    # ── Flow diagram ──────────────────────────────────────────────────────────────────────────
    st.markdown("### How the System Works")
    cols = st.columns(5)
    flow_steps = [
        ("1", "Generate Pine Script", "here in this app"),
        ("2", "Paste into TradingView", "Pine Editor tab"),
        ("3", "Create Alert + Webhook", "TradingView alert manager"),
        ("4", "Signal fires → JSON POST", "TradingView → your endpoint"),
        ("5", "Tradovate executes order", "demo or live account"),
    ]
    for col, (num, title, sub) in zip(cols, flow_steps):
        with col:
            st.markdown(
                f'<div class="flow-box"><b style="font-size:1.1rem;">Step {num}</b><br>'
                f'<b>{title}</b><br>'
                f'<span style="font-size:0.78rem;opacity:0.8;">{sub}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Step 1: Generate Pine Script ─────────────────────────────────────────────────────
    with st.expander("Step 1 — Generate Your Pine Script", expanded=True):
        st.markdown(
            "Click the button below to generate a complete, ready-to-use Pine Script v5 "
            "strategy for TradingView. The script embeds all your current configuration "
            "parameters and includes built-in Tradovate webhook alert messages."
        )

        mode_label = STRATEGY_INFO[config.strategy_mode]["name"]
        st.info(
            f"Generating for: **{mode_label}** with account size "
            f"**{fmt_currency(config.account_size)}**, "
            f"risk **{fmt_pct(config.risk_per_trade_pct)}** per trade. "
            f"Change these on the ⚙️ Strategy Config page."
        )

        if st.button("🖥️ Generate Pine Script", type="primary"):
            with st.spinner("Generating Pine Script…"):
                try:
                    pine_code = generate_pine_script(config)
                    st.session_state["_pine_code"] = pine_code
                except Exception as e:
                    st.error(f"Pine Script generation failed: {e}")
                    st.session_state["_pine_code"] = None

        pine_code = st.session_state.get("_pine_code")
        if pine_code:
            st.success(f"✅ Generated {len(pine_code):,} characters of Pine Script v5 code.")
            st.code(pine_code, language="javascript")
            # Download button
            st.download_button(
                label="⬇️ Download .pine file",
                data=pine_code.encode("utf-8"),
                file_name=f"mnq_{config.strategy_mode}_{datetime.date.today()}.pine",
                mime="text/plain",
            )
            st.markdown("""
            **Next steps:**
            1. Copy the code above (Ctrl+A → Ctrl+C inside the code block)
            2. Open TradingView → Pine Editor (bottom of chart)
            3. Clear the default code and paste yours
            4. Click **Save** then **Add to chart**
            """)

    # ── Step 2: Webhook JSON Template ────────────────────────────────────────────────────
    with st.expander("Step 2 — Webhook Alert Message Setup"):
        st.markdown(
            "These JSON templates are what TradingView sends to Tradovate when an "
            "alert fires. Paste the appropriate template into the **Message** field "
            "when creating your TradingView alert."
        )

        try:
            webhook_template = generate_webhook_json_template(config)
            st.code(webhook_template, language="json")
            st.download_button(
                label="⬇️ Download webhook templates",
                data=webhook_template.encode("utf-8"),
                file_name=f"webhook_templates_{config.strategy_mode}.txt",
                mime="text/plain",
            )
        except Exception as e:
            st.error(f"Could not generate webhook template: {e}")

        st.warning(
            "⚠️ Replace `REPLACE_WITH_TRADOVATE_ACCOUNT_ID` with your actual "
            "Tradovate account spec (e.g., `demo/12345` for paper, `live/67890` for live)."
        )

    # ── Step 3: Alert Configuration Instructions ───────────────────────────────────────
    with st.expander("Step 3 — Alert Configuration Guide"):
        try:
            instructions = generate_alert_setup_instructions()
            st.code(instructions, language="text")
        except Exception as e:
            st.error(f"Could not load instructions: {e}")

    # ── Step 4: Tradovate Connection ───────────────────────────────────────────────────────
    with st.expander("Step 4 — Tradovate API Connection"):
        st.markdown(
            "Enter your Tradovate API credentials to enable direct order execution. "
            "**Always start with Demo mode.** These are stored only in your session "
            "(not persisted to disk)."
        )

        creds = st.session_state.tradovate_credentials

        col1, col2 = st.columns(2)
        with col1:
            creds["username"]  = st.text_input("Tradovate Username", value=creds["username"])
            creds["app_id"]    = st.text_input("App ID", value=creds["app_id"],
                                               help="Found in Tradovate → Settings → API")
            creds["device_id"] = st.text_input("Device ID", value=creds["device_id"],
                                               help="Generate a UUID for your app instance")
        with col2:
            creds["password"]  = st.text_input("Password", value=creds["password"],
                                               type="password")
            creds["cid"]       = st.text_input("Client ID (CID)", value=creds["cid"])
            creds["secret"]    = st.text_input("Secret", value=creds["secret"],
                                               type="password",
                                               help="API secret from Tradovate developer portal")

        creds["demo_mode"] = st.toggle(
            "Demo Mode (recommended)",
            value=creds.get("demo_mode", True),
            help="Demo = no real money. Always test here first.",
        )

        if not creds["demo_mode"]:
            st.markdown(
                '<div class="live-mode-box">🚨 LIVE MODE: Real money at risk. '
                'Ensure you have tested extensively in demo first.</div>',
                unsafe_allow_html=True,
            )

        test_col, status_col = st.columns(2)
        with test_col:
            if st.button("🔌 Test Connection"):
                if not _TRADOVATE_OK:
                    st.error(f"Tradovate client unavailable: {_TRADOVATE_ERR}")
                elif not creds["username"] or not creds["password"]:
                    st.warning("Enter credentials first.")
                else:
                    with st.spinner("Testing Tradovate connection…"):
                        try:
                            client = create_client_from_env()
                            st.session_state.tradovate_connected = True
                            st.success("✅ Connected to Tradovate successfully.")
                        except Exception as e:
                            st.session_state.tradovate_connected = False
                            st.error(f"Connection failed: {e}")

        with status_col:
            if st.session_state.tradovate_connected:
                st.markdown('<span class="status-pill pill-green">✅ CONNECTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-pill pill-yellow">⚪ NOT CONNECTED</span>', unsafe_allow_html=True)

        st.markdown("""
        **Manual connection alternative:**  
        The Pine Script strategy embeds `alert_message` JSON directly in each 
        `strategy.entry()` and `strategy.exit()` call. TradingView will POST this 
        JSON to your Tradovate webhook URL automatically — no local server required.
        """)

    # ── Step 5: Webhook Receiver Status ───────────────────────────────────────────────
    with st.expander("Step 5 — Webhook Receiver Status"):
        st.markdown(
            "The webhook receiver forwards TradingView POST requests to Tradovate. "
            "You can run one locally or use a cloud provider (Railway, Render, etc.)."
        )

        st.info(
            "**Webhook endpoint format:**  \n"
            "`https://live.tradovateapi.com/webhook/YOUR_WEBHOOK_SECRET`\n\n"
            "Get your secret from Tradovate → Settings → Third Party Integrations → Webhooks."
        )

        # Recent webhook log (in-memory, populated if running the webhook receiver separately)
        st.markdown("**Recent Webhook Events (in-session)**")
        webhook_log = st.session_state.webhook_log
        if webhook_log:
            for entry in reversed(webhook_log[-10:]):
                ts = entry.get("ts", "")
                msg = entry.get("msg", "")
                ok  = entry.get("ok", True)
                color = "#4ade80" if ok else "#f87171"
                st.markdown(
                    f'<span style="color:#94a3b8;font-size:0.8rem;">{ts}</span> '
                    f'<span style="color:{color};">{msg}</span>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption(
                "No webhook events received this session. "
                "Events will appear here once TradingView alerts start firing."
            )

        # Simulate a test webhook entry for demo purposes
        if st.button("➕ Add test webhook log entry (demo)"):
            st.session_state.webhook_log.append({
                "ts": now_ct().strftime("%H:%M:%S"),
                "msg": 'DEMO: {"action":"buy","symbol":"MNQH5","qty":"1","strategy":"hybrid1"}',
                "ok": True,
            })
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Backtesting
# ─────────────────────────────────────────────────────────────────────────────────

def page_backtesting():
    """
    Walk-forward backtest runner.
    Fetches historical data (or uses synthetic), runs run_backtest(),
    then displays equity curve, trade list, and performance metrics.
    """
    config = st.session_state.config

    st.title("🧪 Backtesting")
    st.markdown(
        "Run a historical simulation of your strategy on MNQ data. "
        "Walk-forward validation splits data into in-sample (training) "
        "and out-of-sample (test) periods to reduce overfitting."
    )

    st.markdown('<div class="risk-warning">⚠️ Backtest results are <b>not</b> a guarantee of future performance. '
                'Positive backtests can still fail in live markets due to overfitting, regime changes, and execution slippage.</div>',
                unsafe_allow_html=True)

    st.markdown("---")

    # ── Configuration panel ───────────────────────────────────────────────────────────────
    cfg_col, run_col = st.columns([3, 1])

    with cfg_col:
        bc1, bc2, bc3 = st.columns(3)

        with bc1:
            data_source = st.selectbox(
                "Data Source",
                options=["Live (yfinance NQ=F)", "Synthetic (GBM simulation)"],
                help="'Live' fetches real historical NQ data. 'Synthetic' uses modelled data (always available).",
            )
        with bc2:
            period_map = {
                "5 days":   "5d",
                "1 month":  "1mo",
                "3 months": "3mo",
                "6 months": "6mo",
                "1 year":   "1y",
            }
            period_label = st.selectbox(
                "Backtest Period",
                options=list(period_map.keys()),
                index=2,
            )
            period = period_map[period_label]

        with bc3:
            data_interval = st.selectbox(
                "Bar Interval",
                options=["5m", "15m", "1h"],
                index=0,
            )

        wf_col, is_col = st.columns(2)
        with wf_col:
            walk_forward = st.toggle(
                "Walk-Forward Validation",
                value=True,
                help="Splits data into in-sample (training) and out-of-sample (test) portions to reduce overfitting.",
            )
        with is_col:
            if walk_forward:
                in_sample_pct = st.slider(
                    "In-Sample %",
                    min_value=50, max_value=90, value=70, step=5,
                    help="Percentage of data used for indicator calibration. "
                         "The remaining % is the true out-of-sample test.",
                ) / 100.0
                st.caption(f"Out-of-sample test: {(1-in_sample_pct)*100:.0f}% of data")
            else:
                in_sample_pct = 1.0

    with run_col:
        st.markdown("<br>", unsafe_allow_html=True)
        run_bt = st.button("▶️ Run Backtest", type="primary", use_container_width=True)

    # ── Run backtest ──────────────────────────────────────────────────────────────────────
    if run_bt:
        with st.spinner(f"Fetching {period} of {data_interval} data and running backtest…"):
            try:
                # Fetch data
                if "Synthetic" in data_source:
                    from engines.data_fetcher import _generate_synthetic_data
                    df = _generate_synthetic_data(interval=data_interval, periods=800)
                else:
                    if data_interval == "5m" and period not in ("5d", "1mo"):
                        st.warning("yfinance only supports up to 60 days for 5m data. Switching to synthetic.")
                        from engines.data_fetcher import _generate_synthetic_data
                        df = _generate_synthetic_data(interval=data_interval, periods=800)
                    else:
                        df = fetch_mnq_data(period=period, interval=data_interval)

                if df is None or len(df) < 60:
                    st.error("Not enough data to run backtest. Try a longer period or synthetic data.")
                else:
                    result = run_backtest(df, config, walk_forward=walk_forward, in_sample_pct=in_sample_pct)
                    st.session_state.backtest_result = result
                    st.success(f"✅ Backtest complete — {len(result.trades)} trades on {len(df)} bars.")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                st.code(traceback.format_exc())

    # ── Results ─────────────────────────────────────────────────────────────────────────────
    result: Optional[BacktestResult] = st.session_state.backtest_result

    if result is None:
        st.info(
            "No backtest results yet. Configure your strategy on the ⚙️ page, "
            "then click **Run Backtest** above."
        )
        return

    m = result.metrics

    if "error" in m:
        st.error(f"Backtest error: {m['error']}")
        return

    if m.get("total_trades", 0) == 0:
        st.warning(
            "No trades were generated. This usually means the strategy conditions "
            "were too strict for the selected data. Try:\n"
            "- A longer period\n- Loosening volume multiplier\n- Using synthetic data"
        )
        if "note" in m:
            st.info(m["note"])
        return

    # ── Metrics row ───────────────────────────────────────────────────────────────────────────
    st.markdown("### Results Summary")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    metrics = [
        ("Total Trades",    m.get("total_trades", 0),    None),
        ("Win Rate",        fmt_pct(m.get("win_rate", 0)), None),
        ("Avg R:R",         f"{m.get('avg_rr', 0):.2f}",  None),
        ("Profit Factor",   f"{m.get('profit_factor', 0):.2f}", None),
        ("Sharpe Ratio",    f"{m.get('sharpe', 0):.2f}",  None),
        ("Net P&L",         fmt_currency(m.get("net_pnl", 0)), None),
    ]
    for col, (label, value, _) in zip([c1, c2, c3, c4, c5, c6], metrics):
        with col:
            st.metric(label, value)

    c7, c8, c9, c10 = st.columns(4)
    with c7:
        st.metric("Max Drawdown", fmt_currency(m.get("max_drawdown", 0)),
                  f"{fmt_pct(m.get('max_drawdown_pct', 0))}")
    with c8:
        st.metric("Return %", fmt_pct(m.get("return_pct", 0)))
    with c9:
        st.metric("Winners", m.get("winning_trades", 0))
    with c10:
        st.metric("Losers", m.get("losing_trades", 0))

    # Walk-forward note
    if m.get("walk_forward"):
        st.info(
            f"Walk-forward validation: results shown are **out-of-sample only** "
            f"(last {(1 - m.get('in_sample_pct', 0.7)) * 100:.0f}% of data). "
            "This is the more reliable estimate of live performance."
        )

    # ── Equity curve ──────────────────────────────────────────────────────────────────────────
    st.markdown("### Equity Curve")
    if result.equity_curve:
        equity_fig = go.Figure()
        equity_fig.add_trace(go.Scatter(
            y=result.equity_curve,
            mode="lines",
            name="Equity",
            line=dict(color="#60a5fa", width=2),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.08)",
        ))
        equity_fig.add_hline(
            y=config.account_size,
            line_dash="dash",
            line_color="#94a3b8",
            annotation_text="Starting Capital",
            annotation_position="left",
        )
        equity_fig.update_layout(
            **PLOTLY_DARK,
            height=320,
            margin=dict(l=10, r=10, t=20, b=20),
            xaxis_title="Bar #",
            yaxis_title="Account Value ($)",
        )
        st.plotly_chart(equity_fig, use_container_width=True)

    # ── Trade list ──────────────────────────────────────────────────────────────────────────
    st.markdown("### Trade Log")
    if result.trades:
        trades_df = result.to_dataframe()

        # Colour-code P&L column
        display_cols = [
            "direction", "entry_price", "exit_price",
            "stop_loss", "take_profit",
            "pnl", "exit_reason", "bars_held",
            "strategy", "confidence",
        ]
        display_cols = [c for c in display_cols if c in trades_df.columns]

        def _style_pnl(val):
            if pd.isna(val):
                return ""
            color = "#166534" if val > 0 else "#7f1d1d"
            return f"background-color: {color}; color: {'#86efac' if val > 0 else '#fca5a5'};"

        styled = (
            trades_df[display_cols]
            .style
            .applymap(_style_pnl, subset=["pnl"] if "pnl" in display_cols else [])
            .format({
                "entry_price": "{:.2f}",
                "exit_price":  "{:.2f}",
                "stop_loss":   "{:.2f}",
                "take_profit": "{:.2f}",
                "pnl":         "${:.2f}",
                "confidence":  "{:.2f}",
            }, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, height=300)

        # CSV download
        csv_bytes = trades_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Trade Log (CSV)",
            data=csv_bytes,
            file_name=f"backtest_trades_{datetime.date.today()}.csv",
            mime="text/csv",
        )
    else:
        st.info("No trades in log.")

    # ── Summary text ──────────────────────────────────────────────────────────────────────────
    st.markdown("### Full Summary")
    st.code(result.summary(), language="text")


# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 5 — Trade History & Reports
# ─────────────────────────────────────────────────────────────────────────────────

def page_trade_history():
    """
    Shows all closed trades from the SQLite database.
    Includes date filter, daily P&L chart, and CSV/summary export.
    """
    config = st.session_state.config

    st.title("📋 Trade History & Reports")

    # ── Date range filter ───────────────────────────────────────────────────────────────
    filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
    today = datetime.date.today()

    with filter_col1:
        date_from = st.date_input(
            "From",
            value=today - datetime.timedelta(days=30),
            max_value=today,
        )
    with filter_col2:
        date_to = st.date_input("To", value=today, max_value=today)
    with filter_col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Filters"):
            st.rerun()

    # ── Load trades ──────────────────────────────────────────────────────────────────────────
    try:
        all_trades_df = get_all_trades(limit=1000)
    except Exception as e:
        st.error(f"Could not load trades: {e}")
        all_trades_df = pd.DataFrame()

    if all_trades_df.empty:
        st.info(
            "No trade history found. Trades will appear here once:\n"
            "- You run a backtest and use the 'Log Backtest' option\n"
            "- TradingView webhooks start forwarding orders to Tradovate\n"
            "- You manually add test trades via the Settings page\n\n"
            "Try running a **🧪 Backtest** first to see what the dashboard looks like with data."
        )
        # Show sample data option for demo purposes
        if st.button("📊 Load sample data for preview"):
            _seed_sample_trades()
            st.rerun()
        return

    # ── Apply date filter ─────────────────────────────────────────────────────────────────
    if "closed_at" in all_trades_df.columns and len(all_trades_df) > 0:
        all_trades_df["closed_at"] = pd.to_datetime(all_trades_df["closed_at"], errors="coerce")
        mask = (
            (all_trades_df["closed_at"].dt.date >= date_from) &
            (all_trades_df["closed_at"].dt.date <= date_to)
        )
        filtered_df = all_trades_df[mask].copy()
    else:
        filtered_df = all_trades_df.copy()

    st.caption(f"Showing {len(filtered_df)} trades ({date_from} → {date_to})")

    # ── Summary stats ────────────────────────────────────────────────────────────────────
    if len(filtered_df) > 0:
        s1, s2, s3, s4, s5 = st.columns(5)

        total_pnl = filtered_df["pnl"].sum() if "pnl" in filtered_df.columns else 0
        win_rate  = calculate_win_rate(filtered_df) if "pnl" in filtered_df.columns else 0
        avg_rr    = calculate_avg_rr(filtered_df) if "pnl" in filtered_df.columns else 0

        with s1:
            st.metric("Total Net P&L", fmt_currency(total_pnl))
        with s2:
            st.metric("Trades", len(filtered_df))
        with s3:
            st.metric("Win Rate", fmt_pct(win_rate))
        with s4:
            st.metric("Avg R:R", f"{avg_rr:.2f}")
        with s5:
            commission_total = filtered_df["commission"].sum() if "commission" in filtered_df.columns else 0
            st.metric("Total Commission", fmt_currency(commission_total))

    # ── Daily P&L chart ────────────────────────────────────────────────────────────────────────
    try:
        perf_df = get_performance_history(days=90)
        if not perf_df.empty and "net_pnl" in perf_df.columns:
            st.subheader("Daily P&L")
            perf_df = perf_df.sort_values("date")
            colors = ["#4ade80" if v >= 0 else "#f87171" for v in perf_df["net_pnl"]]

            daily_fig = go.Figure(go.Bar(
                x=perf_df["date"],
                y=perf_df["net_pnl"],
                marker_color=colors,
                name="Daily P&L",
            ))
            daily_fig.update_layout(
                **PLOTLY_DARK,
                height=250,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Date",
                yaxis_title="P&L ($)",
            )
            st.plotly_chart(daily_fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Daily P&L chart unavailable: {e}")

    # ── Trade history table ──────────────────────────────────────────────────────────────────────
    st.subheader("Trade Log")

    if len(filtered_df) > 0:
        # Format for display
        display_cols = [
            "id", "symbol", "direction", "strategy",
            "entry_price", "exit_price", "stop_loss", "take_profit",
            "quantity", "pnl", "commission", "exit_reason",
            "session_date", "closed_at",
        ]
        display_cols = [c for c in display_cols if c in filtered_df.columns]

        def _style_row(val):
            if pd.isna(val):
                return ""
            if isinstance(val, (int, float)):
                return f"color: {'#86efac' if val > 0 else '#fca5a5'};" if val != 0 else ""
            return ""

        fmt_dict = {}
        for col in ["entry_price", "exit_price", "stop_loss", "take_profit"]:
            if col in display_cols:
                fmt_dict[col] = "{:.2f}"
        for col in ["pnl", "commission"]:
            if col in display_cols:
                fmt_dict[col] = "${:.2f}"

        styled = (
            filtered_df[display_cols]
            .style
            .applymap(_style_row, subset=["pnl"] if "pnl" in display_cols else [])
            .format(fmt_dict, na_rep="—")
        )
        st.dataframe(styled, use_container_width=True, height=350, hide_index=True)
    else:
        st.info("No trades in the selected date range.")

    # ── Export ──────────────────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Export")

    exp1, exp2 = st.columns(2)

    with exp1:
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"mnq_trades_{date_from}_{date_to}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Download CSV", disabled=True, use_container_width=True)

    with exp2:
        # Simple text report
        if len(filtered_df) > 0 and "pnl" in filtered_df.columns:
            report_lines = [
                f"MNQ Hybrid Algo Trader — Performance Report",
                f"Period: {date_from} to {date_to}",
                f"Generated: {now_ct().strftime('%Y-%m-%d %H:%M CT')}",
                "=" * 50,
                f"Total Trades:    {len(filtered_df)}",
                f"Win Rate:        {win_rate:.1%}",
                f"Avg R:R:         {avg_rr:.2f}",
                f"Net P&L:         {fmt_currency(filtered_df['pnl'].sum())}",
                f"Commission:      {fmt_currency(filtered_df.get('commission', pd.Series([0])).sum())}",
                "=" * 50,
                "NOTE: Past performance does not guarantee future results.",
                "This report is for educational purposes only.",
            ]
            report_text = "\n".join(report_lines)
            st.download_button(
                "⬇️ Download Text Report",
                data=report_text.encode("utf-8"),
                file_name=f"mnq_report_{date_from}_{date_to}.txt",
                mime="text/plain",
                use_container_width=True,
            )
        else:
            st.button("⬇️ Download Text Report", disabled=True, use_container_width=True)


def _seed_sample_trades():
    """Seed the database with a few sample trades for preview purposes."""
    import random
    random.seed(42)
    session_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

    sample_trades = [
        {"direction": "LONG",  "entry_price": 21480.0, "stop_loss": 21455.0,
         "take_profit": 21524.0, "strategy": "hybrid1", "quantity": 1,
         "session_date": session_date},
        {"direction": "SHORT", "entry_price": 21530.0, "stop_loss": 21555.0,
         "take_profit": 21486.0, "strategy": "hybrid1", "quantity": 1,
         "session_date": session_date},
        {"direction": "LONG",  "entry_price": 21510.0, "stop_loss": 21485.0,
         "take_profit": 21554.0, "strategy": "hybrid2", "quantity": 1,
         "session_date": session_date},
    ]

    exit_prices = [21524.0, 21555.0, 21540.0]
    exit_reasons = ["Take Profit", "Stop Loss", "Take Profit"]

    for trade, ep, er in zip(sample_trades, exit_prices, exit_reasons):
        try:
            tid = log_trade(trade)
            direction = trade["direction"]
            gross = (ep - trade["entry_price"]) * 2 if direction == "LONG" else (trade["entry_price"] - ep) * 2
            comm = 1.24  # $0.62 × 2 sides
            close_trade(tid, ep, er, gross - comm, comm, 0.0)
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────────
# PAGE 6 — Settings
# ─────────────────────────────────────────────────────────────────────────────────

def page_settings():
    """
    App-level settings: paper/live toggle, broker status,
    database management, and about info.
    """
    config = st.session_state.config

    st.title("🛠️ Settings")

    # ── Paper / Live mode ────────────────────────────────────────────────────────────────────
    st.subheader("Trading Mode")

    paper_mode = st.toggle(
        "Paper Mode (Simulated)",
        value=config.paper_mode,
        help=TOOLTIPS["paper_mode"],
    )

    if paper_mode:
        st.markdown(
            '<div class="paper-mode-box">📄 <b>PAPER MODE ACTIVE</b> — All orders are simulated. '
            'No real money at risk. Recommended for beginners and testing.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="live-mode-box">🚨 <b>LIVE MODE</b> — Real money is at risk. '
            'Only trade with money you can afford to lose completely. '
            'Ensure all settings have been tested in paper mode for at least 30 days.</div>',
            unsafe_allow_html=True,
        )
        confirm_live = st.checkbox(
            "I understand the risks and confirm I want to trade with real money",
            value=False,
        )
        if not confirm_live:
            st.warning("Check the box above to confirm live mode, or switch back to paper mode.")

    if st.button("💾 Update Trading Mode"):
        config.paper_mode = paper_mode
        st.session_state.config = config
        st.session_state.paper_mode = paper_mode
        st.success("✅ Trading mode updated.")
        st.rerun()

    st.markdown("---")

    # ── Broker connection status ──────────────────────────────────────────────────────────
    st.subheader("Broker Connection Status")

    broker_col1, broker_col2 = st.columns(2)

    with broker_col1:
        st.markdown("**Tradovate API**")
        if st.session_state.tradovate_connected:
            mode_label = "Demo" if st.session_state.tradovate_credentials.get("demo_mode", True) else "LIVE"
            st.markdown(f'<span class="status-pill pill-green">✅ CONNECTED ({mode_label})</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill pill-yellow">⚪ NOT CONNECTED</span>', unsafe_allow_html=True)
            st.caption("Configure credentials on the 🔗 TV Integration page.")

    with broker_col2:
        st.markdown("**TradingView Webhook**")
        webhook_count = len(st.session_state.webhook_log)
        if webhook_count > 0:
            st.markdown(f'<span class="status-pill pill-green">✅ ACTIVE ({webhook_count} events)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-pill pill-blue">ℹ️ No events yet</span>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Database management ────────────────────────────────────────────────────────────────────
    st.subheader("Database Management")

    db_col1, db_col2, db_col3 = st.columns(3)

    with db_col1:
        st.markdown("**Current Database**")
        try:
            all_df = get_all_trades(limit=9999)
            trade_count = len(all_df)
            st.metric("Closed Trades", trade_count)
        except Exception:
            st.metric("Closed Trades", "Error")

    with db_col2:
        try:
            open_trades = get_open_trades()
            st.metric("Open Trades", len(open_trades))
        except Exception:
            st.metric("Open Trades", "Error")

    with db_col3:
        try:
            perf_df = get_performance_history(days=9999)
            st.metric("Daily Summaries", len(perf_df))
        except Exception:
            st.metric("Daily Summaries", "Error")

    # Backup download
    from utils.database import DB_PATH
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            db_bytes = f.read()
        st.download_button(
            "⬇️ Download Database Backup",
            data=db_bytes,
            file_name=f"trades_backup_{datetime.date.today()}.db",
            mime="application/octet-stream",
        )

    # Danger zone — clear trades
    with st.expander("🗑️ Danger Zone — Clear All Trade Data"):
        st.warning(
            "⚠️ This will permanently delete all trade records from the database. "
            "This action cannot be undone. Download a backup first!"
        )
        confirm_clear = st.checkbox("I understand this will permanently delete all trades")
        if confirm_clear:
            if st.button("🗑️ Clear All Trades", type="secondary"):
                try:
                    import sqlite3
                    conn = sqlite3.connect(DB_PATH)
                    conn.execute("DELETE FROM trades")
                    conn.execute("DELETE FROM daily_summary")
                    conn.commit()
                    conn.close()
                    st.success("✅ All trades cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not clear trades: {e}")

    st.markdown("---")

    # ── About ──────────────────────────────────────────────────────────────────────────────
    st.subheader("About")

    st.markdown("""
    **MNQ Hybrid Algo Trader** v1.0

    A hybrid algorithmic trading system for **Micro E-mini Nasdaq-100 (MNQ)** futures,
    combining two popular trading philosophies:

    | Aspect | Contribution |
    |---|---|
    | **QuantVue-style** | 9/21 EMA crossover momentum entries, volume confirmation, trailing stops |
    | **Vector Algorithmics-style** | ATR-based position sizing, strict daily loss limits, session-only trading |

    **Architecture:**
    - Pine Script v5 strategies generated by this app
    - TradingView fires webhook alerts to Tradovate
    - Risk management enforced by both Pine Script and this app's risk engine
    - SQLite database for trade logging and performance tracking

    **Key Safety Features:**
    - Paper mode enabled by default — no real money at risk
    - Max daily loss auto-shutdown
    - Session-only trading (no overnight holds)
    - Walk-forward backtesting to reduce overfitting

    ---
    """)

    st.info(
        "⚠️ This software is provided for **educational and simulation purposes only**. "
        "It is not financial advice. Trading futures involves substantial risk of loss. "
        "Always start with paper trading. Consult a licensed financial advisor before "
        "risking real capital."
    )

    # System info
    with st.expander("System Information"):
        st.markdown(f"""
        - **Python:** {sys.version.split()[0]}
        - **Streamlit:** {st.__version__}
        - **Pandas:** {pd.__version__}
        - **NumPy:** {np.__version__}
        - **Database path:** `{DB_PATH}`
        - **Strategy engine:** {'\u2705 OK' if _STRATEGY_ENGINE_OK else f'\u26a0\ufe0f {_STRATEGY_ENGINE_ERR}'}
        - **Tradovate client:** {'\u2705 OK' if _TRADOVATE_OK else f'\u26a0\ufe0f {_TRADOVATE_ERR}'}
        - **Paper mode:** {config.paper_mode}
        - **Current time (CT):** {now_ct().strftime('%Y-%m-%d %H:%M:%S')}
        """)


# ─────────────────────────────────────────────────────────────────────────────────
# Main app entrypoint — routes to the selected page
# ─────────────────────────────────────────────────────────────────────────────────

def main():
    """Main app entrypoint. Renders sidebar, then routes to the selected page."""
    page = _render_sidebar()

    # Route to the correct page function
    if page == "📊 Dashboard":
        page_dashboard()
    elif page == "⚙️ Strategy Config":
        page_strategy_config()
    elif page == "🔗 TV Integration":
        page_tv_integration()
    elif page == "🧪 Backtesting":
        page_backtesting()
    elif page == "📋 Trade History":
        page_trade_history()
    elif page == "🛠️ Settings":
        page_settings()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
