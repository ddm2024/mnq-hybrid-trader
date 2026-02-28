"""
Pine Script v5 Generator — MNQ Hybrid Algo Trader
==================================================
Generates complete, copy-paste-ready TradingView Pine Script v5 code for:

  Hybrid 1 — Momentum-Volatility Fusion
    QuantVue-style 9/21 EMA crossover + volume confirmation
    Vector Algorithmics-inspired ATR breakout & position sizing
    50 EMA trend filter, ATR trailing stop

  Hybrid 2 — 5m MNQ ORB/Pullback
    Opening Range Breakout (first 15 min of RTH)
    EMA pullback entries in trend direction
    Fixed stops, 1.5-2R targets, RTH-only, max trades/session cap

Both scripts produce TradingView webhook-compatible JSON alerts
targeting Tradovate for live/paper execution.

Usage:
    from engines.pine_generator import generate_pine_script
    from utils.config import StrategyConfig
    pine_code = generate_pine_script(StrategyConfig())
"""

from __future__ import annotations
import json
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────────

def _ct_to_et(ct_time: str) -> str:
    """
    Convert a Central Time HH:MM string to Eastern Time HH:MM string.
    CT is UTC-6 (CST) / UTC-5 (CDT); ET is UTC-5 (EST) / UTC-4 (EDT).
    During standard overlapping hours the offset is always +1 hour.
    """
    try:
        h, m = map(int, ct_time.split(":"))
        h_et = (h + 1) % 24
        return f"{h_et:02d}{m:02d}"  # Pine Script time format: HHMM
    except Exception:
        return ct_time.replace(":", "")


def _commission_ticks(commission_per_contract: float) -> int:
    """
    Convert a per-contract dollar commission to MNQ ticks (1 tick = $0.50).
    Pine Script strategy() accepts slippage in ticks.
    """
    return max(1, round(commission_per_contract / 0.50))


# ─────────────────────────────────────────────────────────────────────────────────
# Hybrid 1 Pine Script — Momentum-Volatility Fusion
# ─────────────────────────────────────────────────────────────────────────────────

def _generate_hybrid1(config: Any) -> str:
    """Build the full Pine Script v5 string for Hybrid 1."""

    session_start_et = _ct_to_et(config.session_start)
    session_end_et   = _ct_to_et(config.session_end)
    slippage_ticks   = max(1, round(config.slippage_pct * 20000 / 0.50))
    commission_ticks = _commission_ticks(config.commission_per_contract)

    script = f'''//@version=5
// ╭────────────────────────────────────────────────────────────────────────╮
// │  MNQ Hybrid 1 — Momentum-Volatility Fusion strategy for TradingView     │
// │  Instrument : CME_MINI:MNQ1! on 5-minute timeframe                      │
// │  Entry: 9/21 EMA crossover + 50 EMA trend + volume + ATR breakout       │
// │  Stops: ATR × {config.atr_stop_multiplier} | Targets: {config.reward_risk_ratio}R | Trailing: {config.trailing_stop_pct}% of profit          │
// │  Session: RTH only ({session_start_et}-{session_end_et} ET) | Max {config.max_trades_per_session} trades/session                │
// ╰────────────────────────────────────────────────────────────────────────╯

strategy(
     title                = "MNQ Hybrid 1 — Momentum-Volatility Fusion",
     shorttitle           = "MNQ-H1",
     overlay              = true,
     default_qty_type     = strategy.fixed,
     default_qty_value    = 1,
     initial_capital      = {int(config.account_size)},
     currency             = currency.USD,
     commission_type      = strategy.commission.cash_per_contract,
     commission_value     = {config.commission_per_contract},
     slippage             = {slippage_ticks},
     calc_on_every_tick   = false,
     process_orders_on_close = false
     )

// ============================================================================
// INPUTS
// ============================================================================

var string GRP_RISK = "Account & Risk Management"
i_accountSize      = input.float({config.account_size},    "Account Size ($)",        group=GRP_RISK, minval=1000,   step=1000)
i_riskPct          = input.float({config.risk_per_trade_pct * 100}, "Risk Per Trade (%)", group=GRP_RISK, minval=0.05, maxval=2.0, step=0.05)
i_maxTrades        = input.int  ({config.max_trades_per_session}, "Max Trades / Session", group=GRP_RISK, minval=1, maxval=20)

var string GRP_EMA = "EMA Settings (QuantVue-style)"
i_fastEma          = input.int  ({config.fast_ema_period},  "Fast EMA Period",         group=GRP_EMA,  minval=2,  maxval=50)
i_slowEma          = input.int  ({config.slow_ema_period},  "Slow EMA Period",         group=GRP_EMA,  minval=5,  maxval=100)
i_trendEma         = input.int  ({config.trend_ema_period}, "Trend Filter EMA Period", group=GRP_EMA,  minval=20, maxval=200)

var string GRP_ATR = "ATR & Volatility Settings (Vector-style)"
i_atrPeriod        = input.int  ({config.atr_period},               "ATR Period",              group=GRP_ATR, minval=5, maxval=50)
i_atrStopMult      = input.float({config.atr_stop_multiplier},      "ATR Stop Multiplier",     group=GRP_ATR, minval=0.5, maxval=5.0, step=0.25)
i_atrBreakoutMult  = input.float({config.atr_breakout_multiplier},  "ATR Breakout Multiplier", group=GRP_ATR, minval=0.25, maxval=3.0, step=0.25)
i_rrRatio          = input.float({config.reward_risk_ratio},        "Reward:Risk Ratio",       group=GRP_ATR, minval=1.0, maxval=5.0, step=0.25)
i_trailPct         = input.float({config.trailing_stop_pct},        "Trailing Stop (%)",       group=GRP_ATR, minval=0.1, maxval=5.0, step=0.1)

var string GRP_VOL = "Volume Filter (QuantVue-style)"
i_volSmaPeriod     = input.int  ({config.volume_sma_period},   "Volume SMA Period",     group=GRP_VOL, minval=5,  maxval=50)
i_volMult          = input.float({config.volume_multiplier},   "Volume Multiplier",     group=GRP_VOL, minval=1.0, maxval=3.0, step=0.1)

var string GRP_SESSION = "Session Filter (RTH Only)"
i_sessionStart     = input.session("0930-1600:23456", "RTH Session",  group=GRP_SESSION)
i_showBgColor      = input.bool  (true,               "Highlight RTH session", group=GRP_SESSION)

var string GRP_DISPLAY = "Display & Overlays"
i_showEmas         = input.bool(true,  "Show EMAs",       group=GRP_DISPLAY)
i_showAtrBands     = input.bool(true,  "Show ATR Bands",  group=GRP_DISPLAY)
i_showSignals      = input.bool(true,  "Show Signals",    group=GRP_DISPLAY)

// ============================================================================
// INDICATOR CALCULATIONS
// ============================================================================

emaFast  = ta.ema(close, i_fastEma)
emaSlow  = ta.ema(close, i_slowEma)
emaTrend = ta.ema(close, i_trendEma)

atrVal   = ta.atr(i_atrPeriod)
atrUpper = close + atrVal * i_atrBreakoutMult
atrLower = close - atrVal * i_atrBreakoutMult

volSma        = ta.sma(volume, i_volSmaPeriod)
volumeOk      = volume > volSma * i_volMult

inSession     = not na(time("5", i_sessionStart))

trendBullish  = close > emaTrend
trendBearish  = close < emaTrend

crossUp       = ta.crossover (emaFast, emaSlow)
crossDown     = ta.crossunder(emaFast, emaSlow)

atrBreakoutUp   = high[1] > atrUpper[1]
atrBreakoutDown = low[1]  < atrLower[1]

atrExpanding  = atrVal > atrVal[1]

var int tradesThisSession = 0
isNewSession = ta.change(time("D")) != 0 or (inSession and not inSession[1])
if isNewSession
    tradesThisSession := 0

// ============================================================================
// ENTRY CONDITIONS
// ============================================================================

underTradeLimit = tradesThisSession < i_maxTrades
noPosition      = strategy.position_size == 0

longCondition = crossUp
             and trendBullish
             and volumeOk
             and atrBreakoutUp
             and inSession
             and underTradeLimit
             and noPosition

shortCondition = crossDown
              and trendBearish
              and volumeOk
              and atrBreakoutDown
              and inSession
              and underTradeLimit
              and noPosition

// ============================================================================
// POSITION SIZING (Vector-style risk-adjusted)
// ============================================================================

stopDistPoints = atrVal * i_atrStopMult
dollarRiskPerContract = stopDistPoints * 2.0
dollarRiskAllowed = i_accountSize * (i_riskPct / 100.0)
positionSize = math.max(1, math.floor(dollarRiskAllowed / dollarRiskPerContract))

// ============================================================================
// ENTRY EXECUTION
// ============================================================================

if longCondition
    stopPrice   = close - stopDistPoints
    targetPrice = close + stopDistPoints * i_rrRatio
    strategy.entry("L", strategy.long, qty=positionSize,
         alert_message = \'{{"action":"buy","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","stopLoss":"\' + str.tostring(stopPrice, "#.##") + \'","takeProfit":"\' + str.tostring(targetPrice, "#.##") + \'","strategy":"hybrid1","side":"long"}}\')
    strategy.exit("L-SL/TP", "L",
         stop   = stopPrice,
         limit  = targetPrice,
         trail_price  = targetPrice,
         trail_offset = close * (i_trailPct / 100.0) / syminfo.mintick,
         alert_message = \'{{"action":"close","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","strategy":"hybrid1","side":"long","reason":"SL/TP/trail"}}\')
    tradesThisSession := tradesThisSession + 1

if shortCondition
    stopPrice   = close + stopDistPoints
    targetPrice = close - stopDistPoints * i_rrRatio
    strategy.entry("S", strategy.short, qty=positionSize,
         alert_message = \'{{"action":"sell","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","stopLoss":"\' + str.tostring(stopPrice, "#.##") + \'","takeProfit":"\' + str.tostring(targetPrice, "#.##") + \'","strategy":"hybrid1","side":"short"}}\')
    strategy.exit("S-SL/TP", "S",
         stop   = stopPrice,
         limit  = targetPrice,
         trail_price  = targetPrice,
         trail_offset = close * (i_trailPct / 100.0) / syminfo.mintick,
         alert_message = \'{{"action":"close","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","strategy":"hybrid1","side":"short","reason":"SL/TP/trail"}}\')
    tradesThisSession := tradesThisSession + 1

sessionEnding = not inSession and inSession[1]
if sessionEnding and strategy.position_size != 0
    strategy.close_all(alert_message = \'{{"action":"closeAll","symbol":"{{{{ticker}}}}","strategy":"hybrid1","reason":"session_end"}}\')

// ============================================================================
// VISUAL OVERLAYS
// ============================================================================

plot(i_showEmas ? emaFast  : na, "EMA Fast",  color=color.new(color.yellow, 0),  linewidth=1)
plot(i_showEmas ? emaSlow  : na, "EMA Slow",  color=color.new(color.orange, 0),  linewidth=2)
plot(i_showEmas ? emaTrend : na, "EMA Trend", color=color.new(color.blue,   0),  linewidth=2)

plot(i_showAtrBands ? atrUpper : na, "ATR Upper", color=color.new(color.red,   70), linewidth=1)
plot(i_showAtrBands ? atrLower : na, "ATR Lower", color=color.new(color.green, 70), linewidth=1)

plotshape(i_showSignals and longCondition,  "Long",  shape.triangleup,   location.belowbar, color.new(color.lime, 0), size=size.small)
plotshape(i_showSignals and shortCondition, "Short", shape.triangledown, location.abovebar, color.new(color.red,  0), size=size.small)

bgcolor(i_showBgColor and inSession ? color.new(color.blue, 92) : na, title="RTH Session")

alertcondition(longCondition,  "H1 Long Entry",    "MNQ Hybrid 1 LONG signal")
alertcondition(shortCondition, "H1 Short Entry",   "MNQ Hybrid 1 SHORT signal")
alertcondition(sessionEnding,  "H1 Session Close", "MNQ Hybrid 1 session closing")
'''
    return script


# ─────────────────────────────────────────────────────────────────────────────────
# Hybrid 2 Pine Script — 5m MNQ ORB/Pullback
# ─────────────────────────────────────────────────────────────────────────────────

def _generate_hybrid2(config: Any) -> str:
    """Build the full Pine Script v5 string for Hybrid 2."""

    slippage_ticks   = max(1, round(config.slippage_pct * 20000 / 0.50))
    commission_ticks = _commission_ticks(config.commission_per_contract)

    script = f'''//@version=5
// ╭────────────────────────────────────────────────────────────────────────╮
// │  MNQ Hybrid 2 — 5m ORB/Pullback Strategy for TradingView              │
// │  Instrument : CME_MINI:MNQ1! on 5-minute timeframe                      │
// │  Entry 1: ORB breakout above/below first {config.orb_period_minutes}-min range in trend direction  │
// │  Entry 2: EMA pullback to 9 EMA in 50 EMA trend direction              │
// │  Fixed stops: {config.stop_loss_points} MNQ pts | Targets: {config.reward_risk_ratio}R | RTH-only | Max {config.max_trades_per_session} trades/session      │
// ╰────────────────────────────────────────────────────────────────────────╯

strategy(
     title                = "MNQ Hybrid 2 — 5m ORB/Pullback",
     shorttitle           = "MNQ-H2",
     overlay              = true,
     default_qty_type     = strategy.fixed,
     default_qty_value    = 1,
     initial_capital      = {int(config.account_size)},
     currency             = currency.USD,
     commission_type      = strategy.commission.cash_per_contract,
     commission_value     = {config.commission_per_contract},
     slippage             = {slippage_ticks},
     calc_on_every_tick   = false,
     process_orders_on_close = false
     )

// ============================================================================
// INPUTS
// ============================================================================

var string GRP_RISK = "Account & Risk Management"
i_accountSize      = input.float({config.account_size},       "Account Size ($)",       group=GRP_RISK, minval=1000, step=1000)
i_riskPct          = input.float({config.risk_per_trade_pct * 100}, "Risk Per Trade (%)", group=GRP_RISK, minval=0.05, maxval=2.0, step=0.05)
i_maxTrades        = input.int  ({config.max_trades_per_session}, "Max Trades / Session", group=GRP_RISK, minval=1, maxval=20)

var string GRP_EMA = "EMA Settings"
i_fastEma          = input.int({config.fast_ema_period},  "Fast EMA Period",         group=GRP_EMA, minval=2,  maxval=50)
i_slowEma          = input.int({config.slow_ema_period},  "Slow EMA Period",         group=GRP_EMA, minval=5,  maxval=100)
i_trendEma         = input.int({config.trend_ema_period}, "Trend Filter EMA Period", group=GRP_EMA, minval=20, maxval=200)

var string GRP_ORB = "Opening Range Breakout (ORB)"
i_orbMinutes       = input.int  ({config.orb_period_minutes}, "ORB Period (minutes)",   group=GRP_ORB, minval=5, maxval=60, step=5)
i_orbAtrFilter     = input.bool (true,                        "Require ATR expansion",  group=GRP_ORB)

var string GRP_EXIT = "Stop Loss & Take Profit"
i_stopPoints       = input.float({config.stop_loss_points}, "Stop Loss (MNQ points)", group=GRP_EXIT, minval=5, maxval=100, step=1)
i_rrRatio          = input.float({config.reward_risk_ratio}, "Reward:Risk Ratio",     group=GRP_EXIT, minval=1.0, maxval=5.0, step=0.25)

var string GRP_ATR = "ATR Settings"
i_atrPeriod        = input.int  ({config.atr_period}, "ATR Period", group=GRP_ATR, minval=5, maxval=50)

var string GRP_VOL = "Volume Filter"
i_volSmaPeriod     = input.int  ({config.volume_sma_period}, "Volume SMA Period",   group=GRP_VOL, minval=5, maxval=50)
i_volMult          = input.float({config.volume_multiplier}, "Volume Multiplier",   group=GRP_VOL, minval=1.0, maxval=3.0, step=0.1)

var string GRP_SESSION = "Session Filter (RTH Only)"
i_sessionStart     = input.session("0930-1600:23456", "RTH Session",  group=GRP_SESSION)
i_showBgColor      = input.bool  (true,               "Highlight RTH session", group=GRP_SESSION)

var string GRP_DISPLAY = "Display & Overlays"
i_showEmas         = input.bool(true,  "Show EMAs",           group=GRP_DISPLAY)
i_showOrb          = input.bool(true,  "Show ORB Levels",     group=GRP_DISPLAY)
i_showSignals      = input.bool(true,  "Show Entry Signals",  group=GRP_DISPLAY)

// ============================================================================
// INDICATOR CALCULATIONS
// ============================================================================

emaFast  = ta.ema(close, i_fastEma)
emaSlow  = ta.ema(close, i_slowEma)
emaTrend = ta.ema(close, i_trendEma)

atrVal       = ta.atr(i_atrPeriod)
atrExpanding = atrVal > atrVal[1]

volSma   = ta.sma(volume, i_volSmaPeriod)
volumeOk = volume > volSma * i_volMult

inSession    = not na(time("5", i_sessionStart))
isNewSession = ta.change(time("D")) != 0 or (inSession and not inSession[1])

trendBullish = close > emaTrend
trendBearish = close < emaTrend

var int tradesThisSession = 0
if isNewSession
    tradesThisSession := 0

// ============================================================================
// OPENING RANGE CALCULATION
// ============================================================================

var float orbHigh    = na
var float orbLow     = na
var bool  orbLocked  = false
var int   orbBarsCount = 0

sessionOpenBar = inSession and not inSession[1]

if sessionOpenBar
    orbHigh      := high
    orbLow       := low
    orbLocked    := false
    orbBarsCount := 1

orbBarsNeeded = math.max(1, math.round(i_orbMinutes / 5))

if inSession and not orbLocked and orbBarsCount > 0
    orbHigh      := math.max(orbHigh, high)
    orbLow       := math.min(orbLow,  low)
    orbBarsCount := orbBarsCount + 1
    if orbBarsCount > orbBarsNeeded
        orbLocked := true

orbExpanded    = orbLocked and not na(orbHigh) and not na(orbLow)
orbRangePoints = orbExpanded ? (orbHigh - orbLow) : na

// ============================================================================
// ENTRY CONDITIONS
// ============================================================================

underTradeLimit = tradesThisSession < i_maxTrades
noPosition      = strategy.position_size == 0

orbBreakLong  = orbExpanded
             and ta.crossover(close, orbHigh)
             and trendBullish
             and (not i_orbAtrFilter or atrExpanding)
             and inSession
             and underTradeLimit
             and noPosition

orbBreakShort = orbExpanded
             and ta.crossunder(close, orbLow)
             and trendBearish
             and (not i_orbAtrFilter or atrExpanding)
             and inSession
             and underTradeLimit
             and noPosition

pullbackTouchLong  = low  <= emaFast and close > emaFast
pullbackBullish    = trendBullish
                  and pullbackTouchLong
                  and volumeOk
                  and orbExpanded
                  and inSession
                  and underTradeLimit
                  and noPosition

pullbackTouchShort = high >= emaFast and close < emaFast
pullbackBearish    = trendBearish
                  and pullbackTouchShort
                  and volumeOk
                  and orbExpanded
                  and inSession
                  and underTradeLimit
                  and noPosition

// ============================================================================
// POSITION SIZING (Vector-style)
// ============================================================================

dollarRiskPerContract = i_stopPoints * 2.0
dollarRiskAllowed     = i_accountSize * (i_riskPct / 100.0)
positionSize          = math.max(1, math.floor(dollarRiskAllowed / dollarRiskPerContract))

// ============================================================================
// ENTRY EXECUTION
// ============================================================================

if orbBreakLong
    stopPrice   = close - i_stopPoints
    targetPrice = close + i_stopPoints * i_rrRatio
    strategy.entry("ORB-L", strategy.long, qty=positionSize,
         alert_message = \'{{"action":"buy","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","stopLoss":"\' + str.tostring(stopPrice, "#.##") + \'","takeProfit":"\' + str.tostring(targetPrice, "#.##") + \'","strategy":"hybrid2","type":"orb_long"}}\')
    strategy.exit("ORB-L-Exit", "ORB-L",
         stop  = stopPrice,
         limit = targetPrice,
         alert_message = \'{{"action":"close","symbol":"{{{{ticker}}}}","strategy":"hybrid2","type":"orb_long","reason":"SL/TP"}}\')
    tradesThisSession := tradesThisSession + 1

if orbBreakShort
    stopPrice   = close + i_stopPoints
    targetPrice = close - i_stopPoints * i_rrRatio
    strategy.entry("ORB-S", strategy.short, qty=positionSize,
         alert_message = \'{{"action":"sell","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","stopLoss":"\' + str.tostring(stopPrice, "#.##") + \'","takeProfit":"\' + str.tostring(targetPrice, "#.##") + \'","strategy":"hybrid2","type":"orb_short"}}\')
    strategy.exit("ORB-S-Exit", "ORB-S",
         stop  = stopPrice,
         limit = targetPrice,
         alert_message = \'{{"action":"close","symbol":"{{{{ticker}}}}","strategy":"hybrid2","type":"orb_short","reason":"SL/TP"}}\')
    tradesThisSession := tradesThisSession + 1

if pullbackBullish
    stopPrice   = close - i_stopPoints
    targetPrice = close + i_stopPoints * i_rrRatio
    strategy.entry("PB-L", strategy.long, qty=positionSize,
         alert_message = \'{{"action":"buy","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","stopLoss":"\' + str.tostring(stopPrice, "#.##") + \'","takeProfit":"\' + str.tostring(targetPrice, "#.##") + \'","strategy":"hybrid2","type":"pullback_long"}}\')
    strategy.exit("PB-L-Exit", "PB-L",
         stop  = stopPrice,
         limit = targetPrice,
         alert_message = \'{{"action":"close","symbol":"{{{{ticker}}}}","strategy":"hybrid2","type":"pullback_long","reason":"SL/TP"}}\')
    tradesThisSession := tradesThisSession + 1

if pullbackBearish
    stopPrice   = close + i_stopPoints
    targetPrice = close - i_stopPoints * i_rrRatio
    strategy.entry("PB-S", strategy.short, qty=positionSize,
         alert_message = \'{{"action":"sell","symbol":"{{{{ticker}}}}","qty":"{{{{strategy.order.contracts}}}}","price":"{{{{strategy.order.price}}}}","stopLoss":"\' + str.tostring(stopPrice, "#.##") + \'","takeProfit":"\' + str.tostring(targetPrice, "#.##") + \'","strategy":"hybrid2","type":"pullback_short"}}\')
    strategy.exit("PB-S-Exit", "PB-S",
         stop  = stopPrice,
         limit = targetPrice,
         alert_message = \'{{"action":"close","symbol":"{{{{ticker}}}}","strategy":"hybrid2","type":"pullback_short","reason":"SL/TP"}}\')
    tradesThisSession := tradesThisSession + 1

sessionEnding = not inSession and inSession[1]
if sessionEnding and strategy.position_size != 0
    strategy.close_all(alert_message = \'{{"action":"closeAll","symbol":"{{{{ticker}}}}","strategy":"hybrid2","reason":"session_end"}}\')

// ============================================================================
// VISUAL OVERLAYS
// ============================================================================

plot(i_showEmas ? emaFast  : na, "EMA Fast",  color=color.new(color.yellow, 0), linewidth=1)
plot(i_showEmas ? emaSlow  : na, "EMA Slow",  color=color.new(color.orange, 0), linewidth=2)
plot(i_showEmas ? emaTrend : na, "EMA Trend", color=color.new(color.blue,   0), linewidth=2)

plot(i_showOrb and orbExpanded ? orbHigh : na, "ORB High",
     color=color.new(color.lime, 0), linewidth=2, style=plot.style_stepline)
plot(i_showOrb and orbExpanded ? orbLow  : na, "ORB Low",
     color=color.new(color.red,  0), linewidth=2, style=plot.style_stepline)

plotshape(i_showSignals and orbBreakLong,    "ORB Long",      shape.triangleup,   location.belowbar, color.lime,    size=size.normal)
plotshape(i_showSignals and orbBreakShort,   "ORB Short",     shape.triangledown, location.abovebar, color.red,     size=size.normal)
plotshape(i_showSignals and pullbackBullish, "Pullback Long", shape.triangleup,   location.belowbar, color.aqua,    size=size.small)
plotshape(i_showSignals and pullbackBearish, "Pullback Short",shape.triangledown, location.abovebar, color.fuchsia, size=size.small)

bgcolor(i_showBgColor and inSession ? color.new(color.blue, 92) : na, title="RTH Session")

alertcondition(orbBreakLong,    "H2 ORB Long",       "MNQ Hybrid 2 ORB LONG breakout")
alertcondition(orbBreakShort,   "H2 ORB Short",      "MNQ Hybrid 2 ORB SHORT breakout")
alertcondition(pullbackBullish, "H2 Pullback Long",  "MNQ Hybrid 2 EMA pullback LONG")
alertcondition(pullbackBearish, "H2 Pullback Short", "MNQ Hybrid 2 EMA pullback SHORT")
alertcondition(sessionEnding,   "H2 Session Close",  "MNQ Hybrid 2 session closing")
'''
    return script


# ─────────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────────

def generate_pine_script(config) -> str:
    """
    Generate Pine Script v5 code from a StrategyConfig object.

    Parameters
    ----------
    config : StrategyConfig
        strategy_mode = "hybrid1" or "hybrid2"

    Returns
    -------
    str
        Complete Pine Script v5 string starting with //@version=5.
    """
    mode = getattr(config, "strategy_mode", "hybrid1").lower().strip()
    if mode == "hybrid2":
        return _generate_hybrid2(config)
    return _generate_hybrid1(config)


def generate_webhook_json_template(config) -> str:
    """
    Generate JSON alert message templates for TradingView webhooks targeting Tradovate.
    """
    mode = getattr(config, "strategy_mode", "hybrid1")

    entry_template = {
        "action":   "{{strategy.order.action}}",
        "symbol":   "{{ticker}}",
        "qty":      "{{strategy.order.contracts}}",
        "price":    "{{strategy.order.price}}",
        "strategy": mode,
        "account":  "REPLACE_WITH_TRADOVATE_ACCOUNT_ID",
        "comment":  "{{strategy.order.comment}}"
    }

    exit_template = {
        "action":   "close",
        "symbol":   "{{ticker}}",
        "qty":      "{{strategy.order.contracts}}",
        "price":    "{{strategy.order.price}}",
        "strategy": mode,
        "account":  "REPLACE_WITH_TRADOVATE_ACCOUNT_ID",
        "reason":   "{{strategy.order.comment}}"
    }

    close_all_template = {
        "action":   "closeAll",
        "symbol":   "{{ticker}}",
        "strategy": mode,
        "account":  "REPLACE_WITH_TRADOVATE_ACCOUNT_ID",
        "reason":   "session_end"
    }

    output = f"""// TradingView Webhook Alert Message Templates
// Strategy: {mode.upper()} — CME_MINI:MNQ1!
//
// INSTRUCTIONS:
//   1. In TradingView, right-click strategy → Add alert on strategy
//   2. Set Condition to the strategy order event (order fills)
//   3. Enable Webhook URL and paste your Tradovate webhook endpoint
//   4. Paste ONE template below into the Message field
//   5. Replace REPLACE_WITH_TRADOVATE_ACCOUNT_ID with e.g. demo/12345

// ENTRY ALERT:
{json.dumps(entry_template, indent=2)}

// EXIT ALERT:
{json.dumps(exit_template, indent=2)}

// SESSION CLOSE ALERT:
{json.dumps(close_all_template, indent=2)}

// Tradovate webhook: https://live.tradovateapi.com/webhook/YOUR_WEBHOOK_SECRET
"""
    return output


def generate_alert_setup_instructions() -> str:
    """Return step-by-step instructions for TradingView alert + Tradovate webhook setup."""
    return """
╔════════════════════════════════════════════════════════════════════════════╗
║   TradingView → Tradovate Webhook Setup Guide                              ║
╚════════════════════════════════════════════════════════════════════════════╝

PART 1 — TRADOVATE WEBHOOK SETUP

Step 1: Log into Tradovate → Settings → Third Party Integrations → Webhooks
Step 2: Create webhook, copy the webhook URL and secret
Step 3: Note your account spec: demo/12345 (paper) or live/12345 (live)

PART 2 — TRADINGVIEW PINE SCRIPT SETUP

Step 4: Open TradingView, open CME_MINI:MNQ1! on 5-minute chart
Step 5: Click Pine Editor, paste the generated script, Save, Add to Chart
Step 6: In strategy Settings, configure inputs to match your risk profile

PART 3 — CREATING ALERTS WITH WEBHOOKS

Step 7: Right-click chart → Add Alert
Step 8: Condition = your strategy name, set to "order fills"
Step 9: Set Trigger to "Once Per Bar Close" (NEVER "Once Per Bar")
Step 10: Enable Webhook URL, paste your Tradovate webhook endpoint
Step 11: In the Message field, paste the entry template from this app
Step 12: Test on PAPER account first. Verify fills in Tradovate demo.

PART 4 — SYMBOL MAPPING

TradingView {{ticker}} for CME_MINI:MNQ1! resolves to the front-month:
  MNQH5 (March), MNQM5 (June), MNQU5 (Sept), MNQZ5 (Dec)
Update alerts manually on quarterly rollover days.

⚠️ Always paper trade for 30+ days before risking real capital.
"""


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    try:
        from utils.config import StrategyConfig
    except ImportError:
        from dataclasses import dataclass
        @dataclass
        class StrategyConfig:
            account_size: float = 25000
            risk_per_trade_pct: float = 0.0025
            max_daily_loss_pct: float = 0.02
            symbol: str = "MNQ"
            timeframe: str = "5m"
            trend_ema_period: int = 50
            fast_ema_period: int = 9
            slow_ema_period: int = 21
            volume_sma_period: int = 20
            volume_multiplier: float = 1.2
            atr_period: int = 14
            atr_stop_multiplier: float = 1.5
            atr_breakout_multiplier: float = 1.0
            stop_loss_points: float = 25.0
            reward_risk_ratio: float = 1.75
            trailing_stop_pct: float = 0.5
            use_atr_stops: bool = True
            max_trades_per_session: int = 6
            session_start: str = "08:30"
            session_end: str = "15:00"
            orb_period_minutes: int = 15
            orb_atr_filter: bool = True
            strategy_mode: str = "hybrid1"
            trend_vs_scalp_bias: float = 0.7
            slippage_pct: float = 0.001
            commission_per_contract: float = 0.62
            paper_mode: bool = True

    cfg1 = StrategyConfig(strategy_mode="hybrid1")
    pine1 = generate_pine_script(cfg1)
    print(f"[OK] Hybrid 1: {len(pine1):,} chars")

    cfg2 = StrategyConfig(strategy_mode="hybrid2")
    pine2 = generate_pine_script(cfg2)
    print(f"[OK] Hybrid 2: {len(pine2):,} chars")

    tmpl = generate_webhook_json_template(cfg1)
    print(f"[OK] Webhook template: {len(tmpl):,} chars")

    instr = generate_alert_setup_instructions()
    print(f"[OK] Instructions: {len(instr):,} chars")

    print("[PASS] All functions executed successfully.")
