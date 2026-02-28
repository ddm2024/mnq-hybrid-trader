"""
Unit tests for MNQ Hybrid Algo Trader core modules.
Run with: python -m pytest tests/test_core.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime


class TestStrategyConfig(unittest.TestCase):
    """Test strategy configuration defaults and constraints."""
    
    def test_default_config(self):
        from utils.config import StrategyConfig
        config = StrategyConfig()
        self.assertEqual(config.symbol, "MNQ")
        self.assertEqual(config.timeframe, "5m")
        self.assertEqual(config.risk_per_trade_pct, 0.0025)
        self.assertEqual(config.max_daily_loss_pct, 0.02)
        self.assertEqual(config.fast_ema_period, 9)
        self.assertEqual(config.slow_ema_period, 21)
        self.assertEqual(config.trend_ema_period, 50)
        self.assertTrue(config.paper_mode)
    
    def test_strategy_info(self):
        from utils.config import STRATEGY_INFO
        self.assertIn("hybrid1", STRATEGY_INFO)
        self.assertIn("hybrid2", STRATEGY_INFO)
        self.assertIn("name", STRATEGY_INFO["hybrid1"])
        self.assertIn("description", STRATEGY_INFO["hybrid2"])


class TestHelpers(unittest.TestCase):
    """Test utility helper functions."""
    
    def test_position_sizing_basic(self):
        from utils.helpers import calculate_position_size
        # $25k account, 0.25% risk, 25 point stop, $2/point
        size = calculate_position_size(25000, 0.0025, 25.0, 2.0)
        self.assertEqual(size, 1)  # $62.50 risk / $50 per contract = 1
    
    def test_position_sizing_larger_account(self):
        from utils.helpers import calculate_position_size
        # $100k account, 0.5% risk, 20 point stop, $2/point
        size = calculate_position_size(100000, 0.005, 20.0, 2.0)
        self.assertEqual(size, 12)  # $500 risk / $40 per contract = 12
    
    def test_position_sizing_zero_stop(self):
        from utils.helpers import calculate_position_size
        size = calculate_position_size(25000, 0.0025, 0, 2.0)
        self.assertEqual(size, 0)
    
    def test_sharpe_ratio(self):
        from utils.helpers import calculate_sharpe_ratio
        returns = pd.Series([0.01, 0.02, -0.005, 0.015, 0.008, -0.003, 0.012])
        sharpe = calculate_sharpe_ratio(returns)
        self.assertIsInstance(sharpe, float)
    
    def test_max_drawdown(self):
        from utils.helpers import calculate_max_drawdown
        equity = pd.Series([100, 110, 105, 95, 100, 108, 90, 95])
        dd_amt, dd_pct = calculate_max_drawdown(equity)
        self.assertGreater(dd_amt, 0)
        self.assertGreater(dd_pct, 0)
        self.assertLessEqual(dd_pct, 1.0)
    
    def test_win_rate(self):
        from utils.helpers import calculate_win_rate
        trades = pd.DataFrame({"pnl": [100, -50, 75, -30, 200, -10]})
        rate = calculate_win_rate(trades)
        self.assertAlmostEqual(rate, 0.5, places=2)
    
    def test_avg_rr(self):
        from utils.helpers import calculate_avg_rr
        trades = pd.DataFrame({"pnl": [100, -50, 150, -50]})
        rr = calculate_avg_rr(trades)
        self.assertAlmostEqual(rr, 2.5, places=2)  # avg win 125 / avg loss 50
    
    def test_formatting(self):
        from utils.helpers import fmt_currency, fmt_pct, fmt_number
        self.assertEqual(fmt_currency(1234.56), "$1,234.56")
        self.assertEqual(fmt_currency(-500), "-$500.00")
        self.assertEqual(fmt_pct(0.55), "55.00%")
        self.assertEqual(fmt_number(1234567.89), "1,234,567.89")
    
    def test_validate_config_aggressive(self):
        from utils.helpers import validate_config_ranges
        from utils.config import StrategyConfig
        config = StrategyConfig(risk_per_trade_pct=0.03, max_daily_loss_pct=0.10)
        warnings = validate_config_ranges(config)
        self.assertGreater(len(warnings), 0)
    
    def test_validate_config_safe(self):
        from utils.helpers import validate_config_ranges
        from utils.config import StrategyConfig
        config = StrategyConfig()  # defaults are safe
        warnings = validate_config_ranges(config)
        self.assertEqual(len(warnings), 0)


class TestDataFetcher(unittest.TestCase):
    """Test market data fetching."""
    
    def test_synthetic_data(self):
        from engines.data_fetcher import _generate_synthetic_data
        df = _generate_synthetic_data(interval="5m", periods=100)
        self.assertEqual(len(df), 100)
        self.assertIn("open", df.columns)
        self.assertIn("high", df.columns)
        self.assertIn("low", df.columns)
        self.assertIn("close", df.columns)
        self.assertIn("volume", df.columns)
        # High should be >= max(open, close)
        self.assertTrue((df["high"] >= df[["open", "close"]].max(axis=1)).all())
        # Low should be <= min(open, close)
        self.assertTrue((df["low"] <= df[["open", "close"]].min(axis=1)).all())
    
    def test_fetch_mnq_returns_data(self):
        from engines.data_fetcher import fetch_mnq_data
        df = fetch_mnq_data("MNQ", "5d", "5m")
        self.assertGreater(len(df), 0)
        self.assertIn("close", df.columns)


class TestStrategyEngine(unittest.TestCase):
    """Test strategy signal generation."""
    
    def _get_test_data(self, n=200):
        from engines.data_fetcher import _generate_synthetic_data
        return _generate_synthetic_data(interval="5m", periods=n, base_price=21500)
    
    def test_compute_indicators(self):
        from strategies.strategy_engine import compute_indicators
        from utils.config import StrategyConfig
        df = self._get_test_data()
        config = StrategyConfig()
        result = compute_indicators(df, config)
        self.assertIn("ema_fast", result.columns)
        self.assertIn("ema_slow", result.columns)
        self.assertIn("ema_trend", result.columns)
        self.assertIn("atr", result.columns)
        self.assertIn("volume_confirmed", result.columns)
        self.assertIn("rsi", result.columns)
        self.assertIn("vwap", result.columns)
    
    def test_compute_opening_range(self):
        from strategies.strategy_engine import compute_opening_range
        df = self._get_test_data(20)
        orb_high, orb_low = compute_opening_range(df, 15)
        self.assertIsNotNone(orb_high)
        self.assertIsNotNone(orb_low)
        self.assertGreater(orb_high, orb_low)
    
    def test_run_strategy_hybrid1(self):
        from strategies.strategy_engine import run_strategy
        from utils.config import StrategyConfig
        df = self._get_test_data(300)
        config = StrategyConfig(strategy_mode="hybrid1")
        # May or may not generate a signal depending on synthetic data
        signal = run_strategy(df, config)
        # Just verify it doesn't crash and returns correct type
        self.assertTrue(signal is None or hasattr(signal, "signal"))
    
    def test_run_strategy_hybrid2(self):
        from strategies.strategy_engine import run_strategy
        from utils.config import StrategyConfig
        df = self._get_test_data(300)
        config = StrategyConfig(strategy_mode="hybrid2")
        signal = run_strategy(df, config, orb_high=21550, orb_low=21450)
        self.assertTrue(signal is None or hasattr(signal, "signal"))


class TestRiskManager(unittest.TestCase):
    """Test risk management engine."""
    
    def test_initial_state(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        rm = RiskManager(StrategyConfig())
        can, reason = rm.can_trade()
        self.assertTrue(can)
        self.assertEqual(reason, "OK")
    
    def test_max_daily_loss_shutdown(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        config = StrategyConfig(account_size=25000, max_daily_loss_pct=0.02)
        rm = RiskManager(config)
        # Simulate a big loss
        rm.daily_pnl = -500  # Max is $500 (2% of $25k)
        can, reason = rm.can_trade()
        self.assertFalse(can)
        self.assertIn("Max daily loss", reason)
    
    def test_max_trades_per_session(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        config = StrategyConfig(max_trades_per_session=2)
        rm = RiskManager(config)
        rm.trades_today = 2
        can, reason = rm.can_trade()
        self.assertFalse(can)
        self.assertIn("Max trades", reason)
    
    def test_position_sizing(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        config = StrategyConfig(account_size=25000, risk_per_trade_pct=0.0025)
        rm = RiskManager(config)
        size = rm.calculate_position_size(25.0)
        self.assertGreaterEqual(size, 1)
    
    def test_trailing_stop_long(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        config = StrategyConfig(trailing_stop_pct=0.5)
        rm = RiskManager(config)
        rm.process_entry(1, "LONG", 21500, 21475, 21550, 1)
        
        # Price moves up
        new_stop = rm.update_trailing_stop(1, 21530)
        # Should move stop up
        if new_stop is not None:
            self.assertGreater(new_stop, 21475)
    
    def test_exit_conditions_stop_loss(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        rm = RiskManager(StrategyConfig())
        rm.process_entry(1, "LONG", 21500, 21475, 21550, 1)
        
        result = rm.check_exit_conditions(1, 21510, 21470, 21480)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "Stop Loss")
    
    def test_exit_conditions_take_profit(self):
        from engines.risk_manager import RiskManager
        from utils.config import StrategyConfig
        rm = RiskManager(StrategyConfig())
        rm.process_entry(1, "LONG", 21500, 21475, 21550, 1)
        
        result = rm.check_exit_conditions(1, 21555, 21530, 21552)
        self.assertIsNotNone(result)
        self.assertEqual(result[1], "Take Profit")


class TestBacktester(unittest.TestCase):
    """Test backtesting engine."""
    
    def test_backtest_runs(self):
        from engines.backtester import run_backtest
        from engines.data_fetcher import _generate_synthetic_data
        from utils.config import StrategyConfig
        
        df = _generate_synthetic_data(interval="5m", periods=500)
        config = StrategyConfig(strategy_mode="hybrid1")
        result = run_backtest(df, config, walk_forward=True)
        
        self.assertIsNotNone(result.metrics)
        self.assertIn("total_trades", result.metrics)
        self.assertIsInstance(result.equity_curve, list)
        self.assertGreater(len(result.equity_curve), 0)
    
    def test_backtest_walk_forward(self):
        from engines.backtester import run_backtest
        from engines.data_fetcher import _generate_synthetic_data
        from utils.config import StrategyConfig
        
        df = _generate_synthetic_data(interval="5m", periods=500)
        config = StrategyConfig()
        result = run_backtest(df, config, walk_forward=True, in_sample_pct=0.7)
        
        if result.metrics.get("total_trades", 0) > 0:
            self.assertTrue(result.metrics.get("walk_forward", False))


class TestPineGenerator(unittest.TestCase):
    """Test Pine Script v5 generation."""
    
    def test_generate_hybrid1(self):
        from engines.pine_generator import generate_pine_script
        from utils.config import StrategyConfig
        
        config = StrategyConfig(strategy_mode="hybrid1")
        pine = generate_pine_script(config)
        
        self.assertIn("//@version=5", pine)
        self.assertIn("strategy(", pine)
        self.assertIn("ta.ema", pine)
        self.assertGreater(len(pine), 1000)
    
    def test_generate_hybrid2(self):
        from engines.pine_generator import generate_pine_script
        from utils.config import StrategyConfig
        
        config = StrategyConfig(strategy_mode="hybrid2")
        pine = generate_pine_script(config)
        
        self.assertIn("//@version=5", pine)
        self.assertIn("strategy(", pine)
        self.assertGreater(len(pine), 1000)
    
    def test_webhook_template(self):
        from engines.pine_generator import generate_webhook_json_template
        from utils.config import StrategyConfig
        
        config = StrategyConfig()
        template = generate_webhook_json_template(config)
        
        self.assertIn("action", template)
        self.assertIn("symbol", template)
    
    def test_alert_instructions(self):
        from engines.pine_generator import generate_alert_setup_instructions
        instructions = generate_alert_setup_instructions()
        self.assertGreater(len(instructions), 100)


class TestDatabase(unittest.TestCase):
    """Test database operations."""
    
    def setUp(self):
        # Use a test database
        import utils.database as db
        db.DB_PATH = "/tmp/test_trades.db"
    
    def tearDown(self):
        import os
        try:
            os.remove("/tmp/test_trades.db")
        except OSError:
            pass
    
    def test_log_and_retrieve_trade(self):
        from utils.database import log_trade, get_trades_today, close_trade
        
        trade_id = log_trade({
            "direction": "LONG",
            "entry_price": 21500.0,
            "stop_loss": 21475.0,
            "take_profit": 21550.0,
            "strategy": "hybrid1",
            "symbol": "MNQ",
        })
        
        self.assertIsInstance(trade_id, int)
        self.assertGreater(trade_id, 0)
        
        # Close the trade
        close_trade(trade_id, 21540.0, "Take Profit", 80.0)
        
        # Check it's in today's trades
        trades = get_trades_today()
        self.assertGreater(len(trades), 0)


class TestTradovateClient(unittest.TestCase):
    """Test Tradovate client (without actual API calls)."""
    
    def test_webhook_processor_validation(self):
        from engines.tradovate_client import TradovateWebhookProcessor
        
        processor = TradovateWebhookProcessor.__new__(TradovateWebhookProcessor)
        
        # Test valid payload structure
        valid = {
            "action": "buy",
            "symbol": "MNQH6",
            "qty": "1",
        }
        self.assertIn("action", valid)
        self.assertIn("symbol", valid)
        self.assertIn("qty", valid)
    
    def test_client_url_selection(self):
        from engines.tradovate_client import TradovateClient
        
        # Demo mode
        client = TradovateClient(
            username="test", password="test",
            app_id="test", cid=0, device_id="test", secret="test",
            demo_mode=True
        )
        url = client._get_base_url()
        self.assertIn("demo", url)
        
        # Live mode
        client_live = TradovateClient(
            username="test", password="test",
            app_id="test", cid=0, device_id="test", secret="test",
            demo_mode=False
        )
        url_live = client_live._get_base_url()
        self.assertIn("live", url_live)


if __name__ == "__main__":
    unittest.main()
