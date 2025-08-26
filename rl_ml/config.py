"""
Configuration file for the RL Forex Trading Agent
"""
import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file (check parent directory as well)
load_dotenv()
load_dotenv('../.env')

@dataclass
class TradingConfig:
    # Instrument settings
    instrument: str = "EUR_USD"
    timeframe: str = "M5"  # 5-minute candles
    
    # Data settings
    start_date: str = "2016-01-01"
    train_end_date: str = "2024-01-01"
    test_end_date: str = "2025-01-01"
    
    # Account settings
    initial_balance: float = 1000.0
    max_stop_loss_pct: float = 0.02  # 2% max SL
    target_multiplier: float = 2.0  # Stop trading when account doubles
    
    # Oanda API settings
    api_key: str = os.getenv("OANDA_API_KEY")
    account_id: str = os.getenv("OANDA_ACCOUNT_ID")
    environment: str = os.getenv("OANDA_ENVIRONMENT", "practice")
    
    # RL settings
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2  # SAC entropy coefficient
    
    # Training settings
    total_timesteps: int = 1_000_000
    log_interval: int = 1000
    eval_freq: int = 10000
    save_freq: int = 50000
    
    # Environment settings
    lookback_window: int = 100  # Number of candles to look back
    no_trade_penalty_base: float = 0.0001
    no_trade_penalty_exp: float = 1.05
    max_no_trade_steps: int = 100
    
    # Technical indicators
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    
    # Session times (UTC hours)
    sessions: Dict[str, tuple] = None
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [21, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [21, 50, 200]
        if self.sessions is None:
            self.sessions = {
                "tokyo": (23, 8),      # 23:00 - 08:00 UTC
                "london": (7, 16),     # 07:00 - 16:00 UTC
                "ny": (12, 21),        # 12:00 - 21:00 UTC
                "sydney": (21, 6),     # 21:00 - 06:00 UTC (next day)
            }
    
    @property
    def base_url(self):
        if self.environment == "practice":
            return "https://api-fxpractice.oanda.com"
        else:
            return "https://api-fxtrade.oanda.com"
    
    @property
    def headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

# Global config instance
CONFIG = TradingConfig()