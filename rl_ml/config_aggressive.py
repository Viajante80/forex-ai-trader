"""
Aggressive Trading Configuration - Forces More Trading Activity
Designed to maximize trading frequency and learning opportunities
"""
import os
from dataclasses import dataclass
from typing import Dict, List
import torch
from dotenv import load_dotenv

load_dotenv()
load_dotenv('../.env')

@dataclass
class AggressiveTradingConfig:
    # Instrument settings
    instrument: str = "EUR_USD"
    timeframe: str = "M5"
    
    # Data settings
    start_date: str = "2016-01-01"
    train_end_date: str = "2024-01-01"
    test_end_date: str = "2025-01-01"
    
    # Account settings
    initial_balance: float = 1000.0
    max_stop_loss_pct: float = 0.01  # Reduced to 1% for more frequent trading
    target_multiplier: float = 2.0
    
    # Oanda API settings
    api_key: str = os.getenv("OANDA_API_KEY")
    account_id: str = os.getenv("OANDA_ACCOUNT_ID")
    environment: str = os.getenv("OANDA_ENVIRONMENT", "practice")
    
    # AGGRESSIVE RL settings - designed to force more trading
    learning_rate: float = 3e-4
    buffer_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.1  # Lower entropy for more decisive actions
    
    # Training settings
    total_timesteps: int = 2_000_000  # More training for aggressive behavior
    log_interval: int = 1000
    eval_freq: int = 10000
    save_freq: int = 25000
    
    # AGGRESSIVE Environment settings - key changes here
    lookback_window: int = 30  # Shorter window for faster decisions
    no_trade_penalty_base: float = 0.01  # 100x stronger penalty for not trading!
    no_trade_penalty_exp: float = 1.1   # Exponential growth
    max_no_trade_steps: int = 20        # Heavy penalty after 20 steps (100 min)
    
    # Aggressive reward settings
    trade_action_bonus: float = 0.005    # Bonus just for taking action
    small_profit_bonus: float = 0.01     # Bonus for any profit
    trade_frequency_target: float = 0.05 # Target 5% of candles to have trades
    min_trades_per_episode: int = 1000   # Minimum trades expected per episode
    
    # Risk settings for aggressive trading
    position_size_multiplier: float = 0.02  # 2% position sizes
    allow_multiple_positions: bool = False   # Keep single position for now
    
    # Technical indicators (reduced for faster processing)
    sma_periods: List[int] = None
    ema_periods: List[int] = None
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    
    # Session times (UTC hours)
    sessions: Dict[str, tuple] = None
    
    # M1 settings
    device: str = None
    use_mps: bool = True
    num_threads: int = 8
    
    def __post_init__(self):
        if self.sma_periods is None:
            self.sma_periods = [21, 50, 200]
        if self.ema_periods is None:
            self.ema_periods = [21, 50, 200]
        if self.sessions is None:
            self.sessions = {
                "tokyo": (23, 8),
                "london": (7, 16),
                "ny": (12, 21),
                "sydney": (21, 6),
            }
        
        # Set device for M1 Mac
        if self.device is None:
            if torch.backends.mps.is_available() and self.use_mps:
                self.device = "mps"
                print("🚀 Aggressive Trading: Using M1 GPU acceleration!")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("🚀 Aggressive Trading: Using CUDA GPU acceleration!")
            else:
                self.device = "cpu"
                print("⚠️  Aggressive Trading: Using CPU")
        
        torch.set_num_threads(self.num_threads)
    
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

# Global aggressive config instance
CONFIG_AGGRESSIVE = AggressiveTradingConfig()