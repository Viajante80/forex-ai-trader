"""
Configuration file for RL Forex Trading Agent - M1 Mac Optimized
"""
import os
from dataclasses import dataclass
from typing import Dict, List
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
load_dotenv('../.env')

@dataclass
class TradingConfigM1:
    # Instrument settings
    instrument: str = "EUR_USD"
    timeframe: str = "M5"
    
    # Data settings - optimized for M1 Mac memory
    start_date: str = "2024-01-01"  # More recent start date
    train_end_date: str = "2024-08-01"
    test_end_date: str = "2024-08-23"
    
    # Account settings
    initial_balance: float = 1000.0
    max_stop_loss_pct: float = 0.02
    target_multiplier: float = 2.0
    
    # Oanda API settings
    api_key: str = os.getenv("OANDA_API_KEY")
    account_id: str = os.getenv("OANDA_ACCOUNT_ID")
    environment: str = os.getenv("OANDA_ENVIRONMENT", "practice")
    
    # RL settings - optimized for M1 Mac
    learning_rate: float = 3e-4
    buffer_size: int = 50_000  # Reduced for M1 Mac
    batch_size: int = 128  # Optimized for M1 GPU
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    
    # Training settings - M1 optimized
    total_timesteps: int = 100_000  # More reasonable for testing
    log_interval: int = 1000
    eval_freq: int = 5000
    save_freq: int = 10000
    
    # Environment settings
    lookback_window: int = 50  # Reduced for better performance
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
    
    # M1 Mac specific settings
    device: str = None
    use_mps: bool = True  # Use Metal Performance Shaders
    num_threads: int = 8  # M1 Pro/Max cores
    
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
                self.device = "mps"  # Metal Performance Shaders for M1
                print("🚀 Using M1 GPU (MPS) acceleration!")
            elif torch.cuda.is_available():
                self.device = "cuda"
                print("🚀 Using CUDA GPU acceleration!")
            else:
                self.device = "cpu"
                print("⚠️  Using CPU (consider enabling MPS for M1)")
        
        # Set torch threads for M1 optimization
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

# Global config instance optimized for M1
CONFIG_M1 = TradingConfigM1()