"""
M1 Mac Optimized Aggressive Trading Configuration
Combines M1 optimization with aggressive trading parameters
"""
import argparse
import sys
import logging
from train_m1 import ForexTradingPipelineM1
from config_aggressive import CONFIG_AGGRESSIVE
from config_m1 import TradingConfigM1
import torch

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveTradingConfigM1(TradingConfigM1):
    """M1 optimized config with aggressive trading parameters"""
    
    def __init__(self):
        super().__init__()
        
        # Copy aggressive settings from CONFIG_AGGRESSIVE
        self.no_trade_penalty_base = CONFIG_AGGRESSIVE.no_trade_penalty_base
        self.no_trade_penalty_exp = CONFIG_AGGRESSIVE.no_trade_penalty_exp  
        self.max_no_trade_steps = CONFIG_AGGRESSIVE.max_no_trade_steps
        self.trade_action_bonus = CONFIG_AGGRESSIVE.trade_action_bonus
        self.small_profit_bonus = CONFIG_AGGRESSIVE.small_profit_bonus
        self.trade_frequency_target = CONFIG_AGGRESSIVE.trade_frequency_target
        self.min_trades_per_episode = CONFIG_AGGRESSIVE.min_trades_per_episode
        self.position_size_multiplier = CONFIG_AGGRESSIVE.position_size_multiplier
        
        # Keep M1 optimized settings
        self.lookback_window = 30  # Aggressive: shorter window
        self.total_timesteps = 1_000_000  # Full training
        
        logger.info("🔥 M1 Aggressive Trading Configuration Loaded!")
        logger.info(f"   No Trade Penalty Base: {self.no_trade_penalty_base}")
        logger.info(f"   Trade Action Bonus: {self.trade_action_bonus}")
        logger.info(f"   Target Trading Frequency: {self.trade_frequency_target*100:.1f}%")

def main():
    """Run M1 aggressive training"""
    parser = argparse.ArgumentParser(description='M1 Aggressive Forex RL Training')
    parser.add_argument('--days-back', type=int, default=60, help='Days of data (default: 60)')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Training timesteps (default: 1M)')
    parser.add_argument('--test-episodes', type=int, default=10, help='Test episodes (default: 10)')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial balance')
    
    args = parser.parse_args()
    
    # Check M1 availability
    if torch.backends.mps.is_available():
        print("🚀 M1 MPS GPU acceleration detected!")
    else:
        print("⚠️  Running on CPU - consider using M1 Mac for best performance")
    
    # Create aggressive M1 config
    config = AggressiveTradingConfigM1()
    config.initial_balance = args.initial_balance
    config.total_timesteps = args.timesteps
    
    print("🔥 Starting M1 Aggressive Trading Training")
    print("=" * 60)
    print(f"📊 Data: {args.days_back} days")
    print(f"🧠 Training: {args.timesteps:,} timesteps")
    print(f"💰 Initial Balance: ${args.initial_balance:,.2f}")
    print(f"🎯 Target: Maximize trading frequency and profits")
    print("=" * 60)
    
    # Create and run pipeline
    pipeline = ForexTradingPipelineM1(config)
    results = pipeline.run_m1_pipeline(
        days_back=args.days_back,
        training_timesteps=args.timesteps,
        test_episodes=args.test_episodes
    )
    
    if results['success']:
        test_results = results['test_results']
        
        print(f"\n🎉 M1 AGGRESSIVE TRAINING COMPLETED! 🎉")
        print(f"Model: {results['model_path']}")
        print(f"Profitability Rate: {test_results['profitability_rate']:.1%}")
        print(f"Mean Returns: {test_results['mean_returns']:.1%}")
        
        # Aggressive-specific assessment
        if test_results['profitability_rate'] >= 0.7:
            print("🚀 OUTSTANDING aggressive performance!")
        elif test_results['profitability_rate'] >= 0.5:
            print("🔥 EXCELLENT aggressive performance!")
        elif test_results['profitability_rate'] >= 0.3:
            print("✅ GOOD aggressive performance!")
        else:
            print("⚠️  Aggressive config needs tuning")
            
        print(f"\nCheck rl_ml/logs/ for detailed analysis")
    else:
        print(f"\n❌ AGGRESSIVE TRAINING FAILED: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()