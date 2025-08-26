"""
Test Aggressive Trading Configuration
Short training run to evaluate aggressive trading behavior
"""
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from config_aggressive import CONFIG_AGGRESSIVE  # Use aggressive config
from data_fetcher import OandaDataFetcher
from technical_indicators import TechnicalIndicators
from trading_env import ForexTradingEnv
from sac_agent import ForexSACAgent
from trading_logger import TradingLogger

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AggressiveConfigTest:
    """
    Test pipeline specifically for aggressive trading configuration
    """
    
    def __init__(self):
        self.config = CONFIG_AGGRESSIVE
        self.data_fetcher = OandaDataFetcher(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.logger_system = TradingLogger(self.config)
        
        # Results storage
        self.results = {}
        
    def fetch_recent_data(self, days_back: int = 30) -> pd.DataFrame:
        """
        Fetch recent data for quick testing
        """
        logger.info(f"Fetching recent data for testing...")
        
        # Get recent data (limit to 500 candles max for Oanda API)
        candles_count = min(500, days_back * 50)  # Conservative estimate
        data = self.data_fetcher.fetch_candles(count=candles_count)
        
        if data is not None:
            logger.info("Adding technical indicators...")
            data = self.indicators.add_all_indicators(data)
            logger.info(f"Data prepared: {len(data)} samples")
            logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")
            return data
        else:
            logger.error("Failed to fetch data")
            return None
    
    def run_aggressive_test(self, days_back: int = 30, timesteps: int = 50000) -> dict:
        """
        Run aggressive configuration test with short training
        """
        try:
            logger.info("=== TESTING AGGRESSIVE TRADING CONFIGURATION ===")
            logger.info(f"Key Aggressive Settings:")
            logger.info(f"  - No Trade Penalty Base: {self.config.no_trade_penalty_base}")
            logger.info(f"  - No Trade Penalty Exp: {self.config.no_trade_penalty_exp}")
            logger.info(f"  - Max No Trade Steps: {self.config.max_no_trade_steps}")
            logger.info(f"  - Trade Action Bonus: {self.config.trade_action_bonus}")
            logger.info(f"  - Lookback Window: {self.config.lookback_window}")
            logger.info(f"  - Position Size Multiplier: {self.config.position_size_multiplier}")
            
            # Step 1: Fetch recent data
            logger.info("Step 1: Fetching recent data...")
            data = self.fetch_recent_data(days_back)
            if data is None:
                return {'success': False, 'error': 'Failed to fetch data'}
            
            # Step 2: Split data (80% training, 20% testing)
            logger.info("Step 2: Splitting data...")
            split_point = int(len(data) * 0.8)
            train_data = data.iloc[:split_point].copy()
            test_data = data.iloc[split_point:].copy()
            
            logger.info(f"Training data: {len(train_data)} samples")
            logger.info(f"Testing data: {len(test_data)} samples")
            
            # Step 3: Create environments
            logger.info("Step 3: Creating environments...")
            train_env_base = ForexTradingEnv(train_data, self.config)
            train_env_monitored = Monitor(train_env_base)
            train_env = DummyVecEnv([lambda: train_env_monitored])
            
            test_env = Monitor(ForexTradingEnv(test_data, self.config))
            
            # Step 4: Train agent (short training)
            logger.info(f"Step 4: Training agent for {timesteps} timesteps...")
            agent = ForexSACAgent(train_env, self.config)
            
            # Start logging
            self.logger_system.start_episode(1, self.config.initial_balance)
            
            # Train with progress tracking
            training_stats = agent.train(timesteps)
            
            # Save model
            model_path = agent.save_model()
            logger.info(f"Model saved to {model_path}")
            
            # Step 5: Quick test
            logger.info("Step 5: Testing aggressive agent...")
            test_results = []
            
            for episode in range(3):  # Quick 3-episode test
                logger.info(f"Test episode {episode + 1}/3")
                
                obs, _ = test_env.reset()
                total_reward = 0
                step = 0
                done = False
                
                while not done:
                    action = agent.get_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    total_reward += reward
                    step += 1
                    done = terminated or truncated
                
                episode_result = {
                    'episode': episode + 1,
                    'final_balance': info['balance'],
                    'total_trades': info['total_trades'],
                    'win_rate': info['win_rate'],
                    'returns': info['returns'],
                    'max_drawdown': info['max_drawdown']
                }
                
                test_results.append(episode_result)
                logger.info(f"  Episode {episode + 1}: Balance=${info['balance']:.2f}, Trades={info['total_trades']}, WinRate={info['win_rate']:.3f}")
            
            # Calculate test summary
            test_summary = {
                'mean_final_balance': np.mean([r['final_balance'] for r in test_results]),
                'mean_returns': np.mean([r['returns'] for r in test_results]),
                'mean_trades': np.mean([r['total_trades'] for r in test_results]),
                'mean_win_rate': np.mean([r['win_rate'] for r in test_results]),
                'profitable_episodes': sum(1 for r in test_results if r['returns'] > 0),
                'episodes': test_results
            }
            
            # Step 6: Results analysis
            logger.info("=== AGGRESSIVE CONFIG TEST RESULTS ===")
            logger.info(f"Training Timesteps: {timesteps}")
            logger.info(f"Mean Final Balance: ${test_summary['mean_final_balance']:.2f}")
            logger.info(f"Mean Returns: {test_summary['mean_returns']:.3f} ({test_summary['mean_returns']*100:.1f}%)")
            logger.info(f"Mean Trades Per Episode: {test_summary['mean_trades']:.1f}")
            logger.info(f"Mean Win Rate: {test_summary['mean_win_rate']:.3f}")
            logger.info(f"Profitable Episodes: {test_summary['profitable_episodes']}/3")
            
            # Trading frequency analysis
            avg_trades_per_episode = test_summary['mean_trades']
            expected_candles_per_episode = len(test_data) - self.config.lookback_window
            trading_frequency = avg_trades_per_episode / expected_candles_per_episode if expected_candles_per_episode > 0 else 0
            
            logger.info(f"Trading Frequency: {trading_frequency:.4f} ({trading_frequency*100:.2f}% of candles)")
            logger.info(f"Target Frequency: {self.config.trade_frequency_target:.4f} ({self.config.trade_frequency_target*100:.2f}% of candles)")
            
            # Assessment
            if test_summary['mean_trades'] >= self.config.trade_frequency_target * expected_candles_per_episode * 0.5:
                assessment = "SUCCESS: Aggressive config is encouraging more trading!"
            else:
                assessment = "NEEDS TUNING: Not enough trading activity despite aggressive settings"
            
            logger.info(f"Assessment: {assessment}")
            
            return {
                'success': True,
                'model_path': model_path,
                'training_stats': training_stats,
                'test_summary': test_summary,
                'trading_frequency': trading_frequency,
                'assessment': assessment,
                'config_type': 'AGGRESSIVE'
            }
            
        except Exception as e:
            logger.error(f"Aggressive config test failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Run aggressive config test"""
    tester = AggressiveConfigTest()
    
    print("🚀 Starting Aggressive Trading Configuration Test")
    print("=" * 60)
    
    # Run test with short training for quick evaluation
    results = tester.run_aggressive_test(days_back=30, timesteps=50000)
    
    if results['success']:
        print("\n✅ AGGRESSIVE CONFIG TEST COMPLETED")
        print(f"Model: {results['model_path']}")
        print(f"Trading Frequency: {results['trading_frequency']*100:.2f}%")
        print(f"Assessment: {results['assessment']}")
    else:
        print(f"\n❌ TEST FAILED: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()