"""
Full Historical M1 Training: 2016-2024 Training, 2024-2025 Testing
Complete professional training with full historical dataset + M1 optimization
"""
import argparse
import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from config_m1 import CONFIG_M1, TradingConfigM1
from config_aggressive import CONFIG_AGGRESSIVE
from data_fetcher import OandaDataFetcher
from technical_indicators import TechnicalIndicators
from trading_env import ForexTradingEnv
from sac_agent_m1 import ForexSACAgentM1
from trading_logger import TradingLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FullHistoricalM1Trainer:
    """
    Full historical M1 training pipeline
    - 2016-2024: Training (8+ years of data)
    - 2024-2025: Testing (1 year of out-of-sample data)
    - M1 GPU optimized
    """
    
    def __init__(self, config: TradingConfigM1 = None, aggressive: bool = False):
        self.config = config or CONFIG_M1
        self.aggressive = aggressive
        
        # Full historical date ranges
        self.config.start_date = "2016-01-01"
        self.config.train_end_date = "2024-01-01" 
        self.config.test_end_date = "2025-01-01"
        
        # Apply aggressive settings if requested
        if aggressive:
            logger.info("🔥 APPLYING AGGRESSIVE TRADING SETTINGS")
            self.config.no_trade_penalty_base = CONFIG_AGGRESSIVE.no_trade_penalty_base
            self.config.no_trade_penalty_exp = CONFIG_AGGRESSIVE.no_trade_penalty_exp
            self.config.max_no_trade_steps = CONFIG_AGGRESSIVE.max_no_trade_steps
            self.config.trade_action_bonus = CONFIG_AGGRESSIVE.trade_action_bonus
            self.config.small_profit_bonus = CONFIG_AGGRESSIVE.small_profit_bonus
            self.config.trade_frequency_target = CONFIG_AGGRESSIVE.trade_frequency_target
            self.config.lookback_window = 30  # Shorter for aggressive
        
        self.data_fetcher = OandaDataFetcher(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.logger_system = TradingLogger(self.config)
        
        # Data storage
        self.full_data = None
        self.train_data = None
        self.test_data = None
        
    def fetch_full_historical_data(self) -> pd.DataFrame:
        """
        Fetch or load complete historical dataset (2016-2025)
        """
        data_file = f"data/{self.config.instrument}_{self.config.timeframe}_FULL_2016_2025_ENHANCED.pkl"
        
        logger.info("🏛️  LOADING FULL HISTORICAL DATASET (2016-2025)")
        
        # Check for existing data in rl_ml/data folder
        if os.path.exists(data_file):
            logger.info(f"Loading cached full historical data from {data_file}")
            try:
                self.full_data = pd.read_pickle(data_file)
                logger.info(f"✅ Loaded {len(self.full_data)} historical samples")
                logger.info(f"📅 Date range: {self.full_data.index[0]} to {self.full_data.index[-1]}")
                logger.info(f"📊 Features: {self.full_data.shape[1]} columns")
                return self.full_data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        # Check for existing enhanced data in local data folder
        enhanced_files = [f for f in os.listdir("data/") if f.startswith(f"{self.config.instrument}_{self.config.timeframe}_enhanced")]
        if enhanced_files:
            latest_file = sorted(enhanced_files)[-1]  # Get most recent
            enhanced_data_file = f"data/{latest_file}"
            logger.info(f"🔍 Found existing enhanced data: {enhanced_data_file}")
            try:
                self.full_data = pd.read_pickle(enhanced_data_file)
                logger.info(f"✅ Loaded enhanced data: {len(self.full_data)} samples")
                logger.info(f"📅 Date range: {self.full_data.index[0]} to {self.full_data.index[-1]}")
                logger.info(f"📊 Features: {self.full_data.shape[1]} columns (with technical indicators)")
                return self.full_data
            except Exception as e:
                logger.warning(f"Failed to load enhanced data: {e}")
        
        # Check for existing data in parent project (part3) and process it
        parent_data_file = f"../part3/data/{self.config.instrument}/{self.config.instrument}_{self.config.timeframe}_with_bid_ask.pkl"
        if os.path.exists(parent_data_file):
            logger.info(f"🔍 Found existing data in parent project: {parent_data_file}")
            logger.info("📊 Processing with technical indicators (this may take a few minutes)...")
            
            # Use the existing processor
            from process_existing_data import add_compatible_indicators
            try:
                raw_data = pd.read_pickle(parent_data_file)
                logger.info(f"✅ Loaded existing data: {len(raw_data)} samples")
                logger.info(f"📅 Date range: {raw_data.index[0]} to {raw_data.index[-1]}")
                
                # Process with technical indicators
                self.full_data = add_compatible_indicators(raw_data)
                
                # Save processed data for future use
                try:
                    os.makedirs(os.path.dirname(data_file), exist_ok=True)
                    self.full_data.to_pickle(data_file)
                    logger.info(f"💾 Enhanced data cached to {data_file}")
                except Exception as e:
                    logger.warning(f"Failed to cache enhanced data: {e}")
                
                logger.info(f"✅ Full historical dataset ready: {self.full_data.shape}")
                return self.full_data
                
            except Exception as e:
                logger.warning(f"Failed to process existing data: {e}")
        
        logger.info("🔄 Fetching fresh historical data from Oanda (this may take 30-60 minutes)...")
        
        # Fetch full historical range
        raw_data = self.data_fetcher.fetch_historical_data(
            start_date=self.config.start_date,
            end_date=self.config.test_end_date
        )
        
        if raw_data is None or len(raw_data) == 0:
            logger.error("❌ Failed to fetch historical data")
            return None
        
        logger.info(f"📈 Raw historical data: {len(raw_data)} candles")
        logger.info(f"📅 Coverage: {raw_data.index[0]} to {raw_data.index[-1]}")
        
        # Add all technical indicators
        logger.info("⚙️  Adding comprehensive technical indicators...")
        self.full_data = self.indicators.add_all_indicators(raw_data)
        
        # Save processed data for future use
        try:
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            self.full_data.to_pickle(data_file)
            logger.info(f"💾 Full historical data cached to {data_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
        
        logger.info(f"✅ Full historical dataset ready: {self.full_data.shape}")
        return self.full_data
    
    def split_historical_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into training and testing based on available data
        """
        if self.full_data is None:
            raise ValueError("No data available. Call fetch_full_historical_data() first.")
        
        # Check actual data range
        data_start = self.full_data.index[0]
        data_end = self.full_data.index[-1]
        logger.info(f"📅 Available data: {data_start} to {data_end}")
        
        # If we have historical data from 2016, use the original split
        if data_start.year <= 2020:
            train_end = pd.to_datetime(self.config.train_end_date).tz_localize('UTC')
        else:
            # Use 80% for training, 20% for testing
            total_samples = len(self.full_data)
            train_samples = int(total_samples * 0.8)
            train_end = self.full_data.index[train_samples]
            logger.info(f"⚠️  Limited historical data - using 80/20 split at {train_end}")
        
        self.train_data = self.full_data[self.full_data.index < train_end].copy()
        self.test_data = self.full_data[self.full_data.index >= train_end].copy()
        
        logger.info("📊 HISTORICAL DATA SPLIT:")
        if len(self.train_data) > 0:
            logger.info(f"   🎓 Training: {len(self.train_data):,} samples ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        else:
            logger.error("   🎓 Training: 0 samples - NO TRAINING DATA!")
            
        if len(self.test_data) > 0:
            logger.info(f"   🧪 Testing:  {len(self.test_data):,} samples ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        else:
            logger.error("   🧪 Testing: 0 samples - NO TEST DATA!")
        
        if len(self.train_data) == 0:
            raise ValueError("No training data available after split!")
            
        return self.train_data, self.test_data
    
    def run_full_training(self, timesteps: int = 2000000, test_episodes: int = 10) -> dict:
        """
        Run complete full historical training
        """
        try:
            config_type = "AGGRESSIVE" if self.aggressive else "STANDARD"
            logger.info(f"🚀 STARTING FULL HISTORICAL M1 TRAINING ({config_type}) 🚀")
            logger.info("=" * 80)
            
            if torch.backends.mps.is_available():
                logger.info("🔥 M1 Mac GPU acceleration enabled!")
            else:
                logger.info("⚠️  Running on CPU")
            
            # Step 1: Load full historical data
            logger.info("Step 1: Loading full historical dataset...")
            if self.fetch_full_historical_data() is None:
                return {'success': False, 'error': 'Failed to load historical data'}
            
            # Step 2: Split data
            logger.info("Step 2: Splitting historical data...")
            self.split_historical_data()
            
            # Step 3: Create environments
            logger.info("Step 3: Creating training environments...")
            train_env_base = ForexTradingEnv(self.train_data, self.config)
            train_env = DummyVecEnv([lambda: Monitor(train_env_base)])
            
            test_env = Monitor(ForexTradingEnv(self.test_data, self.config))
            
            # Step 4: Full historical training
            logger.info(f"Step 4: Starting full historical training ({timesteps:,} timesteps)...")
            logger.info(f"⏰ Estimated time on M1 Mac: {timesteps/200000:.1f} hours")
            
            agent = ForexSACAgentM1(train_env, self.config)
            self.logger_system.start_episode(1, self.config.initial_balance)
            
            # Train with full dataset
            training_stats = agent.train(timesteps)
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"sac_forex_FULL_HISTORICAL_{config_type}_m1_{timestamp}.zip"
            model_path = agent.save_model(model_filename)
            logger.info(f"💾 Full historical model saved: {model_path}")
            
            # Step 5: Comprehensive testing
            logger.info(f"Step 5: Testing on out-of-sample data ({test_episodes} episodes)...")
            test_results = []
            
            for episode in range(test_episodes):
                logger.info(f"Testing episode {episode + 1}/{test_episodes}")
                
                obs, _ = test_env.reset()
                total_reward = 0
                step = 0
                done = False
                
                with torch.no_grad():
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
                
                logger.info(f"Episode {episode + 1}: ${info['balance']:.2f} ({info['returns']:+.1%}) - {info['total_trades']} trades")
            
            # Final analysis
            returns = [r['returns'] for r in test_results]
            trades = [r['total_trades'] for r in test_results]
            win_rates = [r['win_rate'] for r in test_results]
            
            results_summary = {
                'success': True,
                'config_type': config_type,
                'model_path': model_path,
                'training_timesteps': timesteps,
                'mean_returns': np.mean(returns),
                'std_returns': np.std(returns),
                'best_returns': np.max(returns),
                'worst_returns': np.min(returns),
                'mean_trades': np.mean(trades),
                'mean_win_rate': np.mean(win_rates),
                'profitable_episodes': sum(1 for r in returns if r > 0),
                'profitability_rate': sum(1 for r in returns if r > 0) / test_episodes,
                'target_achievers': sum(1 for r in returns if r >= 1.0),  # Doubled account
                'episodes': test_results
            }
            
            # Generate comprehensive report
            report = self.logger_system.generate_performance_report()
            
            # Performance assessment
            prof_rate = results_summary['profitability_rate']
            mean_ret = results_summary['mean_returns']
            
            if prof_rate >= 0.7 and mean_ret >= 0.15:
                assessment = "🚀 OUTSTANDING - Production ready!"
            elif prof_rate >= 0.6 and mean_ret >= 0.10:
                assessment = "🔥 EXCELLENT - Strong performance!"
            elif prof_rate >= 0.4 and mean_ret >= 0.05:
                assessment = "✅ GOOD - Promising results!"
            else:
                assessment = "⚠️  NEEDS IMPROVEMENT - Consider more training"
            
            logger.info("🎯 FULL HISTORICAL TRAINING RESULTS:")
            logger.info(f"   Model: {model_path}")
            logger.info(f"   Training: {timesteps:,} timesteps on 8+ years data")
            logger.info(f"   Profitability Rate: {prof_rate:.1%}")
            logger.info(f"   Mean Returns: {mean_ret:+.1%}")
            logger.info(f"   Best Episode: {results_summary['best_returns']:+.1%}")
            logger.info(f"   Target Achievers: {results_summary['target_achievers']}/{test_episodes}")
            logger.info(f"   Assessment: {assessment}")
            
            results_summary['assessment'] = assessment
            return results_summary
            
        except Exception as e:
            logger.error(f"Full historical training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

def main():
    """Main entry point for full historical training"""
    parser = argparse.ArgumentParser(description='Full Historical M1 Forex RL Training (2016-2025)')
    parser.add_argument('--timesteps', type=int, default=2000000, 
                       help='Training timesteps (default: 2M for full training)')
    parser.add_argument('--test-episodes', type=int, default=10, 
                       help='Test episodes (default: 10)')
    parser.add_argument('--aggressive', action='store_true', 
                       help='Use aggressive trading configuration')
    parser.add_argument('--initial-balance', type=float, default=1000.0, 
                       help='Initial balance (default: $1000)')
    
    args = parser.parse_args()
    
    # M1 check
    if torch.backends.mps.is_available():
        print("🚀 M1 Mac GPU detected - optimal performance!")
    else:
        print("⚠️  CPU training - will be much slower")
    
    config_type = "AGGRESSIVE" if args.aggressive else "STANDARD"
    
    print("🏛️  FULL HISTORICAL FOREX RL TRAINING")
    print("=" * 80)
    print(f"📊 Dataset: Complete 2016-2025 historical data")
    print(f"🎓 Training: 2016-2024 (8+ years)")
    print(f"🧪 Testing: 2024-2025 (out-of-sample)")
    print(f"🧠 Training: {args.timesteps:,} timesteps")
    print(f"💰 Balance: ${args.initial_balance:,.2f}")
    print(f"⚙️  Config: {config_type}")
    print(f"⏰ Est. Time: {args.timesteps/200000:.1f} hours (M1 Mac)")
    print("=" * 80)
    
    # Create config
    config = TradingConfigM1()
    config.initial_balance = args.initial_balance
    config.total_timesteps = args.timesteps
    
    # Create and run trainer
    trainer = FullHistoricalM1Trainer(config, aggressive=args.aggressive)
    results = trainer.run_full_training(args.timesteps, args.test_episodes)
    
    if results['success']:
        print(f"\n🎉 FULL HISTORICAL TRAINING COMPLETED! 🎉")
        print(f"✅ {results['assessment']}")
        print(f"📄 Model: {results['model_path']}")
        print(f"📊 Performance: {results['mean_returns']:+.1%} avg returns")
        print(f"🎯 Success Rate: {results['profitability_rate']:.1%}")
        
        # Save results summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/full_historical_results_{timestamp}.txt"
        with open(results_file, 'w') as f:
            f.write(f"Full Historical Training Results\n")
            f.write(f"================================\n")
            f.write(f"Config: {results['config_type']}\n")
            f.write(f"Model: {results['model_path']}\n")
            f.write(f"Training: {results['training_timesteps']:,} timesteps\n")
            f.write(f"Mean Returns: {results['mean_returns']:+.2%}\n")
            f.write(f"Profitability Rate: {results['profitability_rate']:.2%}\n")
            f.write(f"Assessment: {results['assessment']}\n")
        
        print(f"📝 Detailed results: {results_file}")
    else:
        print(f"\n❌ TRAINING FAILED: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()