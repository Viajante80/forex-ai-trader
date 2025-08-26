"""
Full Historical Training: 2016-2024 Training, 2024-2025 Testing
Professional-grade training with comprehensive historical data
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

class FullHistoricalTrainer:
    """
    Full historical data training pipeline
    2016-2024: Training
    2024-2025: Testing
    """
    
    def __init__(self, config: TradingConfigM1 = None):
        self.config = config or CONFIG_M1
        
        # Override date ranges for full historical training
        self.config.start_date = "2016-01-01"
        self.config.train_end_date = "2024-01-01" 
        self.config.test_end_date = "2025-01-01"
        
        self.data_fetcher = OandaDataFetcher(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.logger_system = TradingLogger(self.config)
        
        # Data storage
        self.full_data = None
        self.train_data = None
        self.test_data = None
        
        # Models and environments
        self.train_env = None
        self.test_env = None
        self.agent = None
        
        logger.info("🚀 Full Historical Trainer Initialized")
        logger.info(f"📅 Training Period: {self.config.start_date} to {self.config.train_end_date}")
        logger.info(f"📅 Testing Period: {self.config.train_end_date} to {self.config.test_end_date}")
    
    def fetch_full_historical_data(self, force_refetch: bool = False) -> pd.DataFrame:
        """
        Fetch complete historical data (2016-2025) with intelligent caching
        """
        cache_file = f"rl_ml/data/{self.config.instrument}_{self.config.timeframe}_full_historical_2016_2025.pkl"
        
        # Check if we have cached data
        if os.path.exists(cache_file) and not force_refetch:
            logger.info("📦 Loading cached full historical data...")
            try:
                self.full_data = pd.read_pickle(cache_file)
                logger.info(f"✅ Loaded cached data: {len(self.full_data)} candles")
                logger.info(f"📊 Date range: {self.full_data.index[0]} to {self.full_data.index[-1]}")
                logger.info(f"🎯 Features: {self.full_data.shape[1]} columns")
                return self.full_data
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cache: {e}, fetching fresh data...")
        
        logger.info("🌐 Fetching full historical data from Oanda (2016-2025)...")
        logger.info("⏳ This will take 20-40 minutes due to API rate limits...")
        
        # Fetch raw historical data
        raw_data = self.data_fetcher.fetch_historical_data(
            start_date=self.config.start_date,
            end_date=self.config.test_end_date
        )
        
        if raw_data is None or len(raw_data) == 0:
            logger.error("❌ Failed to fetch historical data")
            return None
        
        logger.info(f"📈 Raw data fetched: {len(raw_data)} candles")
        logger.info(f"📅 Raw date range: {raw_data.index[0]} to {raw_data.index[-1]}")
        
        # Add technical indicators
        logger.info("🔧 Adding comprehensive technical indicators...")
        logger.info("⏳ Processing 85+ technical features...")
        
        self.full_data = self.indicators.add_all_indicators(raw_data)
        
        if self.full_data is None:
            logger.error("❌ Failed to add technical indicators")
            return None
        
        # Cache the processed data
        logger.info("💾 Caching processed data for future use...")
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            self.full_data.to_pickle(cache_file)
            logger.info(f"✅ Data cached to {cache_file}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to cache data: {e}")
        
        logger.info("🎉 Full historical data preparation completed!")
        logger.info(f"📊 Final dataset: {self.full_data.shape[0]} samples, {self.full_data.shape[1]} features")
        logger.info(f"📅 Date range: {self.full_data.index[0]} to {self.full_data.index[-1]}")
        
        return self.full_data
    
    def split_historical_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training (2016-2024) and testing (2024-2025) sets
        """
        if self.full_data is None:
            raise ValueError("❌ No data available. Call fetch_full_historical_data() first.")
        
        # Split at 2024-01-01
        train_end_date = pd.to_datetime(self.config.train_end_date).tz_localize('UTC')
        
        self.train_data = self.full_data[self.full_data.index < train_end_date].copy()
        self.test_data = self.full_data[self.full_data.index >= train_end_date].copy()
        
        logger.info("📊 Data split completed:")
        logger.info(f"🏋️  Training data (2016-2024): {len(self.train_data):,} samples")
        logger.info(f"🎯 Testing data (2024-2025): {len(self.test_data):,} samples")
        logger.info(f"📈 Train date range: {self.train_data.index[0]} to {self.train_data.index[-1]}")
        logger.info(f"🧪 Test date range: {self.test_data.index[0]} to {self.test_data.index[-1]}")
        
        # Validate data quality
        if len(self.train_data) < 10000:
            logger.warning(f"⚠️  Training data seems small: {len(self.train_data)} samples")
        
        if len(self.test_data) < 1000:
            logger.warning(f"⚠️  Testing data seems small: {len(self.test_data)} samples")
        
        return self.train_data, self.test_data
    
    def create_environments(self):
        """
        Create training and testing environments with historical data
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("❌ Data not split. Call split_historical_data() first.")
        
        logger.info("🏗️  Creating training and testing environments...")
        
        # Training environment (2016-2024)
        logger.info("🏋️  Creating training environment...")
        train_env_base = ForexTradingEnv(self.train_data, self.config)
        train_env_monitored = Monitor(train_env_base)
        self.train_env = DummyVecEnv([lambda: train_env_monitored])
        
        # Testing environment (2024-2025)
        logger.info("🧪 Creating testing environment...")
        test_env_base = ForexTradingEnv(self.test_data, self.config)
        self.test_env = Monitor(test_env_base)
        
        logger.info("✅ Environments created successfully!")
        logger.info(f"🏋️  Training env: {len(self.train_data):,} samples")
        logger.info(f"🧪 Testing env: {len(self.test_data):,} samples")
    
    def train_agent(self, total_timesteps: int = None) -> dict:
        """
        Train SAC agent on historical data (2016-2024)
        """
        if self.train_env is None:
            raise ValueError("❌ Training environment not created.")
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        logger.info("🚀 Starting full historical training...")
        logger.info(f"⏳ Training for {total_timesteps:,} timesteps")
        logger.info(f"📊 Training on {len(self.train_data):,} historical samples (2016-2024)")
        
        # Estimate training time
        if torch.backends.mps.is_available():
            estimated_hours = total_timesteps / 200000  # ~200K steps per hour on M1
            logger.info(f"⌛ Estimated training time on M1: {estimated_hours:.1f} hours")
        else:
            estimated_hours = total_timesteps / 50000   # ~50K steps per hour on CPU
            logger.info(f"⌛ Estimated training time on CPU: {estimated_hours:.1f} hours")
        
        # Create agent
        self.agent = ForexSACAgentM1(self.train_env, self.config)
        
        # Start logging session
        self.logger_system.start_episode(1, self.config.initial_balance)
        
        # Train with progress monitoring
        training_stats = self.agent.train(total_timesteps)
        
        # Save trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"rl_ml/models/sac_forex_full_historical_{timestamp}.zip"
        model_path = self.agent.save_model(model_filename)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"💾 Model saved to: {model_path}")
        
        # Log training completion
        training_info = {
            'balance': training_stats.get('final_performance', {}).get('final_balance', self.config.initial_balance),
            'total_trades': training_stats.get('final_performance', {}).get('total_trades', 0),
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': training_stats.get('final_performance', {}).get('win_rate', 0),
            'total_pnl': training_stats.get('final_performance', {}).get('total_pnl', 0),
            'max_drawdown': training_stats.get('final_performance', {}).get('max_drawdown', 0),
            'returns': training_stats.get('final_performance', {}).get('returns', 0)
        }
        
        self.logger_system.end_episode(training_info)
        
        return {
            'model_path': model_path,
            'training_stats': training_stats,
            'training_info': training_info,
            'training_samples': len(self.train_data),
            'training_period': f"{self.train_data.index[0]} to {self.train_data.index[-1]}"
        }
    
    def test_agent(self, model_path: str = None, n_episodes: int = 10) -> dict:
        """
        Test agent on out-of-sample data (2024-2025)
        """
        if self.test_env is None:
            raise ValueError("❌ Testing environment not created.")
        
        if self.agent is None:
            if model_path is None:
                raise ValueError("❌ No trained agent available and no model path provided")
            
            # Create agent and load model
            self.agent = ForexSACAgentM1(DummyVecEnv([lambda: self.test_env]), self.config)
            self.agent.load_model(model_path)
        
        logger.info("🧪 Starting out-of-sample testing...")
        logger.info(f"📊 Testing on {len(self.test_data):,} samples (2024-2025)")
        logger.info(f"🎯 Running {n_episodes} episodes")
        
        test_results = []
        
        for episode in range(n_episodes):
            logger.info(f"🧪 Testing episode {episode + 1}/{n_episodes}")
            
            self.logger_system.start_episode(episode + 1, self.config.initial_balance)
            
            obs, _ = self.test_env.reset()
            total_reward = 0
            step = 0
            done = False
            
            # Run episode
            with torch.no_grad():
                while not done:
                    action = self.agent.get_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.test_env.step(action)
                    
                    self.logger_system.log_step(step, obs, action, reward, info)
                    
                    total_reward += reward
                    step += 1
                    done = terminated or truncated
                    
                    # Progress update for long episodes
                    if step % 10000 == 0:
                        logger.debug(f"📊 Step {step:,}: Balance=${info['balance']:.2f}, Trades={info['total_trades']}")
            
            self.logger_system.end_episode(info)
            
            episode_result = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': info['balance'],
                'total_trades': info['total_trades'],
                'win_rate': info['win_rate'],
                'returns': info['returns'],
                'max_drawdown': info['max_drawdown'],
                'steps': step,
                'peak_balance': info['peak_balance']
            }
            
            test_results.append(episode_result)
            
            # Episode summary
            logger.info(f"📈 Episode {episode + 1} Results:")
            logger.info(f"  💰 Final Balance: ${episode_result['final_balance']:.2f}")
            logger.info(f"  📊 Returns: {episode_result['returns']:.1%}")
            logger.info(f"  🎯 Win Rate: {episode_result['win_rate']:.3f}")
            logger.info(f"  📈 Total Trades: {episode_result['total_trades']}")
            logger.info(f"  ⚠️  Max Drawdown: {episode_result['max_drawdown']:.3f}")
            
            # Check for exceptional performance
            if episode_result['returns'] >= (self.config.target_multiplier - 1):
                logger.info("🏆 TARGET ACHIEVED! Account doubled!")
            elif episode_result['returns'] >= 0.5:
                logger.info("🎉 EXCELLENT PERFORMANCE! 50%+ returns!")
            elif episode_result['returns'] >= 0.2:
                logger.info("✅ GOOD PERFORMANCE! 20%+ returns!")
        
        # Calculate comprehensive test summary
        returns = [r['returns'] for r in test_results]
        balances = [r['final_balance'] for r in test_results]
        win_rates = [r['win_rate'] for r in test_results]
        drawdowns = [r['max_drawdown'] for r in test_results]
        trades = [r['total_trades'] for r in test_results]
        
        test_summary = {
            'n_episodes': n_episodes,
            'model_path': model_path,
            'test_period': f"{self.test_data.index[0]} to {self.test_data.index[-1]}",
            'test_samples': len(self.test_data),
            'mean_returns': np.mean(returns),
            'std_returns': np.std(returns),
            'best_returns': np.max(returns),
            'worst_returns': np.min(returns),
            'mean_balance': np.mean(balances),
            'best_balance': np.max(balances),
            'worst_balance': np.min(balances),
            'mean_win_rate': np.mean(win_rates),
            'mean_trades': np.mean(trades),
            'mean_drawdown': np.mean(drawdowns),
            'max_drawdown': np.max(drawdowns),
            'profitable_episodes': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / n_episodes,
            'target_reached_episodes': sum(1 for r in returns if r >= (self.config.target_multiplier - 1)),
            'target_success_rate': sum(1 for r in returns if r >= (self.config.target_multiplier - 1)) / n_episodes,
            'episodes': test_results
        }
        
        # Print comprehensive summary
        self._print_test_summary(test_summary)
        
        return test_summary
    
    def _print_test_summary(self, summary: dict):
        """Print comprehensive test summary"""
        print("\n" + "🏆" + "="*78 + "🏆")
        print("           FULL HISTORICAL TRAINING - TEST RESULTS")
        print("🏆" + "="*78 + "🏆")
        print(f"📊 Test Period: {summary['test_period']}")
        print(f"📈 Test Samples: {summary['test_samples']:,} candles")
        print(f"🎯 Episodes: {summary['n_episodes']}")
        print()
        print("💰 PERFORMANCE SUMMARY:")
        print(f"  Mean Returns: {summary['mean_returns']:.1%}")
        print(f"  Best Episode: {summary['best_returns']:.1%}")  
        print(f"  Worst Episode: {summary['worst_returns']:.1%}")
        print(f"  Std Deviation: {summary['std_returns']:.3f}")
        print()
        print("🏦 BALANCE ANALYSIS:")
        print(f"  Initial: ${self.config.initial_balance:.2f}")
        print(f"  Mean Final: ${summary['mean_balance']:.2f}")
        print(f"  Best Final: ${summary['best_balance']:.2f}")
        print(f"  Worst Final: ${summary['worst_balance']:.2f}")
        print()
        print("📊 TRADING STATISTICS:")
        print(f"  Mean Win Rate: {summary['mean_win_rate']:.3f}")
        print(f"  Mean Trades/Episode: {summary['mean_trades']:.1f}")
        print(f"  Mean Drawdown: {summary['mean_drawdown']:.1%}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
        print()
        print("🏆 SUCCESS METRICS:")
        print(f"  Profitable Episodes: {summary['profitable_episodes']}/{summary['n_episodes']}")
        print(f"  Profitability Rate: {summary['profitability_rate']:.1%}")
        print(f"  Target Achieved: {summary['target_reached_episodes']}/{summary['n_episodes']}")
        print(f"  Target Success Rate: {summary['target_success_rate']:.1%}")
        print("🏆" + "="*78 + "🏆")
        
        # Performance assessment
        print("\n🔍 PERFORMANCE ASSESSMENT:")
        if summary['profitability_rate'] >= 0.8:
            print("🏆 OUTSTANDING: Exceptional profitability across time periods!")
        elif summary['profitability_rate'] >= 0.6:
            print("🥇 EXCELLENT: Strong consistent profitability!")
        elif summary['profitability_rate'] >= 0.4:
            print("🥈 GOOD: Decent profitability, room for improvement!")
        else:
            print("🥉 NEEDS WORK: Consider longer training or parameter tuning!")
            
        if summary['target_success_rate'] >= 0.4:
            print("🎯 TARGET MASTERY: Frequently doubles account!")
        elif summary['target_success_rate'] >= 0.2:
            print("📈 TARGET ACHIEVER: Regularly reaches ambitious goals!")
        elif summary['target_success_rate'] >= 0.1:
            print("💪 TARGET CONTENDER: Sometimes reaches high targets!")
        else:
            print("🎯 TARGET OPPORTUNITY: Focus on risk-reward optimization!")
    
    def run_full_historical_training(self, total_timesteps: int = 1000000, 
                                   test_episodes: int = 10,
                                   force_refetch: bool = False) -> dict:
        """
        Run complete historical training pipeline
        """
        try:
            logger.info("🚀 STARTING FULL HISTORICAL TRAINING PIPELINE")
            logger.info("=" * 80)
            
            # Step 1: Fetch historical data
            logger.info("📥 Step 1: Fetching full historical data (2016-2025)...")
            if self.fetch_full_historical_data(force_refetch) is None:
                return {'success': False, 'error': 'Failed to fetch historical data'}
            
            # Step 2: Split data
            logger.info("✂️  Step 2: Splitting data (2016-2024 train, 2024-2025 test)...")
            self.split_historical_data()
            
            # Step 3: Create environments
            logger.info("🏗️  Step 3: Creating training and testing environments...")
            self.create_environments()
            
            # Step 4: Train agent
            logger.info("🏋️  Step 4: Training agent on historical data...")
            training_results = self.train_agent(total_timesteps)
            
            # Step 5: Test agent
            logger.info("🧪 Step 5: Testing agent on out-of-sample data...")
            test_results = self.test_agent(training_results['model_path'], test_episodes)
            
            # Step 6: Generate comprehensive report
            logger.info("📊 Step 6: Generating comprehensive performance report...")
            final_report = self.logger_system.generate_performance_report()
            print(final_report)
            
            logger.info("🎉 FULL HISTORICAL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            
            return {
                'success': True,
                'training_results': training_results,
                'test_results': test_results,
                'model_path': training_results['model_path'],
                'training_samples': len(self.train_data),
                'test_samples': len(self.test_data),
                'final_report': final_report
            }
            
        except Exception as e:
            logger.error(f"❌ Full historical training failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main entry point for full historical training"""
    parser = argparse.ArgumentParser(description='Full Historical Forex RL Training (2016-2024 train, 2024-2025 test)')
    parser.add_argument('--instrument', default='EUR_USD', help='Trading instrument')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Training timesteps (recommended: 1M+)')
    parser.add_argument('--test-episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial account balance')
    parser.add_argument('--force-refetch', action='store_true', help='Force refetch data (ignore cache)')
    parser.add_argument('--use-mps', action='store_true', default=True, help='Use M1 GPU acceleration')
    
    args = parser.parse_args()
    
    # Display training overview
    print("🚀" + "="*78 + "🚀")
    print("         FULL HISTORICAL FOREX RL TRAINING")
    print("🚀" + "="*78 + "🚀")
    print(f"📊 Instrument: {args.instrument}")
    print(f"🏋️  Training Period: 2016-2024 (8 years)")
    print(f"🧪 Testing Period: 2024-2025 (1 year)")
    print(f"⚙️  Training Steps: {args.timesteps:,}")
    print(f"🎯 Test Episodes: {args.test_episodes}")
    print(f"💰 Initial Balance: ${args.initial_balance:.2f}")
    
    # Check M1 availability
    if torch.backends.mps.is_available() and args.use_mps:
        print("🚀 M1 GPU acceleration ENABLED!")
        estimated_hours = args.timesteps / 200000
    else:
        print("⚠️  Using CPU (slower training)")
        estimated_hours = args.timesteps / 50000
    
    print(f"⌛ Estimated total time: {estimated_hours:.1f} hours")
    print("🚀" + "="*78 + "🚀")
    
    # Confirm with user
    if not args.force_refetch:
        response = input("\n📋 Proceed with full historical training? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Training cancelled by user.")
            return
    
    # Create configuration
    config = TradingConfigM1()
    config.instrument = args.instrument
    config.initial_balance = args.initial_balance
    config.total_timesteps = args.timesteps
    config.use_mps = args.use_mps
    
    # Create and run trainer
    trainer = FullHistoricalTrainer(config)
    results = trainer.run_full_historical_training(
        total_timesteps=args.timesteps,
        test_episodes=args.test_episodes,
        force_refetch=args.force_refetch
    )
    
    if results['success']:
        print(f"\n🎉 TRAINING SUCCESS!")
        print(f"💾 Model: {results['model_path']}")
        print(f"📊 Training Samples: {results['training_samples']:,}")
        print(f"🧪 Test Samples: {results['test_samples']:,}")
        print(f"📁 Check rl_ml/logs/ for detailed reports")
        
        # Performance summary
        test_results = results['test_results']
        print(f"\n📈 FINAL PERFORMANCE:")
        print(f"  Profitability Rate: {test_results['profitability_rate']:.1%}")
        print(f"  Mean Returns: {test_results['mean_returns']:.1%}")
        print(f"  Target Success Rate: {test_results['target_success_rate']:.1%}")
        
    else:
        print(f"\n❌ TRAINING FAILED")
        print(f"Error: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()