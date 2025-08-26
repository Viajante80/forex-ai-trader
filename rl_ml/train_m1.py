"""
M1 Mac Optimized Training Pipeline for RL Forex Trading Agent
Optimized for Apple Silicon with MPS acceleration and memory efficiency
"""
import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
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

# Configure logging for M1
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ForexTradingPipelineM1:
    """M1 Mac optimized trading pipeline"""
    
    def __init__(self, config: TradingConfigM1 = None):
        self.config = config or CONFIG_M1
        
        # M1 optimizations
        if torch.backends.mps.is_available():
            logger.info("🚀 M1 Mac detected - enabling optimizations")
        else:
            logger.info("⚠️  Running on non-M1 system")
        
        self.data_fetcher = OandaDataFetcher(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.logger_system = TradingLogger(self.config)
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.full_data = None
        
        # Models and environments
        self.train_env = None
        self.test_env = None
        self.agent = None
        
    def fetch_recent_data(self, days_back: int = 30) -> pd.DataFrame:
        """Fetch recent data optimized for M1 Mac memory constraints"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching M1 optimized data from {start_str} to {end_str}")
        
        # Check cache first
        cache_file = f"rl_ml/data/{self.config.instrument}_{self.config.timeframe}_m1_recent.pkl"
        
        if os.path.exists(cache_file):
            try:
                cached_data = pd.read_pickle(cache_file)
                # Check if cache is recent enough (within 1 day)
                if cached_data.index[-1] > pd.Timestamp.now(tz='UTC') - timedelta(days=1):
                    logger.info(f"Using cached M1 data: {len(cached_data)} samples")
                    self.full_data = cached_data
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        
        # Fetch fresh data
        raw_data = self.data_fetcher.fetch_historical_data(start_str, end_str)
        
        if raw_data is None:
            logger.error("Failed to fetch data")
            return None
        
        logger.info(f"Raw data: {len(raw_data)} candles")
        
        # Add indicators with M1 memory optimization
        logger.info("Adding technical indicators (M1 optimized)...")
        self.full_data = self.indicators.add_all_indicators(raw_data)
        
        # Save to cache
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            self.full_data.to_pickle(cache_file)
            logger.info(f"M1 data cached to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")
        
        logger.info(f"Enhanced M1 data: {self.full_data.shape}")
        return self.full_data
    
    def split_data(self, train_ratio: float = 0.75) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data optimized for M1 Mac memory"""
        if self.full_data is None:
            raise ValueError("No data available. Call fetch_recent_data() first.")
        
        split_point = int(len(self.full_data) * train_ratio)
        self.train_data = self.full_data.iloc[:split_point].copy()
        self.test_data = self.full_data.iloc[split_point:].copy()
        
        logger.info(f"M1 Training data: {len(self.train_data)} samples")
        logger.info(f"M1 Testing data: {len(self.test_data)} samples")
        
        return self.train_data, self.test_data
    
    def create_environments(self):
        """Create M1 optimized environments"""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        logger.info("Creating M1 optimized environments...")
        
        # Training environment
        train_env_base = ForexTradingEnv(self.train_data, self.config)
        train_env_monitored = Monitor(train_env_base)
        self.train_env = DummyVecEnv([lambda: train_env_monitored])
        
        # Testing environment  
        test_env_base = ForexTradingEnv(self.test_data, self.config)
        self.test_env = Monitor(test_env_base)
        
        logger.info("M1 environments created successfully")
    
    def train_agent(self, total_timesteps: int = None) -> dict:
        """Train M1 optimized SAC agent"""
        if self.train_env is None:
            raise ValueError("Training environment not created.")
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        logger.info(f"Starting M1 optimized training for {total_timesteps} timesteps...")
        
        # Create M1 optimized agent
        self.agent = ForexSACAgentM1(self.train_env, self.config)
        
        # Start logging session
        self.logger_system.start_episode(1, self.config.initial_balance)
        
        # M1 optimized training
        training_stats = self.agent.train(total_timesteps)
        
        # Save trained model
        model_path = self.agent.save_model()
        logger.info(f"M1 model saved to {model_path}")
        
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
            'training_info': training_info
        }
    
    def test_agent(self, model_path: str = None, n_episodes: int = 3) -> dict:
        """Test M1 optimized agent"""
        if self.test_env is None:
            raise ValueError("Testing environment not created.")
        
        if self.agent is None:
            if model_path is None:
                raise ValueError("No trained agent available and no model path provided")
            
            # Create agent and load model
            self.agent = ForexSACAgentM1(DummyVecEnv([lambda: self.test_env]), self.config)
            self.agent.load_model(model_path)
        
        logger.info(f"Testing M1 agent for {n_episodes} episodes...")
        
        test_results = []
        
        for episode in range(n_episodes):
            logger.info(f"M1 testing episode {episode + 1}/{n_episodes}")
            
            self.logger_system.start_episode(episode + 1, self.config.initial_balance)
            
            obs, _ = self.test_env.reset()
            total_reward = 0
            step = 0
            done = False
            
            # M1 optimized inference
            with torch.no_grad():
                while not done:
                    action = self.agent.get_action(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.test_env.step(action)
                    
                    self.logger_system.log_step(step, obs, action, reward, info)
                    
                    total_reward += reward
                    step += 1
                    done = terminated or truncated
            
            self.logger_system.end_episode(info)
            
            episode_result = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': info['balance'],
                'total_trades': info['total_trades'],
                'win_rate': info['win_rate'],
                'returns': info['returns'],
                'max_drawdown': info['max_drawdown'],
                'steps': step
            }
            
            test_results.append(episode_result)
            
            logger.info(f"M1 Episode {episode + 1} completed:")
            logger.info(f"  Balance: ${episode_result['final_balance']:.2f}")
            logger.info(f"  Returns: {episode_result['returns']:.1%}")
            logger.info(f"  Trades: {episode_result['total_trades']}")
            logger.info(f"  Win Rate: {episode_result['win_rate']:.3f}")
        
        # Calculate M1 test summary
        returns = [r['returns'] for r in test_results]
        test_summary = {
            'n_episodes': n_episodes,
            'mean_returns': np.mean(returns),
            'best_returns': np.max(returns),
            'worst_returns': np.min(returns),
            'profitable_episodes': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / n_episodes,
            'episodes': test_results
        }
        
        logger.info("=== M1 TEST SUMMARY ===")
        logger.info(f"Profitable Episodes: {test_summary['profitable_episodes']}/{n_episodes}")
        logger.info(f"Mean Returns: {test_summary['mean_returns']:.1%}")
        logger.info(f"Profitability Rate: {test_summary['profitability_rate']:.1%}")
        
        return test_summary
    
    def run_m1_pipeline(self, days_back: int = 30, training_timesteps: int = None,
                        test_episodes: int = 3) -> dict:
        """Run complete M1 optimized pipeline"""
        try:
            logger.info("🚀 STARTING M1 FOREX RL TRADING PIPELINE 🚀")
            
            # Step 1: Fetch recent data
            logger.info("Step 1: Fetching M1 optimized data...")
            if self.fetch_recent_data(days_back) is None:
                return {'success': False, 'error': 'Failed to fetch data'}
            
            # Step 2: Split data
            logger.info("Step 2: Splitting M1 data...")
            self.split_data()
            
            # Step 3: Create environments
            logger.info("Step 3: Creating M1 environments...")
            self.create_environments()
            
            # Step 4: Train agent
            logger.info("Step 4: Training M1 agent...")
            training_results = self.train_agent(training_timesteps)
            
            # Step 5: Test agent
            logger.info("Step 5: Testing M1 agent...")
            test_results = self.test_agent(training_results['model_path'], test_episodes)
            
            # Step 6: Generate report
            logger.info("Step 6: Generating M1 report...")
            report = self.logger_system.generate_performance_report()
            print(report)
            
            logger.info("🎉 M1 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
            
            return {
                'success': True,
                'training_results': training_results,
                'test_results': test_results,
                'model_path': training_results['model_path']
            }
            
        except Exception as e:
            logger.error(f"M1 Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """M1 Main entry point"""
    parser = argparse.ArgumentParser(description='Train M1 Optimized Forex RL Agent')
    parser.add_argument('--instrument', default='EUR_USD', help='Trading instrument')
    parser.add_argument('--days-back', type=int, default=30, help='Days of data to fetch')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--timesteps', type=int, default=50000, help='Training timesteps (M1 optimized)')
    parser.add_argument('--test-episodes', type=int, default=3, help='Test episodes')
    parser.add_argument('--use-mps', action='store_true', default=True, help='Use M1 MPS acceleration')
    
    args = parser.parse_args()
    
    # Check M1 availability
    if torch.backends.mps.is_available() and args.use_mps:
        logger.info("✅ M1 MPS acceleration available and enabled!")
    else:
        logger.info("⚠️  M1 MPS not available or disabled, using CPU")
    
    # Create M1 config
    config = TradingConfigM1()
    config.instrument = args.instrument
    config.initial_balance = args.initial_balance
    config.total_timesteps = args.timesteps
    config.use_mps = args.use_mps
    
    # Create and run M1 pipeline
    pipeline = ForexTradingPipelineM1(config)
    results = pipeline.run_m1_pipeline(
        days_back=args.days_back,
        training_timesteps=args.timesteps,
        test_episodes=args.test_episodes
    )
    
    if results['success']:
        print(f"\n🎉 M1 PIPELINE SUCCESS! 🎉")
        print(f"Model saved to: {results['model_path']}")
        print(f"Check rl_ml/logs/ for detailed reports")
        
        # M1 specific success metrics
        test_results = results['test_results']
        if test_results['profitability_rate'] >= 0.6:
            print("🚀 EXCELLENT M1 performance!")
        elif test_results['profitability_rate'] >= 0.3:
            print("✅ GOOD M1 performance!")
        else:
            print("⚠️  M1 agent needs more training")
    else:
        print(f"\n❌ M1 PIPELINE FAILED")
        print(f"Error: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()