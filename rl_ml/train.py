"""
Main Training Pipeline for RL Forex Trading Agent
Fetches data, trains SAC agent (2016-2024), and tests (2024-2025)
"""
import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from config import CONFIG, TradingConfig
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

class ForexTradingPipeline:
    """
    Complete training and testing pipeline for Forex RL agent
    """
    
    def __init__(self, config: TradingConfig = None):
        self.config = config or CONFIG
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
        
    def fetch_or_load_data(self, force_fetch: bool = False) -> pd.DataFrame:
        """
        Fetch historical data or load from cache
        """
        data_file = f"rl_ml/data/{self.config.instrument}_{self.config.timeframe}_full_data.pkl"
        
        if not force_fetch and os.path.exists(data_file):
            logger.info("Loading cached data...")
            self.full_data = self.data_fetcher.load_data(data_file)
        else:
            logger.info("Fetching fresh data from Oanda...")
            self.full_data = self.data_fetcher.fetch_historical_data()
            
            if self.full_data is not None:
                logger.info("Adding technical indicators...")
                self.full_data = self.indicators.add_all_indicators(self.full_data)
                
                # Save processed data
                self.data_fetcher.save_data(self.full_data, data_file)
                logger.info(f"Processed data saved to {data_file}")
            else:
                logger.error("Failed to fetch data")
                return None
        
        if self.full_data is None:
            logger.error("No data available")
            return None
        
        logger.info(f"Total data points: {len(self.full_data)}")
        logger.info(f"Date range: {self.full_data.index[0]} to {self.full_data.index[-1]}")
        logger.info(f"Features: {len(self.full_data.columns)}")
        
        return self.full_data
    
    def split_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into training (2016-2024) and testing (2024-2025) sets
        """
        if self.full_data is None:
            raise ValueError("No data available. Call fetch_or_load_data() first.")
        
        train_end_date = pd.to_datetime(self.config.train_end_date).tz_localize('UTC')
        
        self.train_data = self.full_data[self.full_data.index < train_end_date].copy()
        self.test_data = self.full_data[self.full_data.index >= train_end_date].copy()
        
        logger.info(f"Training data: {len(self.train_data)} samples ({self.train_data.index[0]} to {self.train_data.index[-1]})")
        logger.info(f"Testing data: {len(self.test_data)} samples ({self.test_data.index[0]} to {self.test_data.index[-1]})")
        
        return self.train_data, self.test_data
    
    def create_environments(self):
        """
        Create training and testing environments
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not split. Call split_data() first.")
        
        # Create training environment
        logger.info("Creating training environment...")
        train_env_base = ForexTradingEnv(self.train_data, self.config)
        train_env_monitored = Monitor(train_env_base)
        self.train_env = DummyVecEnv([lambda: train_env_monitored])
        
        # Create testing environment
        logger.info("Creating testing environment...")
        test_env_base = ForexTradingEnv(self.test_data, self.config)
        self.test_env = Monitor(test_env_base)
        
        logger.info("Environments created successfully")
    
    def train_agent(self, total_timesteps: int = None) -> dict:
        """
        Train the SAC agent
        """
        if self.train_env is None:
            raise ValueError("Training environment not created. Call create_environments() first.")
        
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        logger.info(f"Starting agent training for {total_timesteps} timesteps...")
        
        # Create agent
        self.agent = ForexSACAgent(self.train_env, self.config)
        
        # Start logging session
        self.logger_system.start_episode(1, self.config.initial_balance)
        
        # Train agent
        training_stats = self.agent.train(total_timesteps)
        
        # Save trained model
        model_path = self.agent.save_model()
        logger.info(f"Model saved to {model_path}")
        
        # Log training completion
        training_info = {
            'balance': training_stats.get('final_performance', {}).get('final_balance', self.config.initial_balance),
            'total_trades': training_stats.get('final_performance', {}).get('total_trades', 0),
            'winning_trades': 0,  # Will be calculated from trade history
            'losing_trades': 0,   # Will be calculated from trade history
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
    
    def test_agent(self, model_path: str = None, n_episodes: int = 5) -> dict:
        """
        Test the trained agent
        """
        if self.test_env is None:
            raise ValueError("Testing environment not created. Call create_environments() first.")
        
        if self.agent is None:
            if model_path is None:
                raise ValueError("No trained agent available and no model path provided")
            
            # Create agent and load model
            self.agent = ForexSACAgent(DummyVecEnv([lambda: self.test_env]), self.config)
            self.agent.load_model(model_path)
        
        logger.info(f"Testing agent for {n_episodes} episodes...")
        
        test_results = []
        
        for episode in range(n_episodes):
            logger.info(f"Testing episode {episode + 1}/{n_episodes}")
            
            # Start episode logging
            self.logger_system.start_episode(episode + 1, self.config.initial_balance)
            
            obs, _ = self.test_env.reset()
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                # Get action from agent
                action = self.agent.get_action(obs, deterministic=True)
                
                # Execute action
                obs, reward, terminated, truncated, info = self.test_env.step(action)
                
                # Log step
                self.logger_system.log_step(step, obs, action, reward, info)
                
                total_reward += reward
                step += 1
                done = terminated or truncated
                
                # Log trades if any new ones occurred
                # (This would need to be implemented in the environment to track individual trades)
            
            # End episode logging
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
            
            logger.info(f"Episode {episode + 1} completed:")
            logger.info(f"  Final Balance: ${episode_result['final_balance']:.2f}")
            logger.info(f"  Returns: {episode_result['returns']:.3f}")
            logger.info(f"  Win Rate: {episode_result['win_rate']:.3f}")
            logger.info(f"  Total Trades: {episode_result['total_trades']}")
        
        # Calculate aggregate test statistics
        test_summary = {
            'n_episodes': n_episodes,
            'mean_final_balance': np.mean([r['final_balance'] for r in test_results]),
            'std_final_balance': np.std([r['final_balance'] for r in test_results]),
            'mean_returns': np.mean([r['returns'] for r in test_results]),
            'best_returns': np.max([r['returns'] for r in test_results]),
            'worst_returns': np.min([r['returns'] for r in test_results]),
            'mean_win_rate': np.mean([r['win_rate'] for r in test_results]),
            'mean_trades': np.mean([r['total_trades'] for r in test_results]),
            'profitable_episodes': sum(1 for r in test_results if r['returns'] > 0),
            'profitability_rate': sum(1 for r in test_results if r['returns'] > 0) / n_episodes,
            'episodes': test_results
        }
        
        logger.info("=== TEST SUMMARY ===")
        logger.info(f"Profitable Episodes: {test_summary['profitable_episodes']}/{n_episodes}")
        logger.info(f"Profitability Rate: {test_summary['profitability_rate']:.3f}")
        logger.info(f"Mean Returns: {test_summary['mean_returns']:.3f}")
        logger.info(f"Mean Final Balance: ${test_summary['mean_final_balance']:.2f}")
        logger.info(f"Mean Win Rate: {test_summary['mean_win_rate']:.3f}")
        
        return test_summary
    
    def generate_report(self):
        """
        Generate comprehensive performance report
        """
        report = self.logger_system.generate_performance_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"rl_ml/logs/final_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Final report saved to {report_file}")
        print(report)
        
        # Generate charts
        try:
            self.logger_system.create_performance_charts()
            self.logger_system.export_trades_csv()
        except Exception as e:
            logger.warning(f"Failed to create charts: {e}")
    
    def run_full_pipeline(self, force_fetch: bool = False, 
                         training_timesteps: int = None,
                         test_episodes: int = 5) -> dict:
        """
        Run the complete training and testing pipeline
        """
        try:
            logger.info("=== STARTING FOREX RL TRADING PIPELINE ===")
            
            # Step 1: Fetch/Load Data
            logger.info("Step 1: Fetching data...")
            if self.fetch_or_load_data(force_fetch) is None:
                return {'success': False, 'error': 'Failed to fetch data'}
            
            # Step 2: Split Data
            logger.info("Step 2: Splitting data...")
            self.split_data()
            
            # Step 3: Create Environments
            logger.info("Step 3: Creating environments...")
            self.create_environments()
            
            # Step 4: Train Agent
            logger.info("Step 4: Training agent...")
            training_results = self.train_agent(training_timesteps)
            
            # Step 5: Test Agent
            logger.info("Step 5: Testing agent...")
            test_results = self.test_agent(training_results['model_path'], test_episodes)
            
            # Step 6: Generate Report
            logger.info("Step 6: Generating report...")
            self.generate_report()
            
            logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            
            return {
                'success': True,
                'training_results': training_results,
                'test_results': test_results,
                'model_path': training_results['model_path']
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'success': False, 'error': str(e)}

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Train Forex RL Trading Agent')
    parser.add_argument('--instrument', default='EUR_USD', help='Trading instrument')
    parser.add_argument('--timeframe', default='M5', help='Timeframe')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial account balance')
    parser.add_argument('--force-fetch', action='store_true', help='Force fetch fresh data')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Training timesteps')
    parser.add_argument('--test-episodes', type=int, default=5, help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Create custom config
    config = TradingConfig()
    config.instrument = args.instrument
    config.timeframe = args.timeframe
    config.initial_balance = args.initial_balance
    config.total_timesteps = args.timesteps
    
    # Create and run pipeline
    pipeline = ForexTradingPipeline(config)
    results = pipeline.run_full_pipeline(
        force_fetch=args.force_fetch,
        training_timesteps=args.timesteps,
        test_episodes=args.test_episodes
    )
    
    if results['success']:
        print(f"\n=== PIPELINE SUCCESS ===")
        print(f"Model saved to: {results['model_path']}")
        print(f"Check rl_ml/logs/ for detailed reports and charts")
    else:
        print(f"\n=== PIPELINE FAILED ===")
        print(f"Error: {results['error']}")
        sys.exit(1)

if __name__ == "__main__":
    main()