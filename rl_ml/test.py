"""
Testing script for trained RL Forex Trading Agent
Load trained model and test on different time periods
"""
import argparse
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from stable_baselines3.common.monitor import Monitor

from config import CONFIG, TradingConfig
from data_fetcher import OandaDataFetcher
from technical_indicators import TechnicalIndicators
from trading_env import ForexTradingEnv
from sac_agent import ForexSACAgent
from trading_logger import TradingLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexTester:
    """
    Test trained Forex RL agent
    """
    
    def __init__(self, model_path: str, config: TradingConfig = None):
        self.config = config or CONFIG
        self.model_path = model_path
        self.data_fetcher = OandaDataFetcher(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.logger_system = TradingLogger(self.config)
        
        # Load agent
        self.agent = None
        
    def load_test_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Load and prepare test data
        """
        if start_date is None:
            start_date = self.config.train_end_date
        if end_date is None:
            end_date = self.config.test_end_date
            
        logger.info(f"Loading test data from {start_date} to {end_date}")
        
        # Try to load cached data first
        data_file = f"rl_ml/data/{self.config.instrument}_{self.config.timeframe}_full_data.pkl"
        
        if os.path.exists(data_file):
            full_data = self.data_fetcher.load_data(data_file)
            
            # Filter for test period
            start_dt = pd.to_datetime(start_date).tz_localize('UTC')
            end_dt = pd.to_datetime(end_date).tz_localize('UTC')
            
            test_data = full_data[(full_data.index >= start_dt) & (full_data.index < end_dt)].copy()
            
            if len(test_data) == 0:
                logger.warning("No data found in cached file for test period, fetching fresh data...")
                return self._fetch_fresh_test_data(start_date, end_date)
                
            logger.info(f"Loaded {len(test_data)} test samples from cache")
            return test_data
        else:
            return self._fetch_fresh_test_data(start_date, end_date)
    
    def _fetch_fresh_test_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch fresh test data from Oanda
        """
        logger.info("Fetching fresh test data from Oanda...")
        
        raw_data = self.data_fetcher.fetch_historical_data(start_date, end_date)
        if raw_data is None:
            raise ValueError("Failed to fetch test data")
        
        logger.info("Adding technical indicators to test data...")
        test_data = self.indicators.add_all_indicators(raw_data)
        
        logger.info(f"Prepared {len(test_data)} test samples")
        return test_data
    
    def test_model(self, test_data: pd.DataFrame, n_episodes: int = 10, 
                   deterministic: bool = True) -> dict:
        """
        Test the model on given data
        """
        logger.info(f"Testing model {self.model_path} for {n_episodes} episodes")
        
        # Create test environment
        test_env = ForexTradingEnv(test_data, self.config)
        test_env = Monitor(test_env)
        
        # Load trained agent
        if self.agent is None:
            from stable_baselines3.common.vec_env import DummyVecEnv
            dummy_env = DummyVecEnv([lambda: test_env])
            self.agent = ForexSACAgent(dummy_env, self.config)
            self.agent.load_model(self.model_path)
        
        test_results = []
        
        for episode in range(n_episodes):
            logger.info(f"Testing episode {episode + 1}/{n_episodes}")
            
            # Start episode logging
            self.logger_system.start_episode(episode + 1, self.config.initial_balance)
            
            obs, _ = test_env.reset()
            total_reward = 0
            step = 0
            done = False
            episode_trades = []
            
            while not done:
                # Get action from agent
                action = self.agent.get_action(obs, deterministic=deterministic)
                
                # Execute action
                obs, reward, terminated, truncated, info = test_env.step(action)
                
                # Log step
                self.logger_system.log_step(step, obs, action, reward, info)
                
                total_reward += reward
                step += 1
                done = terminated or truncated
                
                # Track significant events
                if step % 1000 == 0:
                    logger.debug(f"Step {step}: Balance=${info['balance']:.2f}, "
                               f"Trades={info['total_trades']}, WinRate={info['win_rate']:.3f}")
            
            # End episode logging
            self.logger_system.end_episode(info)
            
            episode_result = {
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': info['balance'],
                'total_trades': info['total_trades'],
                'winning_trades': info['winning_trades'],
                'losing_trades': info['losing_trades'],
                'win_rate': info['win_rate'],
                'returns': info['returns'],
                'max_drawdown': info['max_drawdown'],
                'total_pnl': info['total_pnl'],
                'steps': step,
                'peak_balance': info['peak_balance']
            }
            
            test_results.append(episode_result)
            
            # Log episode summary
            logger.info(f"Episode {episode + 1} Results:")
            logger.info(f"  Final Balance: ${episode_result['final_balance']:.2f}")
            logger.info(f"  Returns: {episode_result['returns']:.3f} ({episode_result['returns']*100:.1f}%)")
            logger.info(f"  Total Trades: {episode_result['total_trades']}")
            logger.info(f"  Win Rate: {episode_result['win_rate']:.3f}")
            logger.info(f"  Max Drawdown: {episode_result['max_drawdown']:.3f}")
            
            # Check if target reached
            if episode_result['returns'] >= (self.config.target_multiplier - 1):
                logger.info(f"🎉 TARGET REACHED! Account grew by {episode_result['returns']*100:.1f}%")
        
        # Calculate summary statistics
        returns = [r['returns'] for r in test_results]
        balances = [r['final_balance'] for r in test_results]
        win_rates = [r['win_rate'] for r in test_results]
        drawdowns = [r['max_drawdown'] for r in test_results]
        
        summary = {
            'n_episodes': n_episodes,
            'model_path': self.model_path,
            'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
            'mean_returns': np.mean(returns),
            'std_returns': np.std(returns),
            'best_returns': np.max(returns),
            'worst_returns': np.min(returns),
            'mean_balance': np.mean(balances),
            'best_balance': np.max(balances),
            'worst_balance': np.min(balances),
            'mean_win_rate': np.mean(win_rates),
            'mean_drawdown': np.mean(drawdowns),
            'max_drawdown': np.max(drawdowns),
            'profitable_episodes': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / n_episodes,
            'target_reached_episodes': sum(1 for r in returns if r >= (self.config.target_multiplier - 1)),
            'target_success_rate': sum(1 for r in returns if r >= (self.config.target_multiplier - 1)) / n_episodes,
            'episodes': test_results
        }
        
        return summary
    
    def print_summary(self, summary: dict):
        """
        Print test summary
        """
        print("\n" + "="*60)
        print("         FOREX RL AGENT TEST SUMMARY")
        print("="*60)
        print(f"Model: {summary['model_path']}")
        print(f"Test Period: {summary['test_period']}")
        print(f"Episodes: {summary['n_episodes']}")
        print()
        print("PERFORMANCE METRICS:")
        print(f"  Mean Returns: {summary['mean_returns']:.3f} ({summary['mean_returns']*100:.1f}%)")
        print(f"  Best Returns: {summary['best_returns']:.3f} ({summary['best_returns']*100:.1f}%)")
        print(f"  Worst Returns: {summary['worst_returns']:.3f} ({summary['worst_returns']*100:.1f}%)")
        print(f"  Returns Std Dev: {summary['std_returns']:.3f}")
        print()
        print("BALANCE STATISTICS:")
        print(f"  Initial Balance: ${self.config.initial_balance:.2f}")
        print(f"  Mean Final Balance: ${summary['mean_balance']:.2f}")
        print(f"  Best Final Balance: ${summary['best_balance']:.2f}")
        print(f"  Worst Final Balance: ${summary['worst_balance']:.2f}")
        print()
        print("TRADING STATISTICS:")
        print(f"  Mean Win Rate: {summary['mean_win_rate']:.3f}")
        print(f"  Mean Drawdown: {summary['mean_drawdown']:.3f}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.3f}")
        print()
        print("SUCCESS RATES:")
        print(f"  Profitable Episodes: {summary['profitable_episodes']}/{summary['n_episodes']}")
        print(f"  Profitability Rate: {summary['profitability_rate']:.3f}")
        print(f"  Target Reached ({self.config.target_multiplier}x): {summary['target_reached_episodes']}/{summary['n_episodes']}")
        print(f"  Target Success Rate: {summary['target_success_rate']:.3f}")
        print("="*60)
        
        # Assessment
        print("\nASSESSMENT:")
        if summary['profitability_rate'] >= 0.7:
            print("✅ EXCELLENT: High profitability rate")
        elif summary['profitability_rate'] >= 0.5:
            print("✅ GOOD: Decent profitability rate")
        else:
            print("❌ POOR: Low profitability rate")
            
        if summary['target_success_rate'] >= 0.3:
            print("✅ EXCELLENT: Frequently reaches target")
        elif summary['target_success_rate'] >= 0.1:
            print("✅ GOOD: Sometimes reaches target")
        else:
            print("❌ POOR: Rarely reaches target")
            
        if summary['mean_drawdown'] <= 0.1:
            print("✅ EXCELLENT: Low risk profile")
        elif summary['mean_drawdown'] <= 0.2:
            print("✅ GOOD: Moderate risk profile")
        else:
            print("❌ POOR: High risk profile")

def main():
    """Main entry point for testing"""
    parser = argparse.ArgumentParser(description='Test trained Forex RL agent')
    parser.add_argument('model_path', help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10, help='Number of test episodes')
    parser.add_argument('--instrument', default='EUR_USD', help='Trading instrument')
    parser.add_argument('--timeframe', default='M5', help='Timeframe')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--start-date', help='Test start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Test end date (YYYY-MM-DD)')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions instead of deterministic')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    # Create config
    config = TradingConfig()
    config.instrument = args.instrument
    config.timeframe = args.timeframe
    config.initial_balance = args.initial_balance
    
    # Create tester
    tester = ForexTester(args.model_path, config)
    
    try:
        # Load test data
        test_data = tester.load_test_data(args.start_date, args.end_date)
        
        # Run tests
        results = tester.test_model(
            test_data, 
            n_episodes=args.episodes,
            deterministic=not args.stochastic
        )
        
        # Print summary
        tester.print_summary(results)
        
        # Generate detailed report
        tester.logger_system.generate_performance_report()
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    main()