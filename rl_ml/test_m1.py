"""
M1 Mac Optimized Testing Script for trained RL Forex Trading Agent
"""
import argparse
import os
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import torch
from stable_baselines3.common.monitor import Monitor

from config_m1 import CONFIG_M1, TradingConfigM1
from data_fetcher import OandaDataFetcher
from technical_indicators import TechnicalIndicators
from trading_env import ForexTradingEnv
from sac_agent_m1 import ForexSACAgentM1
from trading_logger import TradingLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexTesterM1:
    """M1 Mac optimized Forex tester"""
    
    def __init__(self, model_path: str, config: TradingConfigM1 = None):
        self.config = config or CONFIG_M1
        self.model_path = model_path
        self.data_fetcher = OandaDataFetcher(self.config)
        self.indicators = TechnicalIndicators(self.config)
        self.logger_system = TradingLogger(self.config)
        
        # M1 optimizations
        if torch.backends.mps.is_available():
            logger.info("🚀 M1 Mac testing optimizations enabled")
        else:
            logger.info("⚠️  Running M1 test on non-M1 system")
        
        self.agent = None
        
    def load_test_data(self, days_back: int = 7) -> pd.DataFrame:
        """Load M1 optimized test data"""
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Loading M1 test data from {start_str} to {end_str}")
        
        # Try cached data first
        cache_file = f"rl_ml/data/{self.config.instrument}_{self.config.timeframe}_m1_test.pkl"
        
        if os.path.exists(cache_file):
            try:
                cached_data = pd.read_pickle(cache_file)
                if cached_data.index[-1] > pd.Timestamp.now(tz='UTC') - timedelta(hours=12):
                    logger.info(f"Using cached M1 test data: {len(cached_data)} samples")
                    return cached_data
            except Exception as e:
                logger.warning(f"Failed to load M1 cache: {e}")
        
        # Fetch fresh data
        raw_data = self.data_fetcher.fetch_historical_data(start_str, end_str)
        if raw_data is None:
            raise ValueError("Failed to fetch M1 test data")
        
        logger.info("Adding M1 optimized technical indicators...")
        test_data = self.indicators.add_all_indicators(raw_data)
        
        # Cache for future use
        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            test_data.to_pickle(cache_file)
            logger.info(f"M1 test data cached")
        except Exception as e:
            logger.warning(f"Failed to cache M1 test data: {e}")
        
        logger.info(f"M1 test data prepared: {len(test_data)} samples")
        return test_data
    
    def test_m1_model(self, test_data: pd.DataFrame, n_episodes: int = 5, 
                     deterministic: bool = True) -> dict:
        """Test M1 model with optimizations"""
        
        logger.info(f"Testing M1 model {self.model_path} for {n_episodes} episodes")
        
        # Create M1 test environment
        test_env = ForexTradingEnv(test_data, self.config)
        test_env = Monitor(test_env)
        
        # Load M1 agent
        if self.agent is None:
            from stable_baselines3.common.vec_env import DummyVecEnv
            dummy_env = DummyVecEnv([lambda: test_env])
            self.agent = ForexSACAgentM1(dummy_env, self.config)
            self.agent.load_model(self.model_path)
        
        test_results = []
        
        for episode in range(n_episodes):
            logger.info(f"M1 testing episode {episode + 1}/{n_episodes}")
            
            self.logger_system.start_episode(episode + 1, self.config.initial_balance)
            
            obs, _ = test_env.reset()
            total_reward = 0
            step = 0
            done = False
            
            # M1 optimized inference loop
            with torch.no_grad():
                while not done:
                    action = self.agent.get_action(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    
                    self.logger_system.log_step(step, obs, action, reward, info)
                    
                    total_reward += reward
                    step += 1
                    done = terminated or truncated
                    
                    # M1 memory optimization
                    if step % 500 == 0:
                        logger.debug(f"M1 Step {step}: Balance=${info['balance']:.2f}")
            
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
            
            # M1 episode summary
            logger.info(f"M1 Episode {episode + 1} Results:")
            logger.info(f"  💰 Final Balance: ${episode_result['final_balance']:.2f}")
            logger.info(f"  📈 Returns: {episode_result['returns']:.1%}")
            logger.info(f"  🎯 Win Rate: {episode_result['win_rate']:.3f}")
            logger.info(f"  📊 Trades: {episode_result['total_trades']}")
            logger.info(f"  ⚠️  Max Drawdown: {episode_result['max_drawdown']:.3f}")
            
            # Check M1 target achievement
            if episode_result['returns'] >= (self.config.target_multiplier - 1):
                logger.info("🎉 M1 TARGET REACHED!")
        
        # Calculate M1 summary statistics
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
    
    def print_m1_summary(self, summary: dict):
        """Print M1 optimized test summary"""
        print("\n" + "🚀" + "="*58 + "🚀")
        print("      M1 MAC FOREX RL AGENT TEST SUMMARY")
        print("🚀" + "="*58 + "🚀")
        print(f"Model: {summary['model_path']}")
        print(f"Test Period: {summary['test_period']}")
        print(f"Episodes: {summary['n_episodes']}")
        print()
        print("📊 PERFORMANCE METRICS:")
        print(f"  Mean Returns: {summary['mean_returns']:.1%}")
        print(f"  Best Returns: {summary['best_returns']:.1%}")
        print(f"  Worst Returns: {summary['worst_returns']:.1%}")
        print(f"  Returns Std Dev: {summary['std_returns']:.3f}")
        print()
        print("💰 BALANCE STATISTICS:")
        print(f"  Initial Balance: ${self.config.initial_balance:.2f}")
        print(f"  Mean Final Balance: ${summary['mean_balance']:.2f}")
        print(f"  Best Final Balance: ${summary['best_balance']:.2f}")
        print(f"  Worst Final Balance: ${summary['worst_balance']:.2f}")
        print()
        print("🎯 TRADING STATISTICS:")
        print(f"  Mean Win Rate: {summary['mean_win_rate']:.3f}")
        print(f"  Mean Drawdown: {summary['mean_drawdown']:.3f}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.3f}")
        print()
        print("🏆 SUCCESS RATES:")
        print(f"  Profitable Episodes: {summary['profitable_episodes']}/{summary['n_episodes']}")
        print(f"  Profitability Rate: {summary['profitability_rate']:.1%}")
        print(f"  Target Reached ({self.config.target_multiplier}x): {summary['target_reached_episodes']}/{summary['n_episodes']}")
        print(f"  Target Success Rate: {summary['target_success_rate']:.1%}")
        print("🚀" + "="*58 + "🚀")
        
        # M1 Assessment
        print("\n🔍 M1 ASSESSMENT:")
        if summary['profitability_rate'] >= 0.7:
            print("🏆 EXCELLENT: Outstanding M1 performance!")
        elif summary['profitability_rate'] >= 0.5:
            print("✅ GOOD: Solid M1 performance!")
        else:
            print("⚠️  NEEDS WORK: M1 agent requires more training")
            
        if summary['target_success_rate'] >= 0.3:
            print("🎯 EXCELLENT: Frequently reaches target on M1!")
        elif summary['target_success_rate'] >= 0.1:
            print("📈 GOOD: Sometimes reaches target on M1!")
        else:
            print("📉 POOR: Rarely reaches target - increase training")
            
        if summary['mean_drawdown'] <= 0.1:
            print("🛡️  EXCELLENT: Low risk M1 profile!")
        elif summary['mean_drawdown'] <= 0.2:
            print("⚖️  GOOD: Moderate risk M1 profile!")
        else:
            print("⚠️  HIGH RISK: Consider risk management improvements")
        
        # M1 specific recommendations
        print("\n💡 M1 OPTIMIZATION NOTES:")
        if torch.backends.mps.is_available():
            print("✅ M1 MPS acceleration active")
        else:
            print("⚠️  M1 MPS not detected - check PyTorch installation")

def main():
    """M1 Main testing entry point"""
    parser = argparse.ArgumentParser(description='Test M1 trained Forex RL agent')
    parser.add_argument('model_path', help='Path to M1 trained model')
    parser.add_argument('--episodes', type=int, default=5, help='Number of test episodes')
    parser.add_argument('--days-back', type=int, default=7, help='Days of test data')
    parser.add_argument('--instrument', default='EUR_USD', help='Trading instrument')
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Initial balance')
    parser.add_argument('--stochastic', action='store_true', help='Use stochastic actions')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        logger.error(f"M1 model file not found: {args.model_path}")
        return
    
    # Check M1 availability
    if torch.backends.mps.is_available():
        logger.info("🚀 M1 MPS available for testing!")
    else:
        logger.info("⚠️  M1 MPS not available")
    
    # Create M1 config
    config = TradingConfigM1()
    config.instrument = args.instrument
    config.initial_balance = args.initial_balance
    
    # Create M1 tester
    tester = ForexTesterM1(args.model_path, config)
    
    try:
        # Load M1 test data
        test_data = tester.load_test_data(args.days_back)
        
        # Run M1 tests
        results = tester.test_m1_model(
            test_data, 
            n_episodes=args.episodes,
            deterministic=not args.stochastic
        )
        
        # Print M1 summary
        tester.print_m1_summary(results)
        
        # Generate M1 report
        report = tester.logger_system.generate_performance_report()
        print("\n" + report)
        
    except Exception as e:
        logger.error(f"M1 testing failed: {e}")
        raise

if __name__ == "__main__":
    main()