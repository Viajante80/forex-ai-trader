"""
Multi-Agent RL Trainer for Forex Trading
Trains and compares SAC, PPO, TD3, and A2C agents on the same dataset
M1 Mac optimized for all algorithms
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
import json
import time
from typing import Dict, List, Any

from config_m1 import CONFIG_M1, TradingConfigM1
from config_aggressive import CONFIG_AGGRESSIVE
from data_fetcher import OandaDataFetcher
from technical_indicators import TechnicalIndicators
from trading_env import ForexTradingEnv
from sac_agent_m1 import ForexSACAgentM1
from ppo_agent_m1 import ForexPPOAgentM1
from td3_agent_m1 import ForexTD3AgentM1
from a2c_agent_m1 import ForexA2CAgentM1
from trading_logger import TradingLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiAgentTrainer:
    """
    Multi-agent RL trainer for forex trading
    Supports SAC, PPO, TD3, and A2C algorithms
    """
    
    def __init__(self, config: TradingConfigM1 = None, aggressive: bool = False):
        self.config = config or CONFIG_M1
        self.aggressive = aggressive
        
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
        
        # Available algorithms
        self.available_algorithms = ['sac', 'ppo', 'td3', 'a2c']
        self.agent_classes = {
            'sac': ForexSACAgentM1,
            'ppo': ForexPPOAgentM1,
            'td3': ForexTD3AgentM1,
            'a2c': ForexA2CAgentM1
        }
        
        # Data storage
        self.train_data = None
        self.test_data = None
        self.results = {}
        
    def load_data(self) -> bool:
        """Load and prepare training/testing data"""
        logger.info("🏛️  LOADING HISTORICAL DATASET")
        
        # Check for existing enhanced data
        data_files = [f for f in os.listdir("data/") if f.startswith(f"{self.config.instrument}_{self.config.timeframe}_") and f.endswith(".pkl")]
        
        if not data_files:
            logger.error("❌ No data files found in data/ directory")
            return False
        
        # Use the most recent data file
        latest_file = sorted(data_files)[-1]
        data_path = f"data/{latest_file}"
        
        try:
            full_data = pd.read_pickle(data_path)
            logger.info(f"✅ Loaded {len(full_data)} samples from {data_path}")
            logger.info(f"📅 Date range: {full_data.index[0]} to {full_data.index[-1]}")
            
            # Split data (80/20 split for consistency)
            total_samples = len(full_data)
            train_samples = int(total_samples * 0.8)
            
            self.train_data = full_data.iloc[:train_samples].copy()
            self.test_data = full_data.iloc[train_samples:].copy()
            
            logger.info(f"📊 Data split - Training: {len(self.train_data):,} samples, Testing: {len(self.test_data):,} samples")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False
    
    def create_environments(self, algorithm: str) -> tuple:
        """Create training and testing environments for a specific algorithm"""
        # Create base environments
        train_env_base = ForexTradingEnv(self.train_data, self.config)
        test_env_base = ForexTradingEnv(self.test_data, self.config)
        
        # Wrap environments
        if algorithm in ['sac', 'ppo']:  # These typically use vectorized environments
            train_env = DummyVecEnv([lambda: Monitor(train_env_base)])
            test_env = Monitor(test_env_base)
        else:  # TD3, A2C
            train_env = DummyVecEnv([lambda: Monitor(train_env_base)])
            test_env = Monitor(test_env_base)
        
        return train_env, test_env
    
    def train_algorithm(self, algorithm: str, timesteps: int, test_episodes: int = 5) -> Dict[str, Any]:
        """Train a specific algorithm"""
        logger.info(f"🚀 TRAINING {algorithm.upper()} AGENT")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            # Create environments
            train_env, test_env = self.create_environments(algorithm)
            
            # Create agent
            agent_class = self.agent_classes[algorithm]
            agent = agent_class(train_env, self.config)
            
            # Train the agent
            logger.info(f"Training {algorithm.upper()} for {timesteps:,} timesteps...")
            training_stats = agent.train(timesteps)
            
            # Save the model
            config_type = "AGGRESSIVE" if self.aggressive else "STANDARD"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"{algorithm}_forex_{config_type}_m1_{timestamp}.zip"
            model_path = agent.save_model(model_filename)
            
            # Test the agent
            logger.info(f"Testing {algorithm.upper()} for {test_episodes} episodes...")
            test_results = []
            
            for episode in range(test_episodes):
                obs, _ = test_env.reset()
                total_reward = 0
                done = False
                
                with torch.no_grad():
                    while not done:
                        action = agent.get_action(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = test_env.step(action)
                        total_reward += reward
                        done = terminated or truncated
                
                test_results.append({
                    'episode': episode + 1,
                    'final_balance': info['balance'],
                    'total_trades': info['total_trades'],
                    'win_rate': info['win_rate'],
                    'returns': info['returns'],
                    'max_drawdown': info['max_drawdown']
                })
            
            # Calculate metrics
            returns = [r['returns'] for r in test_results]
            balances = [r['final_balance'] for r in test_results]
            win_rates = [r['win_rate'] for r in test_results]
            
            training_time = time.time() - start_time
            
            result_summary = {
                'algorithm': algorithm.upper(),
                'success': True,
                'model_path': model_path,
                'training_timesteps': timesteps,
                'training_time_minutes': training_time / 60,
                'mean_returns': np.mean(returns),
                'std_returns': np.std(returns),
                'best_returns': np.max(returns),
                'worst_returns': np.min(returns),
                'mean_balance': np.mean(balances),
                'mean_win_rate': np.mean(win_rates),
                'profitable_episodes': sum(1 for r in returns if r > 0),
                'profitability_rate': sum(1 for r in returns if r > 0) / test_episodes,
                'test_episodes': test_results
            }
            
            logger.info(f"✅ {algorithm.upper()} training completed!")
            logger.info(f"   Training Time: {training_time/60:.1f} minutes")
            logger.info(f"   Mean Returns: {result_summary['mean_returns']:+.2%}")
            logger.info(f"   Profitability: {result_summary['profitability_rate']:.1%}")
            
            return result_summary
            
        except Exception as e:
            logger.error(f"❌ {algorithm.upper()} training failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'algorithm': algorithm.upper(),
                'success': False,
                'error': str(e),
                'training_time_minutes': (time.time() - start_time) / 60
            }
    
    def train_all_algorithms(self, algorithms: List[str], timesteps: int, test_episodes: int = 5) -> Dict[str, Any]:
        """Train all specified algorithms and compare results"""
        logger.info("🎯 MULTI-AGENT FOREX TRADING COMPARISON")
        logger.info("=" * 80)
        logger.info(f"Algorithms: {', '.join([a.upper() for a in algorithms])}")
        logger.info(f"Timesteps per algorithm: {timesteps:,}")
        logger.info(f"Test episodes: {test_episodes}")
        logger.info(f"Config: {'AGGRESSIVE' if self.aggressive else 'STANDARD'}")
        logger.info("=" * 80)
        
        # Load data
        if not self.load_data():
            return {'success': False, 'error': 'Failed to load data'}
        
        # Train each algorithm
        all_results = {}
        successful_algorithms = []
        
        for algorithm in algorithms:
            if algorithm not in self.available_algorithms:
                logger.warning(f"⚠️  Unknown algorithm: {algorithm}, skipping...")
                continue
            
            result = self.train_algorithm(algorithm, timesteps, test_episodes)
            all_results[algorithm] = result
            
            if result['success']:
                successful_algorithms.append(algorithm)
        
        # Generate comparison
        if successful_algorithms:
            comparison = self.generate_comparison(all_results, successful_algorithms)
            all_results['comparison'] = comparison
            
            # Save complete results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            config_type = "aggressive" if self.aggressive else "standard"
            results_file = f"logs/multi_agent_results_{config_type}_{timestamp}.json"
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"📝 Complete results saved: {results_file}")
            
        return {
            'success': True,
            'algorithms_trained': successful_algorithms,
            'results': all_results
        }
    
    def generate_comparison(self, results: Dict, successful_algorithms: List[str]) -> Dict[str, Any]:
        """Generate performance comparison between algorithms"""
        logger.info("📊 GENERATING ALGORITHM COMPARISON")
        
        comparison_metrics = {}
        ranking = {}
        
        # Extract metrics for comparison
        for algorithm in successful_algorithms:
            result = results[algorithm]
            comparison_metrics[algorithm] = {
                'mean_returns': result['mean_returns'],
                'profitability_rate': result['profitability_rate'],
                'stability': 1.0 / (result['std_returns'] + 0.01),  # Inverse of volatility
                'training_time': result['training_time_minutes'],
                'best_returns': result['best_returns'],
                'worst_returns': result['worst_returns']
            }
        
        # Rank algorithms by different metrics
        for metric in ['mean_returns', 'profitability_rate', 'stability']:
            ranking[metric] = sorted(
                successful_algorithms,
                key=lambda x: comparison_metrics[x][metric],
                reverse=True
            )
        
        # Training speed ranking (fastest first)
        ranking['training_speed'] = sorted(
            successful_algorithms,
            key=lambda x: comparison_metrics[x]['training_time']
        )
        
        # Overall score (weighted combination)
        overall_scores = {}
        for algorithm in successful_algorithms:
            metrics = comparison_metrics[algorithm]
            score = (
                metrics['mean_returns'] * 0.4 +
                metrics['profitability_rate'] * 0.3 +
                metrics['stability'] * 0.2 +
                (1.0 / (metrics['training_time'] / 60 + 0.1)) * 0.1  # Speed bonus
            )
            overall_scores[algorithm] = score
        
        ranking['overall'] = sorted(
            successful_algorithms,
            key=lambda x: overall_scores[x],
            reverse=True
        )
        
        # Generate recommendations
        best_overall = ranking['overall'][0]
        best_returns = ranking['mean_returns'][0]
        best_stability = ranking['stability'][0]
        fastest = ranking['training_speed'][0]
        
        recommendations = {
            'best_overall': best_overall,
            'best_returns': best_returns,
            'most_stable': best_stability,
            'fastest_training': fastest,
            'summary': f"{best_overall.upper()} is the best overall performer"
        }
        
        logger.info("🏆 ALGORITHM RANKING:")
        for i, algorithm in enumerate(ranking['overall'], 1):
            metrics = comparison_metrics[algorithm]
            logger.info(f"   {i}. {algorithm.upper()}: "
                       f"{metrics['mean_returns']:+.2%} returns, "
                       f"{metrics['profitability_rate']:.1%} profitable, "
                       f"{metrics['training_time']:.1f}min training")
        
        return {
            'comparison_metrics': comparison_metrics,
            'ranking': ranking,
            'recommendations': recommendations,
            'overall_scores': overall_scores
        }

def main():
    """Main entry point for multi-agent training"""
    parser = argparse.ArgumentParser(description='Multi-Agent Forex RL Trading Comparison')
    parser.add_argument('--algorithms', nargs='+', default=['sac', 'ppo', 'td3', 'a2c'],
                       choices=['sac', 'ppo', 'td3', 'a2c'],
                       help='Algorithms to train and compare')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training timesteps per algorithm (default: 500K)')
    parser.add_argument('--test-episodes', type=int, default=5,
                       help='Test episodes per algorithm (default: 5)')
    parser.add_argument('--aggressive', action='store_true',
                       help='Use aggressive trading configuration')
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                       help='Initial balance (default: $1000)')
    
    args = parser.parse_args()
    
    # M1 check
    if torch.backends.mps.is_available():
        print("🚀 M1 Mac GPU detected - optimal performance!")
    else:
        print("⚠️  CPU training - will be slower")
    
    config_type = "AGGRESSIVE" if args.aggressive else "STANDARD"
    
    print("🎯 MULTI-AGENT FOREX RL COMPARISON")
    print("=" * 80)
    print(f"🧠 Algorithms: {', '.join([a.upper() for a in args.algorithms])}")
    print(f"📊 Timesteps: {args.timesteps:,} per algorithm")
    print(f"🧪 Test Episodes: {args.test_episodes} per algorithm")
    print(f"💰 Balance: ${args.initial_balance:,.2f}")
    print(f"⚙️  Config: {config_type}")
    print(f"⏰ Est. Total Time: {len(args.algorithms) * args.timesteps/200000:.1f} hours")
    print("=" * 80)
    
    # Create config
    config = TradingConfigM1()
    config.initial_balance = args.initial_balance
    config.total_timesteps = args.timesteps
    
    # Create and run trainer
    trainer = MultiAgentTrainer(config, aggressive=args.aggressive)
    results = trainer.train_all_algorithms(args.algorithms, args.timesteps, args.test_episodes)
    
    if results['success']:
        print(f"\n🎉 MULTI-AGENT TRAINING COMPLETED! 🎉")
        print(f"✅ Successfully trained: {', '.join([a.upper() for a in results['algorithms_trained']])}")
        
        if 'comparison' in results['results']:
            comparison = results['results']['comparison']
            recommendations = comparison['recommendations']
            print(f"\n🏆 RECOMMENDATIONS:")
            print(f"   🥇 Best Overall: {recommendations['best_overall'].upper()}")
            print(f"   💰 Best Returns: {recommendations['best_returns'].upper()}")
            print(f"   📈 Most Stable: {recommendations['most_stable'].upper()}")
            print(f"   ⚡ Fastest Training: {recommendations['fastest_training'].upper()}")
        
    else:
        print(f"\n❌ TRAINING FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()