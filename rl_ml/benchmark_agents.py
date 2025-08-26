"""
Benchmark and Compare Existing RL Models
Loads saved models and compares their performance on the same test dataset
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
from stable_baselines3 import SAC, PPO, TD3, A2C
import json
import glob
from typing import Dict, List, Any

from config_m1 import CONFIG_M1, TradingConfigM1
from trading_env import ForexTradingEnv
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AgentBenchmark:
    """
    Benchmark existing trained models
    """
    
    def __init__(self, config: TradingConfigM1 = None):
        self.config = config or CONFIG_M1
        self.test_data = None
        self.models = {}
        self.results = {}
        
        # Algorithm to class mapping
        self.algorithm_classes = {
            'sac': SAC,
            'ppo': PPO,
            'td3': TD3,
            'a2c': A2C
        }
    
    def load_test_data(self) -> bool:
        """Load test data for benchmarking"""
        logger.info("🔍 Loading test data...")
        
        # Find the most recent data file
        data_files = [f for f in os.listdir("data/") if f.startswith(f"{self.config.instrument}_{self.config.timeframe}_") and f.endswith(".pkl")]
        
        if not data_files:
            logger.error("❌ No data files found in data/ directory")
            return False
        
        latest_file = sorted(data_files)[-1]
        data_path = f"data/{latest_file}"
        
        try:
            full_data = pd.read_pickle(data_path)
            
            # Use last 20% as test data (same split as training)
            total_samples = len(full_data)
            test_start = int(total_samples * 0.8)
            self.test_data = full_data.iloc[test_start:].copy()
            
            logger.info(f"✅ Loaded test data: {len(self.test_data):,} samples")
            logger.info(f"📅 Test period: {self.test_data.index[0]} to {self.test_data.index[-1]}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load test data: {e}")
            return False
    
    def discover_models(self) -> Dict[str, List[str]]:
        """Discover all available trained models"""
        logger.info("🔍 Discovering trained models...")
        
        model_dir = "models"
        if not os.path.exists(model_dir):
            logger.warning("⚠️  Models directory not found")
            return {}
        
        discovered_models = {
            'sac': [],
            'ppo': [],
            'td3': [],
            'a2c': []
        }
        
        # Search for model files
        for algorithm in discovered_models.keys():
            pattern = f"{model_dir}/{algorithm}_forex_*_m1_*.zip"
            model_files = glob.glob(pattern)
            discovered_models[algorithm] = model_files
        
        # Log discovered models
        total_models = sum(len(models) for models in discovered_models.values())
        logger.info(f"📊 Discovered {total_models} models:")
        
        for algorithm, models in discovered_models.items():
            if models:
                logger.info(f"   {algorithm.upper()}: {len(models)} models")
                for model in models:
                    logger.info(f"     - {os.path.basename(model)}")
        
        return discovered_models
    
    def load_model(self, algorithm: str, model_path: str):
        """Load a specific model"""
        try:
            # Create dummy environment for model loading
            dummy_data = self.test_data.iloc[:100].copy()
            dummy_env = ForexTradingEnv(dummy_data, self.config)
            dummy_env = Monitor(dummy_env)
            
            # Load the model
            model_class = self.algorithm_classes[algorithm]
            model = model_class.load(model_path, env=dummy_env)
            
            logger.info(f"✅ Loaded {algorithm.upper()} model: {os.path.basename(model_path)}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load {algorithm.upper()} model {model_path}: {e}")
            return None
    
    def benchmark_model(self, algorithm: str, model_path: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Benchmark a single model"""
        logger.info(f"🧪 Benchmarking {algorithm.upper()}: {os.path.basename(model_path)}")
        
        # Load model
        model = self.load_model(algorithm, model_path)
        if model is None:
            return {'success': False, 'error': 'Failed to load model'}
        
        # Create test environment
        test_env = ForexTradingEnv(self.test_data, self.config)
        test_env = Monitor(test_env)
        
        # Run benchmark episodes
        episode_results = []
        
        for episode in range(n_episodes):
            obs, _ = test_env.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            with torch.no_grad():
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = test_env.step(action)
                    total_reward += reward
                    step_count += 1
                    done = terminated or truncated
            
            episode_results.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': info['balance'],
                'total_trades': info['total_trades'],
                'win_rate': info['win_rate'],
                'returns': info['returns'],
                'max_drawdown': info['max_drawdown'],
                'steps': step_count
            })
        
        # Calculate statistics
        returns = [r['returns'] for r in episode_results]
        balances = [r['final_balance'] for r in episode_results]
        win_rates = [r['win_rate'] for r in episode_results]
        trades = [r['total_trades'] for r in episode_results]
        
        benchmark_result = {
            'success': True,
            'algorithm': algorithm.upper(),
            'model_path': model_path,
            'model_name': os.path.basename(model_path),
            'n_episodes': n_episodes,
            'mean_returns': np.mean(returns),
            'std_returns': np.std(returns),
            'best_returns': np.max(returns),
            'worst_returns': np.min(returns),
            'mean_balance': np.mean(balances),
            'best_balance': np.max(balances),
            'worst_balance': np.min(balances),
            'mean_win_rate': np.mean(win_rates),
            'mean_trades': np.mean(trades),
            'profitable_episodes': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / n_episodes,
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-6),
            'episode_details': episode_results
        }
        
        logger.info(f"📊 {algorithm.upper()} Results: "
                   f"{benchmark_result['mean_returns']:+.2%} returns, "
                   f"{benchmark_result['profitability_rate']:.1%} profitable")
        
        return benchmark_result
    
    def run_benchmark(self, algorithms: List[str] = None, n_episodes: int = 10, max_models_per_algo: int = 3) -> Dict[str, Any]:
        """Run benchmark on available models"""
        logger.info("🏁 STARTING MODEL BENCHMARK")
        logger.info("=" * 60)
        
        # Load test data
        if not self.load_test_data():
            return {'success': False, 'error': 'Failed to load test data'}
        
        # Discover models
        available_models = self.discover_models()
        
        if algorithms is None:
            algorithms = list(available_models.keys())
        
        # Benchmark models
        all_results = {}
        
        for algorithm in algorithms:
            if algorithm not in available_models or not available_models[algorithm]:
                logger.warning(f"⚠️  No models found for {algorithm.upper()}")
                continue
            
            # Limit number of models per algorithm
            models_to_test = available_models[algorithm][:max_models_per_algo]
            
            algorithm_results = []
            for model_path in models_to_test:
                result = self.benchmark_model(algorithm, model_path, n_episodes)
                if result['success']:
                    algorithm_results.append(result)
            
            if algorithm_results:
                all_results[algorithm] = algorithm_results
        
        # Generate comparison
        if all_results:
            comparison = self.generate_comparison(all_results)
            all_results['comparison'] = comparison
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"logs/benchmark_results_{timestamp}.json"
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            logger.info(f"📝 Benchmark results saved: {results_file}")
        
        return {
            'success': True,
            'results': all_results
        }
    
    def generate_comparison(self, results: Dict) -> Dict[str, Any]:
        """Generate comparison between all benchmarked models"""
        logger.info("📊 GENERATING MODEL COMPARISON")
        
        # Flatten results for comparison
        all_models = []
        for algorithm, algorithm_results in results.items():
            if algorithm != 'comparison':
                for result in algorithm_results:
                    all_models.append(result)
        
        # Sort by different metrics
        rankings = {
            'returns': sorted(all_models, key=lambda x: x['mean_returns'], reverse=True),
            'profitability': sorted(all_models, key=lambda x: x['profitability_rate'], reverse=True),
            'sharpe_ratio': sorted(all_models, key=lambda x: x['sharpe_ratio'], reverse=True),
            'stability': sorted(all_models, key=lambda x: -x['std_returns'])  # Lower volatility is better
        }
        
        # Overall ranking (weighted score)
        for model in all_models:
            model['overall_score'] = (
                model['mean_returns'] * 0.3 +
                model['profitability_rate'] * 0.25 +
                model['sharpe_ratio'] * 0.25 +
                (1.0 / (model['std_returns'] + 0.01)) * 0.2
            )
        
        rankings['overall'] = sorted(all_models, key=lambda x: x['overall_score'], reverse=True)
        
        # Best models
        best_models = {
            'best_returns': rankings['returns'][0],
            'most_profitable': rankings['profitability'][0],
            'best_sharpe': rankings['sharpe_ratio'][0],
            'most_stable': rankings['stability'][0],
            'best_overall': rankings['overall'][0]
        }
        
        logger.info("🏆 TOP PERFORMING MODELS:")
        logger.info(f"   🥇 Best Overall: {best_models['best_overall']['algorithm']} - {best_models['best_overall']['model_name']}")
        logger.info(f"      Score: {best_models['best_overall']['overall_score']:.3f}, Returns: {best_models['best_overall']['mean_returns']:+.2%}")
        
        logger.info(f"   💰 Best Returns: {best_models['best_returns']['algorithm']} - {best_models['best_returns']['model_name']}")
        logger.info(f"      Returns: {best_models['best_returns']['mean_returns']:+.2%}")
        
        logger.info(f"   📈 Most Profitable: {best_models['most_profitable']['algorithm']} - {best_models['most_profitable']['model_name']}")
        logger.info(f"      Profitability: {best_models['most_profitable']['profitability_rate']:.1%}")
        
        return {
            'rankings': rankings,
            'best_models': best_models,
            'total_models_tested': len(all_models)
        }

def main():
    """Main entry point for benchmarking"""
    parser = argparse.ArgumentParser(description='Benchmark Trained Forex RL Models')
    parser.add_argument('--algorithms', nargs='+', default=['sac', 'ppo', 'td3', 'a2c'],
                       choices=['sac', 'ppo', 'td3', 'a2c'],
                       help='Algorithms to benchmark')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Test episodes per model (default: 10)')
    parser.add_argument('--max-models', type=int, default=3,
                       help='Max models to test per algorithm (default: 3)')
    
    args = parser.parse_args()
    
    print("🏁 MODEL BENCHMARK SUITE")
    print("=" * 60)
    print(f"🧠 Algorithms: {', '.join([a.upper() for a in args.algorithms])}")
    print(f"🧪 Episodes per model: {args.episodes}")
    print(f"📊 Max models per algorithm: {args.max_models}")
    print("=" * 60)
    
    # Create benchmarker
    benchmark = AgentBenchmark()
    results = benchmark.run_benchmark(args.algorithms, args.episodes, args.max_models)
    
    if results['success']:
        print(f"\n🎉 BENCHMARK COMPLETED! 🎉")
        
        if 'comparison' in results['results']:
            comparison = results['results']['comparison']
            best_models = comparison['best_models']
            
            print(f"\n🏆 CHAMPION MODELS:")
            print(f"   🥇 Overall Winner: {best_models['best_overall']['algorithm']} ({best_models['best_overall']['mean_returns']:+.2%})")
            print(f"   💰 Highest Returns: {best_models['best_returns']['algorithm']} ({best_models['best_returns']['mean_returns']:+.2%})")
            print(f"   📈 Most Consistent: {best_models['most_profitable']['algorithm']} ({best_models['most_profitable']['profitability_rate']:.1%} win rate)")
            
            print(f"\n📊 Total models tested: {comparison['total_models_tested']}")
        
    else:
        print(f"\n❌ BENCHMARK FAILED: {results.get('error', 'Unknown error')}")
        sys.exit(1)

if __name__ == "__main__":
    main()