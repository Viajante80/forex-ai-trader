"""
High-Accuracy RL Training Script
Train models specifically designed to achieve >75% win rate
Uses ensemble methods, conservative trading, and advanced reward shaping
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
from typing import Dict, Any

from config_m1 import CONFIG_M1, TradingConfigM1
from config_aggressive import CONFIG_AGGRESSIVE
from high_accuracy_agent import ForexHighAccuracyAgent
from high_accuracy_env import HighAccuracyTradingEnv
from trading_logger import TradingLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HighAccuracyTrainer:
    """
    Specialized trainer for high-accuracy (>75% win rate) forex trading
    """
    
    def __init__(self, config: TradingConfigM1 = None, 
                 min_accuracy_target: float = 0.75,
                 max_trades_per_episode: int = 50):
        
        self.config = config or CONFIG_M1
        self.min_accuracy_target = min_accuracy_target
        self.max_trades_per_episode = max_trades_per_episode
        
        # High-accuracy specific settings
        self.config.initial_balance = 1000.0
        self.config.lookback_window = 50  # Longer lookback for better pattern recognition
        
        self.logger_system = TradingLogger(self.config)
        
        # Data storage
        self.train_data = None
        self.test_data = None
        
        logger.info(f"🎯 High-Accuracy Trainer initialized")
        logger.info(f"   Target accuracy: >{min_accuracy_target:.1%}")
        logger.info(f"   Max trades per episode: {max_trades_per_episode}")
    
    def load_data(self) -> bool:
        """Load and prepare data for high-accuracy training"""
        logger.info("🏛️  LOADING DATA FOR HIGH-ACCURACY TRAINING")
        
        # Prioritize enhanced dataset, fallback to other files
        enhanced_file = f"{self.config.instrument}_{self.config.timeframe}_FULL_2016_2025_ENHANCED.pkl"
        
        if os.path.exists(f"data/{enhanced_file}"):
            latest_file = enhanced_file
            logger.info("🎯 Using ENHANCED dataset with 139 indicators!")
        else:
            # Find any available data file
            data_files = [f for f in os.listdir("data/") if f.startswith(f"{self.config.instrument}_{self.config.timeframe}_") and f.endswith(".pkl")]
            
            if not data_files:
                logger.error("❌ No data files found in data/ directory")
                return False
            
            latest_file = sorted(data_files)[-1]
            logger.warning("⚠️  Using non-enhanced dataset - may reduce accuracy potential")
        data_path = f"data/{latest_file}"
        
        try:
            full_data = pd.read_pickle(data_path)
            logger.info(f"✅ Loaded {len(full_data):,} samples from {data_path}")
            
            # Special split for high-accuracy training (90/10 - more training data)
            total_samples = len(full_data)
            train_samples = int(total_samples * 0.9)
            
            self.train_data = full_data.iloc[:train_samples].copy()
            self.test_data = full_data.iloc[train_samples:].copy()
            
            logger.info(f"📊 High-accuracy data split:")
            logger.info(f"   Training: {len(self.train_data):,} samples (90%)")
            logger.info(f"   Testing: {len(self.test_data):,} samples (10%)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False
    
    def create_high_accuracy_environment(self, data: pd.DataFrame) -> HighAccuracyTradingEnv:
        """Create specialized high-accuracy trading environment"""
        
        env = HighAccuracyTradingEnv(
            data=data,
            config=self.config,
            min_confidence=self.min_accuracy_target,
            max_trades_per_episode=self.max_trades_per_episode,
            accuracy_window=20
        )
        
        return env
    
    def train_high_accuracy_model(self, timesteps: int = 2000000, 
                                 ensemble_size: int = 3) -> Dict[str, Any]:
        """Train high-accuracy model with ensemble approach"""
        
        logger.info("🎯 STARTING HIGH-ACCURACY TRAINING")
        logger.info("=" * 80)
        logger.info(f"🧠 Target Accuracy: >{self.min_accuracy_target:.1%}")
        logger.info(f"🔢 Ensemble Size: {ensemble_size}")
        logger.info(f"📊 Training Timesteps: {timesteps:,}")
        logger.info(f"🎲 Max Trades/Episode: {self.max_trades_per_episode}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Create high-accuracy environment
            train_env_base = self.create_high_accuracy_environment(self.train_data)
            train_env = DummyVecEnv([lambda: Monitor(train_env_base)])
            
            # Create high-accuracy agent
            agent = ForexHighAccuracyAgent(train_env, self.config)
            agent.ensemble_size = ensemble_size
            agent.min_confidence_threshold = self.min_accuracy_target
            
            # Train ensemble
            logger.info("🚀 Training high-accuracy ensemble...")
            training_stats = agent.train_ensemble(timesteps)
            
            # Save the ensemble
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = agent.save_ensemble(f"models/high_accuracy_{timestamp}")
            
            training_time = time.time() - start_time
            
            logger.info("✅ High-accuracy training completed!")
            logger.info(f"   Training time: {training_time/60:.1f} minutes")
            logger.info(f"   Model saved: {model_path}")
            
            return {
                'success': True,
                'model_path': model_path,
                'training_time_minutes': training_time / 60,
                'ensemble_size': ensemble_size,
                'target_accuracy': self.min_accuracy_target,
                'training_stats': training_stats
            }
            
        except Exception as e:
            logger.error(f"❌ High-accuracy training failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'success': False,
                'error': str(e),
                'training_time_minutes': (time.time() - start_time) / 60
            }
    
    def test_high_accuracy_model(self, agent: ForexHighAccuracyAgent, 
                                n_episodes: int = 20) -> Dict[str, Any]:
        """Comprehensive testing of high-accuracy model"""
        
        logger.info(f"🧪 TESTING HIGH-ACCURACY MODEL ({n_episodes} episodes)")
        
        # Create test environment
        test_env = self.create_high_accuracy_environment(self.test_data)
        
        # Run comprehensive evaluation
        results = agent.evaluate_high_accuracy(n_episodes)
        
        # Additional analysis
        episode_accuracies = [r['accuracy'] for r in results['episode_details']]
        high_accuracy_episodes = sum(1 for acc in episode_accuracies if acc >= 0.75)
        
        # Calculate consistency metrics
        accuracy_std = np.std(episode_accuracies)
        min_accuracy = np.min(episode_accuracies) if episode_accuracies else 0.0
        max_accuracy = np.max(episode_accuracies) if episode_accuracies else 0.0
        
        enhanced_results = {
            **results,
            'consistency_score': 1.0 / (accuracy_std + 0.01),  # Higher = more consistent
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'accuracy_range': max_accuracy - min_accuracy,
            'episodes_tested': n_episodes,
            'success_rate_75pct': high_accuracy_episodes / n_episodes
        }
        
        # Performance assessment
        overall_accuracy = results['overall_accuracy']
        high_acc_rate = results['high_accuracy_rate']
        
        if overall_accuracy >= 0.75 and high_acc_rate >= 0.6:
            assessment = "🚀 EXCEPTIONAL - Target achieved!"
        elif overall_accuracy >= 0.70 and high_acc_rate >= 0.4:
            assessment = "🔥 EXCELLENT - Close to target!"
        elif overall_accuracy >= 0.65:
            assessment = "✅ GOOD - Promising results!"
        else:
            assessment = "⚠️  NEEDS IMPROVEMENT - More training required"
        
        enhanced_results['assessment'] = assessment
        
        return enhanced_results
    
    def run_complete_high_accuracy_training(self, timesteps: int = 2000000,
                                           ensemble_size: int = 3,
                                           test_episodes: int = 20) -> Dict[str, Any]:
        """Run complete high-accuracy training and testing pipeline"""
        
        logger.info("🎯 COMPLETE HIGH-ACCURACY TRAINING PIPELINE")
        logger.info("=" * 80)
        
        # Load data
        if not self.load_data():
            return {'success': False, 'error': 'Failed to load data'}
        
        # Train model
        training_result = self.train_high_accuracy_model(timesteps, ensemble_size)
        
        if not training_result['success']:
            return training_result
        
        # Test model (create new agent for testing)
        try:
            test_env = self.create_high_accuracy_environment(self.test_data)
            test_env = DummyVecEnv([lambda: Monitor(test_env)])
            
            agent = ForexHighAccuracyAgent(test_env, self.config)
            agent.ensemble_size = ensemble_size
            
            # Load the trained ensemble
            # Note: In practice, you'd load the saved ensemble here
            # For now, we'll use a placeholder
            
            testing_result = {'overall_accuracy': 0.0, 'assessment': 'Model loading not implemented in test'}
            
        except Exception as e:
            logger.warning(f"Testing failed: {e}")
            testing_result = {'error': str(e)}
        
        # Save complete results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"logs/high_accuracy_results_{timestamp}.json"
        
        complete_results = {
            'training': training_result,
            'testing': testing_result,
            'configuration': {
                'target_accuracy': self.min_accuracy_target,
                'ensemble_size': ensemble_size,
                'timesteps': timesteps,
                'test_episodes': test_episodes,
                'max_trades_per_episode': self.max_trades_per_episode
            },
            'timestamp': timestamp
        }
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        logger.info(f"📝 Complete results saved: {results_file}")
        
        return complete_results

def main():
    """Main entry point for high-accuracy training"""
    parser = argparse.ArgumentParser(description='High-Accuracy Forex RL Training (>75% win rate)')
    parser.add_argument('--target-accuracy', type=float, default=0.75,
                       help='Target accuracy threshold (default: 0.75)')
    parser.add_argument('--timesteps', type=int, default=2000000,
                       help='Training timesteps (default: 2M)')
    parser.add_argument('--ensemble-size', type=int, default=3,
                       help='Ensemble size (default: 3)')
    parser.add_argument('--test-episodes', type=int, default=20,
                       help='Test episodes (default: 20)')
    parser.add_argument('--max-trades', type=int, default=50,
                       help='Max trades per episode (default: 50)')
    parser.add_argument('--initial-balance', type=float, default=1000.0,
                       help='Initial balance (default: $1000)')
    
    args = parser.parse_args()
    
    # M1 check
    if torch.backends.mps.is_available():
        print("🚀 M1 Mac GPU detected - optimal performance!")
    else:
        print("⚠️  CPU training - will be slower")
    
    print("🎯 HIGH-ACCURACY FOREX RL TRAINING")
    print("=" * 80)
    print(f"🎯 Target Accuracy: >{args.target_accuracy:.1%}")
    print(f"🧠 Ensemble Size: {args.ensemble_size}")
    print(f"📊 Training Timesteps: {args.timesteps:,}")
    print(f"🧪 Test Episodes: {args.test_episodes}")
    print(f"🎲 Max Trades/Episode: {args.max_trades}")
    print(f"💰 Initial Balance: ${args.initial_balance:,.2f}")
    print(f"⏰ Est. Time: {args.timesteps * args.ensemble_size / 200000:.1f} hours")
    print("=" * 80)
    
    # Create config
    config = TradingConfigM1()
    config.initial_balance = args.initial_balance
    
    # Create and run trainer
    trainer = HighAccuracyTrainer(
        config=config,
        min_accuracy_target=args.target_accuracy,
        max_trades_per_episode=args.max_trades
    )
    
    results = trainer.run_complete_high_accuracy_training(
        timesteps=args.timesteps,
        ensemble_size=args.ensemble_size,
        test_episodes=args.test_episodes
    )
    
    if results.get('training', {}).get('success'):
        training = results['training']
        testing = results.get('testing', {})
        
        print(f"\n🎉 HIGH-ACCURACY TRAINING COMPLETED! 🎉")
        print(f"✅ Training Time: {training['training_time_minutes']:.1f} minutes")
        print(f"📄 Model: {training['model_path']}")
        
        if 'overall_accuracy' in testing:
            print(f"🎯 Test Accuracy: {testing['overall_accuracy']:.1%}")
            print(f"📊 Assessment: {testing.get('assessment', 'N/A')}")
        
        target_achieved = testing.get('overall_accuracy', 0) >= args.target_accuracy
        print(f"🏆 Target Achieved: {'✅ YES' if target_achieved else '❌ NO'}")
        
    else:
        error = results.get('training', {}).get('error', 'Unknown error')
        print(f"\n❌ HIGH-ACCURACY TRAINING FAILED: {error}")
        sys.exit(1)

if __name__ == "__main__":
    main()