"""
Quick test script for high-accuracy RL system
Tests with smaller parameters to validate >75% accuracy is achievable
"""
import argparse
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

from config_m1 import TradingConfigM1
from high_accuracy_env import HighAccuracyTradingEnv
from high_accuracy_agent import ForexHighAccuracyAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_high_accuracy_system():
    """Quick test of high-accuracy system"""
    
    print("🎯 TESTING HIGH-ACCURACY SYSTEM")
    print("=" * 50)
    
    # Load enhanced dataset
    enhanced_file = "EUR_USD_M5_FULL_2016_2025_ENHANCED.pkl"
    
    if os.path.exists(f"data/{enhanced_file}"):
        latest_file = enhanced_file
        print("🎯 Using ENHANCED dataset with 139 indicators!")
    else:
        # Fallback to any available data file
        data_files = [f for f in os.listdir("data/") if f.endswith(".pkl")]
        if not data_files:
            print("❌ No data files found")
            return
        
        latest_file = sorted(data_files)[-1]
        print("⚠️  Using non-enhanced dataset")
    data_path = f"data/{latest_file}"
    data = pd.read_pickle(data_path)
    
    print(f"✅ Loaded {len(data):,} samples")
    
    # Use small subset for quick test
    test_data = data.iloc[:1000].copy()  # Just 1000 samples for quick test
    
    # Create config
    config = TradingConfigM1()
    config.initial_balance = 1000.0
    
    # Create high-accuracy environment
    env = HighAccuracyTradingEnv(
        data=test_data,
        config=config,
        min_confidence=0.75,
        max_trades_per_episode=10,  # Very conservative
        accuracy_window=5
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    
    print("🔧 Environment created")
    
    # Create agent
    agent = ForexHighAccuracyAgent(vec_env, config)
    agent.ensemble_size = 2  # Small ensemble for quick test
    agent.min_confidence_threshold = 0.75
    
    print("🧠 Agent created")
    
    # Quick training
    print("🚀 Starting quick training (10k timesteps)...")
    try:
        training_stats = agent.train_ensemble(10000)
        print("✅ Training completed")
        
        # Quick evaluation
        print("🧪 Testing accuracy...")
        results = agent.evaluate_high_accuracy(n_episodes=5)
        
        print("\n📊 RESULTS:")
        print(f"   Overall Accuracy: {results['overall_accuracy']:.1%}")
        print(f"   Episodes >75%: {results['episodes_above_75pct']}/5")
        print(f"   Avg Trades/Episode: {results['avg_trades_per_episode']:.1f}")
        print(f"   Target Achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
        
        if results['overall_accuracy'] >= 0.75:
            print("\n🎉 HIGH-ACCURACY SYSTEM VALIDATED!")
            print("✅ >75% win rate achieved in test!")
        else:
            print(f"\n⚠️  Need more training - achieved {results['overall_accuracy']:.1%}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_high_accuracy_system()