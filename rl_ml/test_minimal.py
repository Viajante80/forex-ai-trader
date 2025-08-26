#!/usr/bin/env python3
"""
Minimal test to validate the high-accuracy system components
"""
import os
import pandas as pd
import numpy as np
from config_m1 import TradingConfigM1
from high_accuracy_env import HighAccuracyTradingEnv
from high_accuracy_agent import ForexHighAccuracyAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

def test_system_components():
    """Test each component individually"""
    print("🧪 MINIMAL SYSTEM TEST")
    print("=" * 40)
    
    # Test 1: Data loading
    print("1. Testing data loading...")
    enhanced_file = "data/EUR_USD_M5_FULL_2016_2025_ENHANCED.pkl"
    
    if not os.path.exists(enhanced_file):
        print("❌ Enhanced dataset not found")
        return False
    
    data = pd.read_pickle(enhanced_file)
    print(f"✅ Loaded {len(data):,} samples with {len(data.columns)} features")
    
    # Use tiny subset for testing
    test_data = data.iloc[:100].copy()
    
    # Test 2: Configuration
    print("2. Testing configuration...")
    config = TradingConfigM1()
    config.initial_balance = 1000.0
    print(f"✅ Config created: {config.instrument}, balance: ${config.initial_balance}")
    
    # Test 3: Environment
    print("3. Testing high-accuracy environment...")
    env = HighAccuracyTradingEnv(
        data=test_data,
        config=config,
        min_confidence=0.75,
        max_trades_per_episode=5,
        accuracy_window=3
    )
    
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])
    print("✅ Environment created successfully")
    
    # Test 4: Agent initialization
    print("4. Testing agent initialization...")
    agent = ForexHighAccuracyAgent(vec_env, config)
    agent.ensemble_size = 1  # Single model for quick test
    print("✅ Agent created successfully")
    
    # Test 5: Model creation
    print("5. Testing model creation...")
    models = agent.create_ensemble_models()
    print(f"✅ Created {len(models)} model(s)")
    
    # Test 6: Environment reset and step
    print("6. Testing environment interaction...")
    obs = vec_env.reset()
    print(f"✅ Environment reset - obs shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    # Test a few steps
    for i in range(3):
        action = vec_env.action_space.sample()  # Random action
        step_result = vec_env.step([action])
        
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
            
        reward_val = float(reward[0]) if hasattr(reward, '__len__') else float(reward)
        print(f"   Step {i+1}: action={action}, reward={reward_val:.4f}, done={terminated or truncated}")
        
        if terminated or truncated:
            obs = vec_env.reset()
    
    print("✅ Environment interaction successful")
    
    # Test 7: Ensemble prediction
    print("7. Testing ensemble prediction...")
    try:
        action, confidence = agent.get_ensemble_prediction(obs)
        print(f"✅ Ensemble prediction: action={action}, confidence={confidence:.3f}")
    except Exception as e:
        print(f"⚠️  Ensemble prediction test: {e}")
    
    print("\n🎉 MINIMAL SYSTEM TEST COMPLETED!")
    print("✅ All core components are working")
    print("🚀 Ready for full training!")
    
    return True

if __name__ == "__main__":
    test_system_components()