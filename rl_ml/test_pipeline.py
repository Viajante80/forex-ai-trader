"""
Test the complete RL trading pipeline with recent data
"""
import logging
from data_fetcher import OandaDataFetcher
from technical_indicators import TechnicalIndicators
from trading_env import ForexTradingEnv
from sac_agent import ForexSACAgent
from trading_logger import TradingLogger
from config import CONFIG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete pipeline with recent data"""
    
    logger.info("=== TESTING COMPLETE RL FOREX PIPELINE ===")
    
    # Step 1: Fetch recent data
    logger.info("Step 1: Fetching recent data for testing...")
    fetcher = OandaDataFetcher()
    data = fetcher.fetch_candles(count=2000)  # Get ~7 days of 5-min data
    
    if data is None:
        logger.error("Failed to fetch data")
        return False
    
    logger.info(f"Fetched {len(data)} data points")
    
    # Step 2: Add technical indicators
    logger.info("Step 2: Adding technical indicators...")
    indicators = TechnicalIndicators()
    enhanced_data = indicators.add_all_indicators(data)
    logger.info(f"Enhanced data shape: {enhanced_data.shape}")
    
    # Step 3: Split data (use 80% for training, 20% for testing)
    split_point = int(len(enhanced_data) * 0.8)
    train_data = enhanced_data.iloc[:split_point].copy()
    test_data = enhanced_data.iloc[split_point:].copy()
    
    logger.info(f"Training data: {len(train_data)} samples")
    logger.info(f"Testing data: {len(test_data)} samples")
    
    # Step 4: Create environments
    logger.info("Step 4: Creating environments...")
    train_env_base = ForexTradingEnv(train_data, CONFIG)
    train_env = DummyVecEnv([lambda: Monitor(train_env_base)])
    
    test_env = ForexTradingEnv(test_data, CONFIG)
    test_env = Monitor(test_env)
    
    # Step 5: Create agent and train
    logger.info("Step 5: Training agent (short training)...")
    agent = ForexSACAgent(train_env, CONFIG)
    training_stats = agent.train(total_timesteps=500)  # Very short training for testing
    
    logger.info("Training completed!")
    logger.info(f"Training stats keys: {list(training_stats.keys())}")
    
    # Step 6: Test agent
    logger.info("Step 6: Testing agent...")
    test_results = []
    
    for episode in range(2):  # Test 2 episodes
        logger.info(f"Testing episode {episode + 1}")
        
        obs, _ = test_env.reset()
        total_reward = 0
        step = 0
        done = False
        
        while not done:
            action = agent.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward
            step += 1
            done = terminated or truncated
            
            if step % 100 == 0:
                logger.debug(f"Step {step}: Balance=${info['balance']:.2f}")
        
        episode_result = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'final_balance': info['balance'],
            'total_trades': info['total_trades'],
            'win_rate': info['win_rate'],
            'returns': info['returns']
        }
        
        test_results.append(episode_result)
        
        logger.info(f"Episode {episode + 1} completed:")
        logger.info(f"  Final Balance: ${episode_result['final_balance']:.2f}")
        logger.info(f"  Returns: {episode_result['returns']:.3f}")
        logger.info(f"  Total Trades: {episode_result['total_trades']}")
        logger.info(f"  Win Rate: {episode_result['win_rate']:.3f}")
    
    # Step 7: Save model and generate report
    logger.info("Step 7: Saving model...")
    model_path = agent.save_model()
    logger.info(f"Model saved to: {model_path}")
    
    # Summary
    logger.info("\n=== TEST PIPELINE SUMMARY ===")
    logger.info(f"✅ Data fetching: Success ({len(enhanced_data)} samples)")
    logger.info(f"✅ Technical indicators: Success ({enhanced_data.shape[1]} features)")
    logger.info(f"✅ Environment creation: Success")
    logger.info(f"✅ Agent training: Success ({training_stats['total_timesteps']} steps)")
    logger.info(f"✅ Agent testing: Success ({len(test_results)} episodes)")
    logger.info(f"✅ Model saving: Success")
    
    mean_returns = sum(r['returns'] for r in test_results) / len(test_results)
    mean_balance = sum(r['final_balance'] for r in test_results) / len(test_results)
    
    logger.info(f"\nTest Results:")
    logger.info(f"  Mean Returns: {mean_returns:.3f}")
    logger.info(f"  Mean Final Balance: ${mean_balance:.2f}")
    
    logger.info("\n🎉 COMPLETE PIPELINE TEST SUCCESSFUL! 🎉")
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    if not success:
        exit(1)