"""
Quick training script with recent data for testing
"""
import logging
from datetime import datetime, timedelta
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

def quick_train_and_test():
    """Quick training and testing with recent data"""
    
    logger.info("=== QUICK RL FOREX TRAINING ===")
    
    # Step 1: Fetch recent data for training (last 30 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    logger.info(f"Fetching data from {start_date} to {end_date}")
    
    fetcher = OandaDataFetcher()
    raw_data = fetcher.fetch_historical_data(start_date, end_date)
    
    if raw_data is None:
        logger.error("Failed to fetch training data")
        return False
    
    logger.info(f"Fetched {len(raw_data)} raw candles")
    
    # Step 2: Add technical indicators
    logger.info("Adding technical indicators...")
    indicators = TechnicalIndicators()
    full_data = indicators.add_all_indicators(raw_data)
    
    logger.info(f"Enhanced data shape: {full_data.shape}")
    
    # Step 3: Split data (80% training, 20% testing)
    split_point = int(len(full_data) * 0.8)
    train_data = full_data.iloc[:split_point].copy()
    test_data = full_data.iloc[split_point:].copy()
    
    logger.info(f"Training: {len(train_data)} samples")
    logger.info(f"Testing: {len(test_data)} samples")
    
    # Step 4: Create environments
    logger.info("Creating environments...")
    
    # Training environment
    train_env_base = ForexTradingEnv(train_data, CONFIG)
    train_env = DummyVecEnv([lambda: Monitor(train_env_base)])
    
    # Testing environment
    test_env = ForexTradingEnv(test_data, CONFIG)
    test_env = Monitor(test_env)
    
    # Step 5: Train agent
    logger.info("Training SAC agent...")
    
    agent = ForexSACAgent(train_env, CONFIG)
    training_stats = agent.train(total_timesteps=5000)  # Moderate training
    
    logger.info("Training completed!")
    
    # Step 6: Test agent
    logger.info("Testing trained agent...")
    
    n_test_episodes = 3
    test_results = []
    
    for episode in range(n_test_episodes):
        logger.info(f"Testing episode {episode + 1}/{n_test_episodes}")
        
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
        
        episode_result = {
            'episode': episode + 1,
            'total_reward': total_reward,
            'final_balance': info['balance'],
            'total_trades': info['total_trades'],
            'win_rate': info['win_rate'],
            'returns': info['returns'],
            'max_drawdown': info['max_drawdown']
        }
        
        test_results.append(episode_result)
        
        logger.info(f"Episode {episode + 1} Results:")
        logger.info(f"  Final Balance: ${episode_result['final_balance']:.2f}")
        logger.info(f"  Returns: {episode_result['returns']:.1%}")
        logger.info(f"  Total Trades: {episode_result['total_trades']}")
        logger.info(f"  Win Rate: {episode_result['win_rate']:.3f}")
        logger.info(f"  Max Drawdown: {episode_result['max_drawdown']:.3f}")
    
    # Step 7: Save model and results
    logger.info("Saving model...")
    model_path = agent.save_model()
    
    # Summary
    mean_returns = sum(r['returns'] for r in test_results) / len(test_results)
    mean_balance = sum(r['final_balance'] for r in test_results) / len(test_results)
    profitable_episodes = sum(1 for r in test_results if r['returns'] > 0)
    
    logger.info("\n" + "="*50)
    logger.info("         QUICK TRAINING SUMMARY")
    logger.info("="*50)
    logger.info(f"Training Data: {len(train_data)} samples")
    logger.info(f"Testing Data: {len(test_data)} samples") 
    logger.info(f"Training Steps: {training_stats['total_timesteps']}")
    logger.info(f"Model Saved: {model_path}")
    logger.info("")
    logger.info("TEST RESULTS:")
    logger.info(f"  Episodes: {n_test_episodes}")
    logger.info(f"  Mean Returns: {mean_returns:.1%}")
    logger.info(f"  Mean Final Balance: ${mean_balance:.2f}")
    logger.info(f"  Profitable Episodes: {profitable_episodes}/{n_test_episodes}")
    logger.info(f"  Success Rate: {profitable_episodes/n_test_episodes:.1%}")
    logger.info("="*50)
    
    # Assessment
    if profitable_episodes >= n_test_episodes * 0.6:
        logger.info("✅ GOOD: Agent shows promising results!")
    elif profitable_episodes >= n_test_episodes * 0.3:
        logger.info("⚠️  MODERATE: Agent needs more training")
    else:
        logger.info("❌ POOR: Agent needs significant improvements")
    
    logger.info(f"\n🎯 Ready for full training with more data and timesteps!")
    logger.info(f"📁 Model saved at: {model_path}")
    
    return True

if __name__ == "__main__":
    success = quick_train_and_test()
    if not success:
        exit(1)