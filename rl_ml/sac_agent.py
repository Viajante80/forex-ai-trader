"""
Soft Actor-Critic (SAC) Agent for Forex Trading
Optimized for continuous action spaces with entropy regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
import os
from datetime import datetime
import pickle
from config import CONFIG

logger = logging.getLogger(__name__)

class TradingCallback(BaseCallback):
    """
    Custom callback for trading agent to log performance metrics
    """
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_balances = []
        self.episode_trades = []
        self.episode_win_rates = []
        
    def _on_step(self) -> bool:
        # Log training progress
        if self.n_calls % self.log_interval == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_balance = np.mean(self.episode_balances[-100:])
                mean_win_rate = np.mean(self.episode_win_rates[-100:])
                mean_trades = np.mean(self.episode_trades[-100:])
                
                logger.info(f"Step {self.n_calls}: Mean Reward: {mean_reward:.4f}, "
                           f"Mean Balance: {mean_balance:.2f}, "
                           f"Mean Win Rate: {mean_win_rate:.3f}, "
                           f"Mean Trades: {mean_trades:.1f}")
        
        return True
    
    def _on_rollout_end(self) -> None:
        # Collect episode statistics from info buffer
        if hasattr(self.training_env, 'get_attr'):
            try:
                for env in self.training_env.envs:
                    if hasattr(env, 'episode_info') and env.episode_info:
                        info = env.episode_info
                        self.episode_rewards.append(info.get('episode_reward', 0))
                        self.episode_lengths.append(info.get('episode_length', 0))
                        self.episode_balances.append(info.get('balance', 1000))
                        self.episode_trades.append(info.get('total_trades', 0))
                        self.episode_win_rates.append(info.get('win_rate', 0))
            except Exception as e:
                # Silently continue if episode info is not available
                pass

class ForexSACAgent:
    """
    SAC Agent specialized for Forex Trading
    """
    
    def __init__(self, env, config=None):
        self.config = config or CONFIG
        self.env = env
        self.model = None
        self.training_history = {
            'rewards': [],
            'balances': [],
            'win_rates': [],
            'trades': [],
            'timestamps': []
        }
        
    def create_model(self) -> SAC:
        """
        Create SAC model with custom configuration
        """
        # Custom policy network architecture
        policy_kwargs = {
            'net_arch': [256, 256, 128],  # Actor and critic network architecture
            'activation_fn': torch.nn.ReLU,
            'normalize_images': False,
        }
        
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            tau=self.config.tau,
            ent_coef=self.config.alpha,  # Entropy coefficient
            target_update_interval=1,
            gradient_steps=1,
            optimize_memory_usage=False,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            seed=42
        )
        
        logger.info("SAC model created with custom configuration")
        return self.model
    
    def train(self, total_timesteps: int = None) -> Dict[str, Any]:
        """
        Train the SAC agent
        """
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        if self.model is None:
            self.create_model()
        
        # Create callback for logging
        callback = TradingCallback(log_interval=self.config.log_interval)
        
        logger.info(f"Starting SAC training for {total_timesteps} timesteps...")
        
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=self.config.log_interval,
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            logger.info("Training completed successfully")
            
            # Collect training statistics
            training_stats = {
                'total_timesteps': total_timesteps,
                'episode_rewards': callback.episode_rewards,
                'episode_balances': callback.episode_balances,
                'episode_win_rates': callback.episode_win_rates,
                'episode_trades': callback.episode_trades,
                'final_performance': self._evaluate_performance()
            }
            
            return training_stats
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True) -> Dict[str, Any]:
        """
        Evaluate the trained agent
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        episode_rewards = []
        episode_balances = []
        episode_trades = []
        episode_win_rates = []
        episode_returns = []
        
        logger.info(f"Evaluating agent for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(total_reward)
            episode_balances.append(info['balance'])
            episode_trades.append(info['total_trades'])
            episode_win_rates.append(info['win_rate'])
            episode_returns.append(info['returns'])
            
            logger.info(f"Episode {episode + 1}: "
                       f"Reward: {total_reward:.2f}, "
                       f"Balance: {info['balance']:.2f}, "
                       f"Trades: {info['total_trades']}, "
                       f"Win Rate: {info['win_rate']:.3f}")
        
        evaluation_results = {
            'n_episodes': n_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_balance': np.mean(episode_balances),
            'std_balance': np.std(episode_balances),
            'mean_trades': np.mean(episode_trades),
            'mean_win_rate': np.mean(episode_win_rates),
            'mean_returns': np.mean(episode_returns),
            'best_balance': np.max(episode_balances),
            'worst_balance': np.min(episode_balances),
            'profitable_episodes': sum(1 for r in episode_returns if r > 0),
            'profitability_rate': sum(1 for r in episode_returns if r > 0) / n_episodes
        }
        
        logger.info("Evaluation Results:")
        for key, value in evaluation_results.items():
            logger.info(f"  {key}: {value}")
        
        return evaluation_results
    
    def _evaluate_performance(self) -> Dict[str, Any]:
        """
        Quick performance evaluation
        """
        try:
            obs, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
            
            return {
                'total_reward': total_reward,
                'final_balance': info['balance'],
                'total_trades': info['total_trades'],
                'win_rate': info['win_rate'],
                'returns': info['returns'],
                'max_drawdown': info['max_drawdown']
            }
        except Exception as e:
            logger.warning(f"Performance evaluation failed: {e}")
            return {}
    
    def save_model(self, filename: str = None):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rl_ml/models/sac_forex_model_{timestamp}.zip"
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        self.model.save(filename)
        logger.info(f"Model saved to {filename}")
        
        # Save training history
        history_filename = filename.replace('.zip', '_history.pkl')
        with open(history_filename, 'wb') as f:
            pickle.dump(self.training_history, f)
        logger.info(f"Training history saved to {history_filename}")
        
        return filename
    
    def load_model(self, filename: str):
        """
        Load a trained model
        """
        try:
            self.model = SAC.load(filename, env=self.env)
            logger.info(f"Model loaded from {filename}")
            
            # Try to load training history
            history_filename = filename.replace('.zip', '_history.pkl')
            try:
                with open(history_filename, 'rb') as f:
                    self.training_history = pickle.load(f)
                logger.info(f"Training history loaded from {history_filename}")
            except FileNotFoundError:
                logger.warning(f"Training history file not found: {history_filename}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def get_action(self, observation, deterministic: bool = True) -> np.ndarray:
        """
        Get action from the trained model
        """
        if self.model is None:
            raise ValueError("Model must be trained or loaded before getting actions")
        
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    def update_training_history(self, episode_info: Dict[str, Any]):
        """
        Update training history with episode information
        """
        self.training_history['rewards'].append(episode_info.get('total_reward', 0))
        self.training_history['balances'].append(episode_info.get('balance', 1000))
        self.training_history['win_rates'].append(episode_info.get('win_rate', 0))
        self.training_history['trades'].append(episode_info.get('total_trades', 0))
        self.training_history['timestamps'].append(datetime.now())

if __name__ == "__main__":
    # Test the SAC agent
    from trading_env import ForexTradingEnv
    from data_fetcher import OandaDataFetcher
    from technical_indicators import TechnicalIndicators
    
    # Prepare test data
    fetcher = OandaDataFetcher()
    data = fetcher.fetch_candles(count=1000)
    
    if data is not None:
        indicators = TechnicalIndicators()
        enhanced_data = indicators.add_all_indicators(data)
        
        # Create environment
        env = ForexTradingEnv(enhanced_data)
        env = Monitor(env)  # Wrap with Monitor for logging
        env = DummyVecEnv([lambda: env])  # Vectorize environment
        
        # Create and test agent
        agent = ForexSACAgent(env)
        
        print("Testing SAC agent creation...")
        agent.create_model()
        print("SAC model created successfully!")
        
        # Test short training
        print("Testing short training run...")
        try:
            training_stats = agent.train(total_timesteps=1000)
            print("Training test completed!")
            print(f"Training stats keys: {list(training_stats.keys())}")
        except Exception as e:
            print(f"Training test failed: {e}")
    
    else:
        print("Failed to fetch data for SAC agent testing")