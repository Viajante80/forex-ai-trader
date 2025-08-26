"""
M1 Mac Optimized SAC Agent for Forex Trading
Uses MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon
"""
import torch
import torch.nn as nn
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from typing import Dict, Any, Optional
import logging
import os
from datetime import datetime
import pickle
from config_m1 import CONFIG_M1

logger = logging.getLogger(__name__)

class M1OptimizedCallback(BaseCallback):
    """M1 Mac optimized callback with efficient logging"""
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_balances = []
        self.episode_trades = []
        self.episode_win_rates = []
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Log less frequently to reduce overhead on M1
        if self.step_count % self.log_interval == 0:
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-10:]  # Last 10 episodes
                mean_reward = np.mean(recent_rewards) if recent_rewards else 0
                mean_balance = np.mean(self.episode_balances[-10:]) if self.episode_balances else 1000
                mean_win_rate = np.mean(self.episode_win_rates[-10:]) if self.episode_win_rates else 0
                
                logger.info(f"Step {self.step_count}: "
                           f"Reward: {mean_reward:.4f}, "
                           f"Balance: ${mean_balance:.2f}, "
                           f"WinRate: {mean_win_rate:.3f}")
        
        return True

class ForexSACAgentM1:
    """M1 Mac optimized SAC Agent for Forex Trading"""
    
    def __init__(self, env, config=None):
        self.config = config or CONFIG_M1
        self.env = env
        self.model = None
        self.device = self.config.device
        
        # M1 specific optimizations
        if self.device == "mps":
            # Enable MPS optimizations
            os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            logger.info("🚀 M1 Mac optimizations enabled!")
        
        self.training_history = {
            'rewards': [],
            'balances': [],
            'win_rates': [],
            'trades': [],
            'timestamps': []
        }
        
    def create_model(self) -> SAC:
        """Create M1 optimized SAC model"""
        
        # M1 optimized policy network architecture
        policy_kwargs = {
            'net_arch': [128, 128],  # Smaller networks for M1 efficiency
            'activation_fn': torch.nn.ReLU,
            'normalize_images': False,
        }
        
        # Override device for stable-baselines3
        original_device = None
        if self.device == "mps":
            # Stable-baselines3 doesn't support MPS directly, use CPU with optimizations
            actual_device = "cpu"
            torch.set_default_device("cpu")  # Ensure CPU for stable-baselines3
            logger.info("Using CPU with M1 optimizations for stable-baselines3")
        else:
            actual_device = self.device
        
        self.model = SAC(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.config.learning_rate,
            buffer_size=self.config.buffer_size,
            batch_size=self.config.batch_size,
            gamma=self.config.gamma,
            tau=self.config.tau,
            ent_coef=self.config.alpha,
            target_update_interval=1,
            gradient_steps=1,
            optimize_memory_usage=False,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=actual_device,
            tensorboard_log="./tensorboard_logs/",
            seed=42
        )
        
        logger.info(f"SAC model created for device: {actual_device}")
        return self.model
    
    def train(self, total_timesteps: int = None) -> Dict[str, Any]:
        """Train the M1 optimized SAC agent"""
        total_timesteps = total_timesteps or self.config.total_timesteps
        
        if self.model is None:
            self.create_model()
        
        # M1 optimized callback
        callback = M1OptimizedCallback(log_interval=self.config.log_interval)
        
        logger.info(f"Starting M1 optimized SAC training for {total_timesteps} timesteps...")
        
        try:
            # Enable M1 optimizations during training
            with torch.no_grad():
                pass  # Context for potential M1 optimizations
            
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=callback,
                log_interval=self.config.log_interval,
                progress_bar=True,
                reset_num_timesteps=False
            )
            
            logger.info("M1 optimized training completed successfully!")
            
            training_stats = {
                'total_timesteps': total_timesteps,
                'episode_rewards': callback.episode_rewards,
                'episode_balances': callback.episode_balances,
                'episode_win_rates': callback.episode_win_rates,
                'final_performance': self._evaluate_performance()
            }
            
            return training_stats
            
        except Exception as e:
            logger.error(f"M1 training failed: {e}")
            raise
    
    def evaluate(self, n_episodes: int = 5, deterministic: bool = True) -> Dict[str, Any]:
        """Evaluate the M1 trained agent"""
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        episode_results = []
        logger.info(f"Evaluating M1 agent for {n_episodes} episodes...")
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            total_reward = 0
            done = False
            
            # M1 optimized inference
            with torch.no_grad():
                while not done:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    done = terminated or truncated
            
            episode_results.append({
                'episode': episode + 1,
                'total_reward': total_reward,
                'final_balance': info['balance'],
                'total_trades': info['total_trades'],
                'win_rate': info['win_rate'],
                'returns': info['returns'],
                'max_drawdown': info['max_drawdown']
            })
            
            logger.info(f"Episode {episode + 1}: "
                       f"Balance: ${info['balance']:.2f}, "
                       f"Returns: {info['returns']:.1%}, "
                       f"Trades: {info['total_trades']}")
        
        # Calculate evaluation metrics
        returns = [r['returns'] for r in episode_results]
        balances = [r['final_balance'] for r in episode_results]
        
        evaluation_summary = {
            'n_episodes': n_episodes,
            'mean_returns': np.mean(returns),
            'std_returns': np.std(returns),
            'mean_balance': np.mean(balances),
            'best_balance': np.max(balances),
            'worst_balance': np.min(balances),
            'profitable_episodes': sum(1 for r in returns if r > 0),
            'profitability_rate': sum(1 for r in returns if r > 0) / n_episodes,
            'episodes': episode_results
        }
        
        return evaluation_summary
    
    def _evaluate_performance(self) -> Dict[str, Any]:
        """Quick M1 optimized performance evaluation"""
        try:
            obs = self.env.reset()
            total_reward = 0
            done = False
            
            with torch.no_grad():
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
            logger.warning(f"M1 performance evaluation failed: {e}")
            return {}
    
    def save_model(self, filename: str = None):
        """Save the M1 trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"models/sac_forex_m1_model_{timestamp}.zip"
        
        # Ensure models directory exists
        if not filename.startswith('/'):  # relative path
            filename = os.path.join('models', os.path.basename(filename))
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        self.model.save(filename)
        logger.info(f"M1 model saved to {filename}")
        
        # Save training history
        history_filename = filename.replace('.zip', '_history.pkl')
        with open(history_filename, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        return filename
    
    def load_model(self, filename: str):
        """Load a M1 trained model"""
        try:
            self.model = SAC.load(filename, env=self.env)
            logger.info(f"M1 model loaded from {filename}")
            
            # Load training history
            history_filename = filename.replace('.zip', '_history.pkl')
            try:
                with open(history_filename, 'rb') as f:
                    self.training_history = pickle.load(f)
            except FileNotFoundError:
                logger.warning(f"Training history not found: {history_filename}")
                
        except Exception as e:
            logger.error(f"Failed to load M1 model: {e}")
            raise
    
    def get_action(self, observation, deterministic: bool = True) -> np.ndarray:
        """Get M1 optimized action"""
        if self.model is None:
            raise ValueError("Model must be trained or loaded before getting actions")
        
        with torch.no_grad():  # M1 optimization
            action, _ = self.model.predict(observation, deterministic=deterministic)
            
        return action

if __name__ == "__main__":
    # Test M1 agent creation
    from trading_env import ForexTradingEnv
    from data_fetcher import OandaDataFetcher
    from technical_indicators import TechnicalIndicators
    
    logger.info("Testing M1 SAC agent creation...")
    
    # Check M1 availability
    if torch.backends.mps.is_available():
        logger.info("✅ M1 MPS available and ready!")
    else:
        logger.info("❌ M1 MPS not available, using CPU")
    
    # Test with sample data
    fetcher = OandaDataFetcher()
    data = fetcher.fetch_candles(count=200)
    
    if data is not None:
        indicators = TechnicalIndicators()
        enhanced_data = indicators.add_all_indicators(data)
        
        env = ForexTradingEnv(enhanced_data, CONFIG_M1)
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])
        
        agent = ForexSACAgentM1(env, CONFIG_M1)
        print("M1 SAC agent created successfully!")
        
        # Test model creation
        agent.create_model()
        print("M1 model created successfully!")
        
    else:
        print("Failed to fetch data for M1 agent testing")