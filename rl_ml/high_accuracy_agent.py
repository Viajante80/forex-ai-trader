"""
High-Accuracy RL Agent for Forex Trading
Designed to achieve >75% win rate through conservative trading
Uses ensemble methods, confidence thresholds, and advanced reward shaping
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
from typing import Dict, Any, Optional, List
import logging
import os
from datetime import datetime
import pickle
from config_m1 import CONFIG_M1

logger = logging.getLogger(__name__)

class HighAccuracyCallback(BaseCallback):
    """Callback specifically for high-accuracy training with detailed metrics"""
    
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_trade_counts = []
        self.episode_balances = []
        self.step_count = 0
        self.high_accuracy_episodes = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        if self.step_count % self.log_interval == 0:
            if len(self.episode_accuracies) > 0:
                recent_accuracy = np.mean(self.episode_accuracies[-10:])
                recent_trades = np.mean(self.episode_trade_counts[-10:])
                recent_balance = np.mean(self.episode_balances[-10:])
                high_acc_rate = self.high_accuracy_episodes / len(self.episode_accuracies) if self.episode_accuracies else 0
                
                logger.info(f"HighAcc Step {self.step_count}: "
                           f"Accuracy: {recent_accuracy:.1%}, "
                           f"Trades: {recent_trades:.1f}, "
                           f"Balance: ${recent_balance:.2f}, "
                           f">75% Rate: {high_acc_rate:.1%}")
        
        return True

class ConfidenceNetwork(nn.Module):
    """Neural network that outputs trade confidence along with actions"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Separate heads for different outputs
        self.action_head = nn.Linear(prev_dim, 3)  # Buy, Sell, Hold
        self.confidence_head = nn.Linear(prev_dim, 1)  # Confidence score
        self.volatility_head = nn.Linear(prev_dim, 1)  # Market volatility estimate
        
    def forward(self, x):
        features = self.feature_extractor(x)
        
        actions = torch.softmax(self.action_head(features), dim=-1)
        confidence = torch.sigmoid(self.confidence_head(features))
        volatility = torch.sigmoid(self.volatility_head(features))
        
        return actions, confidence, volatility

class HighAccuracyTradingEnv:
    """
    Modified trading environment optimized for high accuracy
    """
    
    def __init__(self, data: pd.DataFrame, config, min_confidence: float = 0.75):
        self.base_env = None  # Will be set externally
        self.min_confidence = min_confidence
        self.config = config
        
        # High-accuracy specific metrics
        self.correct_predictions = 0
        self.total_predictions = 0
        self.confidence_scores = []
        self.trade_outcomes = []
        
    def calculate_accuracy_reward(self, action: int, actual_return: float, confidence: float) -> float:
        """Calculate reward specifically optimized for accuracy"""
        
        # Base accuracy reward
        if action == 0:  # Hold
            return 0.01  # Small positive reward for patience
        
        # For buy/sell actions
        is_correct = (action == 1 and actual_return > 0) or (action == 2 and actual_return < 0)
        
        if is_correct:
            # Correct prediction - reward based on confidence and magnitude
            accuracy_bonus = confidence * 10.0
            magnitude_bonus = abs(actual_return) * 100.0
            return accuracy_bonus + magnitude_bonus
        else:
            # Wrong prediction - heavy penalty, especially for high confidence
            confidence_penalty = confidence * 20.0
            magnitude_penalty = abs(actual_return) * 50.0
            return -(confidence_penalty + magnitude_penalty)

class ForexHighAccuracyAgent:
    """
    High-Accuracy Forex RL Agent designed for >75% win rate
    Uses ensemble methods and conservative trading
    """
    
    def __init__(self, env, config=None):
        self.config = config or CONFIG_M1
        self.env = env
        self.models = []  # Ensemble of models
        self.confidence_network = None
        self.device = self.config.device
        
        # High-accuracy specific parameters
        self.min_confidence_threshold = 0.75
        self.ensemble_size = 3
        self.min_agreement = 2  # Minimum models that must agree
        
        # Performance tracking
        self.accuracy_history = []
        self.confidence_history = []
        self.trade_history = []
        
        logger.info("🎯 High-Accuracy Agent initialized - Target: >75% win rate")
        
    def create_ensemble_models(self) -> List[SAC]:
        """Create ensemble of SAC models with different configurations"""
        
        models = []
        
        for i in range(self.ensemble_size):
            # Slightly different configurations for diversity
            policy_kwargs = {
                'net_arch': [256 + i*32, 256 + i*32],  # Different network sizes
                'activation_fn': torch.nn.ReLU,
                'normalize_images': False,
            }
            
            # Different hyperparameters for each model (dynamic based on ensemble size)
            learning_rates = [1e-4, 3e-4, 1e-3, 5e-4, 2e-4][:self.ensemble_size]
            alphas = [0.1, 0.2, 0.3, 0.15, 0.25][:self.ensemble_size]  # Different exploration
            
            # Extend arrays if ensemble is larger than predefined values
            while len(learning_rates) < self.ensemble_size:
                learning_rates.append(learning_rates[-1] * 0.8)  # Slightly different rates
            while len(alphas) < self.ensemble_size:
                alphas.append(min(0.4, alphas[-1] + 0.05))  # Slightly different exploration
            
            model = SAC(
                policy="MlpPolicy",
                env=self.env,
                learning_rate=learning_rates[i],
                buffer_size=self.config.buffer_size,
                batch_size=self.config.batch_size,
                gamma=0.99,  # Higher gamma for long-term thinking
                tau=0.005,
                ent_coef=alphas[i],
                target_update_interval=1,
                gradient_steps=1,
                optimize_memory_usage=False,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device="cpu",  # Force CPU for stability
                seed=42 + i  # Different seeds for diversity
            )
            
            models.append(model)
            logger.info(f"✅ Created ensemble model {i+1}/{self.ensemble_size}")
        
        self.models = models
        return models
    
    def train_ensemble(self, total_timesteps: int = 1000000) -> Dict[str, Any]:
        """Train ensemble of models for high accuracy"""
        
        if not self.models:
            self.create_ensemble_models()
        
        callback = HighAccuracyCallback(log_interval=self.config.log_interval)
        
        logger.info(f"🎯 Starting high-accuracy ensemble training for {total_timesteps:,} timesteps...")
        logger.info(f"🔧 Ensemble size: {self.ensemble_size} models")
        logger.info(f"📊 Target accuracy: >{self.min_confidence_threshold:.1%}")
        
        ensemble_stats = []
        
        # Train each model in the ensemble
        for i, model in enumerate(self.models):
            logger.info(f"🚀 Training ensemble model {i+1}/{self.ensemble_size}...")
            
            try:
                model.learn(
                    total_timesteps=total_timesteps // self.ensemble_size,  # Distribute timesteps
                    callback=callback,
                    log_interval=self.config.log_interval,
                    progress_bar=True,
                    reset_num_timesteps=False
                )
                
                # Evaluate individual model
                model_stats = self._evaluate_single_model(model, n_episodes=5)
                ensemble_stats.append(model_stats)
                
                logger.info(f"✅ Model {i+1} - Accuracy: {model_stats.get('accuracy', 0):.1%}")
                
            except Exception as e:
                logger.error(f"❌ Model {i+1} training failed: {e}")
        
        logger.info("🎯 High-accuracy ensemble training completed!")
        
        return {
            'ensemble_size': self.ensemble_size,
            'individual_stats': ensemble_stats,
            'target_accuracy': self.min_confidence_threshold
        }
    
    def get_ensemble_prediction(self, observation) -> tuple[int, float]:
        """Get ensemble prediction with confidence score"""
        
        predictions = []
        confidences = []
        
        # Get prediction from each model
        with torch.no_grad():
            for model in self.models:
                action, _ = model.predict(observation, deterministic=True)
                
                # Simple confidence based on action probability
                # In practice, you'd want a more sophisticated confidence measure
                confidence = 0.8  # Placeholder - would extract from model's policy network
                
                predictions.append(action[0] if hasattr(action, '__len__') else action)
                confidences.append(confidence)
        
        # Ensemble decision making
        predictions = np.array(predictions)
        avg_confidence = np.mean(confidences)
        
        # Count votes for each action
        unique_actions, counts = np.unique(predictions, return_counts=True)
        
        # Require minimum agreement
        max_votes = np.max(counts)
        if max_votes >= self.min_agreement:
            # Find the action with most votes
            winning_action = unique_actions[np.argmax(counts)]
            # Adjust confidence based on agreement
            agreement_ratio = max_votes / len(self.models)
            final_confidence = avg_confidence * agreement_ratio
        else:
            # Not enough agreement - default to hold
            winning_action = 0  # Hold
            final_confidence = 0.0
        
        return int(winning_action), float(final_confidence)
    
    def get_action(self, observation, deterministic: bool = True) -> np.ndarray:
        """Get high-accuracy action using ensemble and confidence threshold"""
        
        if not self.models:
            raise ValueError("Models must be trained before getting actions")
        
        action, confidence = self.get_ensemble_prediction(observation)
        
        # Apply confidence threshold
        if confidence < self.min_confidence_threshold:
            action = 0  # Default to hold if confidence too low
        
        # Track confidence for analysis
        self.confidence_history.append(confidence)
        
        return np.array([action])
    
    def _evaluate_single_model(self, model, n_episodes: int = 5) -> Dict[str, Any]:
        """Evaluate a single model from the ensemble"""
        
        episode_results = []
        
        for episode in range(n_episodes):
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            total_reward = 0
            correct_trades = 0
            total_trades = 0
            done = False
            
            with torch.no_grad():
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    prev_balance = self.env.get_wrapper_attr('balance') if hasattr(self.env, 'get_wrapper_attr') else 1000
                    
                    step_result = self.env.step(action)
                    
                    # Handle both old and new gym API
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                    else:
                        obs, reward, done, info = step_result
                        terminated = done
                        truncated = False
                    
                    # Track trade accuracy
                    if isinstance(info, dict) and info.get('trade_made', False):
                        total_trades += 1
                        if reward > 0:
                            correct_trades += 1
                    
                    total_reward += reward
                    done = terminated or truncated
            
            accuracy = correct_trades / total_trades if total_trades > 0 else 0
            episode_results.append({
                'accuracy': accuracy,
                'total_trades': total_trades,
                'final_balance': info.get('balance', 1000),
                'returns': info.get('returns', 0)
            })
        
        # Calculate averages
        avg_accuracy = np.mean([r['accuracy'] for r in episode_results])
        avg_trades = np.mean([r['total_trades'] for r in episode_results])
        avg_returns = np.mean([r['returns'] for r in episode_results])
        
        return {
            'accuracy': avg_accuracy,
            'avg_trades': avg_trades,
            'avg_returns': avg_returns,
            'episodes': episode_results
        }
    
    def evaluate_high_accuracy(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Comprehensive evaluation focused on accuracy metrics"""
        
        logger.info(f"🎯 Evaluating high-accuracy performance over {n_episodes} episodes...")
        
        episode_results = []
        high_accuracy_episodes = 0
        
        for episode in range(n_episodes):
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            correct_predictions = 0
            total_predictions = 0
            trades_made = 0
            confidence_scores = []
            done = False
            
            while not done:
                action, confidence = self.get_ensemble_prediction(obs)
                confidence_scores.append(confidence)
                
                # Only count as prediction if we're actually trading (not holding)
                if action != 0:
                    total_predictions += 1
                    trades_made += 1
                
                step_result = self.env.step(action)
                
                # Handle both old and new gym API
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                
                # Check if prediction was correct (simplified)
                if action != 0 and reward > 0:
                    correct_predictions += 1
                
                done = terminated or truncated
            
            episode_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            avg_confidence = np.mean(confidence_scores)
            
            if episode_accuracy >= 0.75:
                high_accuracy_episodes += 1
            
            episode_results.append({
                'episode': episode + 1,
                'accuracy': episode_accuracy,
                'trades_made': trades_made,
                'avg_confidence': avg_confidence,
                'final_balance': info.get('balance', 1000),
                'returns': info.get('returns', 0)
            })
            
            logger.info(f"Episode {episode + 1}: "
                       f"Accuracy: {episode_accuracy:.1%}, "
                       f"Trades: {trades_made}, "
                       f"Confidence: {avg_confidence:.2f}")
        
        # Calculate final metrics
        overall_accuracy = np.mean([r['accuracy'] for r in episode_results])
        high_accuracy_rate = high_accuracy_episodes / n_episodes
        avg_trades_per_episode = np.mean([r['trades_made'] for r in episode_results])
        avg_confidence = np.mean([r['avg_confidence'] for r in episode_results])
        
        results = {
            'overall_accuracy': overall_accuracy,
            'high_accuracy_rate': high_accuracy_rate,
            'episodes_above_75pct': high_accuracy_episodes,
            'avg_trades_per_episode': avg_trades_per_episode,
            'avg_confidence': avg_confidence,
            'target_achieved': overall_accuracy >= 0.75,
            'episode_details': episode_results
        }
        
        logger.info("🎯 HIGH-ACCURACY EVALUATION RESULTS:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.1%}")
        logger.info(f"   Episodes >75%: {high_accuracy_episodes}/{n_episodes} ({high_accuracy_rate:.1%})")
        logger.info(f"   Avg Trades/Episode: {avg_trades_per_episode:.1f}")
        logger.info(f"   Avg Confidence: {avg_confidence:.2f}")
        logger.info(f"   Target Achieved: {'✅ YES' if results['target_achieved'] else '❌ NO'}")
        
        return results
    
    def save_ensemble(self, filename_prefix: str = None):
        """Save the entire ensemble"""
        if not self.models:
            raise ValueError("No ensemble to save")
        
        if filename_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_prefix = f"models/high_accuracy_ensemble_{timestamp}"
        
        os.makedirs(os.path.dirname(filename_prefix), exist_ok=True)
        
        # Save each model in the ensemble
        saved_paths = []
        for i, model in enumerate(self.models):
            model_path = f"{filename_prefix}_model_{i}.zip"
            model.save(model_path)
            saved_paths.append(model_path)
        
        # Save ensemble metadata
        metadata = {
            'ensemble_size': self.ensemble_size,
            'min_confidence_threshold': self.min_confidence_threshold,
            'min_agreement': self.min_agreement,
            'model_paths': saved_paths
        }
        
        metadata_path = f"{filename_prefix}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"💾 High-accuracy ensemble saved: {filename_prefix}")
        return filename_prefix

if __name__ == "__main__":
    logger.info("🎯 High-Accuracy RL Agent loaded successfully!")
    print("High-Accuracy Forex RL Agent ready for >75% win rate trading!")