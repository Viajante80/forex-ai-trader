"""
Comprehensive Logging System for RL Forex Trading Agent
Tracks win rate, trades, account values, and performance metrics
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import CONFIG

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rl_ml/logs/trading.log'),
        logging.StreamHandler()
    ]
)

class TradingLogger:
    """
    Comprehensive logging system for trading performance
    """
    
    def __init__(self, config=None, log_file: str = None):
        self.config = config or CONFIG
        self.logger = logging.getLogger(__name__)
        
        # Create logs directory if it doesn't exist
        os.makedirs('rl_ml/logs', exist_ok=True)
        
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f'rl_ml/logs/trading_session_{timestamp}.json'
        
        self.log_file = log_file
        
        # Initialize logging data structures
        self.session_data = {
            'session_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'start_time': datetime.now().isoformat(),
            'config': self._config_to_dict(),
            'episodes': [],
            'trades': [],
            'performance_metrics': {},
            'training_progress': []
        }
        
        # Performance tracking
        self.episode_count = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # Episode tracking
        self.current_episode = None
        
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization"""
        config_dict = {}
        for key, value in self.config.__dict__.items():
            if isinstance(value, (str, int, float, bool, list)):
                config_dict[key] = value
            elif isinstance(value, dict):
                config_dict[key] = value
            else:
                config_dict[key] = str(value)
        return config_dict
    
    def start_episode(self, episode_id: int, initial_balance: float = None):
        """Start logging a new episode"""
        self.episode_count += 1
        self.current_episode = {
            'episode_id': episode_id,
            'start_time': datetime.now().isoformat(),
            'initial_balance': initial_balance or self.config.initial_balance,
            'trades_in_episode': [],
            'step_data': [],
            'final_metrics': {}
        }
        
        self.logger.info(f"Started episode {episode_id} with initial balance: ${initial_balance or self.config.initial_balance:.2f}")
    
    def log_step(self, step: int, observation: np.ndarray, action: np.ndarray, 
                 reward: float, info: Dict[str, Any]):
        """Log data for each step"""
        if self.current_episode is None:
            return
        
        step_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'action': action.tolist() if isinstance(action, np.ndarray) else action,
            'reward': float(reward),
            'balance': float(info.get('balance', 0)),
            'equity': float(info.get('equity', 0)),
            'total_trades': int(info.get('total_trades', 0)),
            'win_rate': float(info.get('win_rate', 0)),
            'max_drawdown': float(info.get('max_drawdown', 0)),
            'position': info.get('current_position', 'FLAT'),
            'no_trade_steps': int(info.get('no_trade_steps', 0)),
            'daily_trades_count': int(info.get('daily_trades_count', 0)),
            'returns': float(info.get('returns', 0))
        }
        
        self.current_episode['step_data'].append(step_data)
        
        # Log significant events
        if info.get('total_trades', 0) > len(self.current_episode['trades_in_episode']):
            self.logger.info(f"Episode {self.current_episode['episode_id']}, Step {step}: "
                           f"New trade executed. Balance: ${info['balance']:.2f}, "
                           f"Win Rate: {info['win_rate']:.3f}")
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log individual trade data"""
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'action': trade_data.get('action', 'UNKNOWN'),
            'entry_price': float(trade_data.get('entry_price', 0)),
            'exit_price': float(trade_data.get('exit_price', 0)),
            'stop_loss': float(trade_data.get('stop_loss', 0)),
            'take_profit': float(trade_data.get('take_profit', 0)),
            'size': float(trade_data.get('size', 0)),
            'pnl': float(trade_data.get('pnl', 0)),
            'exit_reason': trade_data.get('exit_reason', 'UNKNOWN'),
            'duration_minutes': trade_data.get('duration_minutes', 0),
            'session_weight': float(trade_data.get('session_weight', 1.0))
        }
        
        self.session_data['trades'].append(trade_log)
        
        if self.current_episode:
            self.current_episode['trades_in_episode'].append(trade_log)
        
        # Update trade statistics
        self.total_trades += 1
        if trade_log['pnl'] > 0:
            self.winning_trades += 1
            trade_status = "WINNING"
        else:
            self.losing_trades += 1
            trade_status = "LOSING"
        
        self.logger.info(f"{trade_status} Trade: {trade_log['action']} "
                        f"Entry: {trade_log['entry_price']:.5f}, "
                        f"Exit: {trade_log['exit_price']:.5f}, "
                        f"PnL: ${trade_log['pnl']:.2f}, "
                        f"Reason: {trade_log['exit_reason']}")
    
    def end_episode(self, final_info: Dict[str, Any]):
        """End current episode and log final metrics"""
        if self.current_episode is None:
            return
        
        # Calculate episode metrics
        episode_metrics = {
            'end_time': datetime.now().isoformat(),
            'final_balance': float(final_info.get('balance', 0)),
            'final_equity': float(final_info.get('equity', 0)),
            'total_trades': int(final_info.get('total_trades', 0)),
            'winning_trades': int(final_info.get('winning_trades', 0)),
            'losing_trades': int(final_info.get('losing_trades', 0)),
            'win_rate': float(final_info.get('win_rate', 0)),
            'total_pnl': float(final_info.get('total_pnl', 0)),
            'max_drawdown': float(final_info.get('max_drawdown', 0)),
            'returns': float(final_info.get('returns', 0)),
            'episode_duration': len(self.current_episode['step_data']),
            'trades_per_day': len(self.current_episode['trades_in_episode']) / max(1, len(self.current_episode['step_data']) / 288)  # Assuming 5-min candles
        }
        
        self.current_episode['final_metrics'] = episode_metrics
        self.session_data['episodes'].append(self.current_episode)
        
        self.logger.info(f"Episode {self.current_episode['episode_id']} completed:")
        self.logger.info(f"  Final Balance: ${episode_metrics['final_balance']:.2f}")
        self.logger.info(f"  Total Trades: {episode_metrics['total_trades']}")
        self.logger.info(f"  Win Rate: {episode_metrics['win_rate']:.3f}")
        self.logger.info(f"  Returns: {episode_metrics['returns']:.3f}")
        self.logger.info(f"  Max Drawdown: {episode_metrics['max_drawdown']:.3f}")
        
        self.current_episode = None
        
        # Save session data
        self.save_session_data()
    
    def log_training_progress(self, timestep: int, metrics: Dict[str, Any]):
        """Log training progress metrics"""
        progress_data = {
            'timestep': timestep,
            'timestamp': datetime.now().isoformat(),
            'mean_reward': float(metrics.get('mean_reward', 0)),
            'mean_balance': float(metrics.get('mean_balance', 0)),
            'mean_win_rate': float(metrics.get('mean_win_rate', 0)),
            'mean_trades': float(metrics.get('mean_trades', 0)),
            'learning_rate': float(metrics.get('learning_rate', 0)),
            'entropy': float(metrics.get('entropy', 0))
        }
        
        self.session_data['training_progress'].append(progress_data)
        
        self.logger.info(f"Training Progress - Step {timestep}: "
                        f"Mean Reward: {progress_data['mean_reward']:.4f}, "
                        f"Mean Balance: ${progress_data['mean_balance']:.2f}, "
                        f"Mean Win Rate: {progress_data['mean_win_rate']:.3f}")
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.session_data['episodes']:
            return {}
        
        episodes_df = pd.DataFrame([ep['final_metrics'] for ep in self.session_data['episodes']])
        trades_df = pd.DataFrame(self.session_data['trades']) if self.session_data['trades'] else pd.DataFrame()
        
        metrics = {
            'total_episodes': len(self.session_data['episodes']),
            'total_trades': len(self.session_data['trades']),
            'overall_win_rate': self.winning_trades / max(self.total_trades, 1),
            'profitable_episodes': len(episodes_df[episodes_df['returns'] > 0]) if not episodes_df.empty else 0,
            'episode_profitability_rate': len(episodes_df[episodes_df['returns'] > 0]) / max(len(episodes_df), 1) if not episodes_df.empty else 0,
        }
        
        if not episodes_df.empty:
            metrics.update({
                'mean_final_balance': float(episodes_df['final_balance'].mean()),
                'std_final_balance': float(episodes_df['final_balance'].std()),
                'best_episode_return': float(episodes_df['returns'].max()),
                'worst_episode_return': float(episodes_df['returns'].min()),
                'mean_episode_return': float(episodes_df['returns'].mean()),
                'mean_max_drawdown': float(episodes_df['max_drawdown'].mean()),
                'max_drawdown_ever': float(episodes_df['max_drawdown'].max()),
                'mean_trades_per_episode': float(episodes_df['total_trades'].mean()),
                'consistency_score': float(episodes_df['returns'].std()),  # Lower is more consistent
            })
        
        if not trades_df.empty:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            metrics.update({
                'average_winning_trade': float(winning_trades['pnl'].mean()) if not winning_trades.empty else 0,
                'average_losing_trade': float(losing_trades['pnl'].mean()) if not losing_trades.empty else 0,
                'largest_win': float(trades_df['pnl'].max()),
                'largest_loss': float(trades_df['pnl'].min()),
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty and losing_trades['pnl'].sum() != 0 else float('inf'),
                'average_trade_duration': float(trades_df['duration_minutes'].mean()) if 'duration_minutes' in trades_df.columns else 0
            })
        
        self.session_data['performance_metrics'] = metrics
        return metrics
    
    def generate_performance_report(self) -> str:
        """Generate a comprehensive performance report"""
        metrics = self.calculate_performance_metrics()
        
        report = f"""
=== FOREX RL TRADING AGENT PERFORMANCE REPORT ===
Session ID: {self.session_data['session_id']}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE:
- Total Episodes: {metrics.get('total_episodes', 0)}
- Total Trades: {metrics.get('total_trades', 0)}
- Overall Win Rate: {metrics.get('overall_win_rate', 0):.3f}
- Profitable Episodes: {metrics.get('profitable_episodes', 0)}/{metrics.get('total_episodes', 0)}
- Episode Profitability Rate: {metrics.get('episode_profitability_rate', 0):.3f}

BALANCE & RETURNS:
- Mean Final Balance: ${metrics.get('mean_final_balance', 0):.2f}
- Best Episode Return: {metrics.get('best_episode_return', 0):.3f}
- Worst Episode Return: {metrics.get('worst_episode_return', 0):.3f}
- Mean Episode Return: {metrics.get('mean_episode_return', 0):.3f}

RISK METRICS:
- Mean Max Drawdown: {metrics.get('mean_max_drawdown', 0):.3f}
- Maximum Drawdown Ever: {metrics.get('max_drawdown_ever', 0):.3f}
- Consistency Score: {metrics.get('consistency_score', 0):.4f}

TRADING METRICS:
- Mean Trades per Episode: {metrics.get('mean_trades_per_episode', 0):.1f}
- Average Winning Trade: ${metrics.get('average_winning_trade', 0):.2f}
- Average Losing Trade: ${metrics.get('average_losing_trade', 0):.2f}
- Largest Win: ${metrics.get('largest_win', 0):.2f}
- Largest Loss: ${metrics.get('largest_loss', 0):.2f}
- Profit Factor: {metrics.get('profit_factor', 0):.2f}
- Average Trade Duration: {metrics.get('average_trade_duration', 0):.1f} minutes

CONFIGURATION:
- Instrument: {self.config.instrument}
- Timeframe: {self.config.timeframe}
- Initial Balance: ${self.config.initial_balance:.2f}
- Max Stop Loss: {self.config.max_stop_loss_pct:.1%}
- Target Multiplier: {self.config.target_multiplier}x

=== END REPORT ===
        """
        
        return report
    
    def save_session_data(self):
        """Save session data to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.session_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save session data: {e}")
    
    def create_performance_charts(self, save_path: str = None):
        """Create performance visualization charts"""
        if not self.session_data['episodes']:
            self.logger.warning("No episode data available for charting")
            return
        
        episodes_df = pd.DataFrame([ep['final_metrics'] for ep in self.session_data['episodes']])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Balance Over Episodes', 'Win Rate Over Episodes',
                           'Returns Distribution', 'Drawdown Over Episodes'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Balance over episodes
        fig.add_trace(
            go.Scatter(y=episodes_df['final_balance'], mode='lines+markers',
                      name='Final Balance', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Win rate over episodes
        fig.add_trace(
            go.Scatter(y=episodes_df['win_rate'], mode='lines+markers',
                      name='Win Rate', line=dict(color='green')),
            row=1, col=2
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(x=episodes_df['returns'], name='Returns', 
                        marker=dict(color='orange')),
            row=2, col=1
        )
        
        # Max drawdown over episodes
        fig.add_trace(
            go.Scatter(y=episodes_df['max_drawdown'], mode='lines+markers',
                      name='Max Drawdown', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Trading Performance Dashboard")
        
        if save_path is None:
            save_path = f"rl_ml/logs/performance_charts_{self.session_data['session_id']}.html"
        
        fig.write_html(save_path)
        self.logger.info(f"Performance charts saved to {save_path}")
    
    def export_trades_csv(self, filename: str = None):
        """Export trades data to CSV"""
        if not self.session_data['trades']:
            self.logger.warning("No trades data to export")
            return
        
        trades_df = pd.DataFrame(self.session_data['trades'])
        
        if filename is None:
            filename = f"rl_ml/logs/trades_{self.session_data['session_id']}.csv"
        
        trades_df.to_csv(filename, index=False)
        self.logger.info(f"Trades data exported to {filename}")

if __name__ == "__main__":
    # Test the logging system
    logger = TradingLogger()
    
    # Simulate some episode data
    logger.start_episode(1, 1000.0)
    
    # Simulate some steps
    for step in range(5):
        fake_obs = np.random.random((100, 85))
        fake_action = np.array([1.0, 0.01, 0.02])
        fake_reward = np.random.random() * 0.1
        fake_info = {
            'balance': 1000 + step * 10,
            'equity': 1000 + step * 10,
            'total_trades': step,
            'win_rate': 0.6,
            'max_drawdown': 0.05,
            'current_position': 'LONG' if step % 2 else 'FLAT',
            'no_trade_steps': 0,
            'daily_trades_count': step,
            'returns': step * 0.01
        }
        logger.log_step(step, fake_obs, fake_action, fake_reward, fake_info)
    
    # Log a trade
    trade_data = {
        'action': 'BUY',
        'entry_price': 1.16050,
        'exit_price': 1.16100,
        'stop_loss': 1.15950,
        'take_profit': 1.16150,
        'size': 10000,
        'pnl': 50.0,
        'exit_reason': 'TP',
        'duration_minutes': 25,
        'session_weight': 1.5
    }
    logger.log_trade(trade_data)
    
    # End episode
    final_info = {
        'balance': 1050.0,
        'equity': 1050.0,
        'total_trades': 5,
        'winning_trades': 3,
        'losing_trades': 2,
        'win_rate': 0.6,
        'total_pnl': 50.0,
        'max_drawdown': 0.05,
        'returns': 0.05
    }
    logger.end_episode(final_info)
    
    # Generate report
    report = logger.generate_performance_report()
    print(report)
    
    print("Trading logger test completed successfully!")