"""
Forex Trading Environment for RL Agent
Realistic environment with ask/bid execution, risk management, and intraday focus
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from enum import Enum
import logging
from datetime import datetime, time
from config import CONFIG

logger = logging.getLogger(__name__)

class Action(Enum):
    HOLD = 0
    BUY = 1
    SELL = 2

class Position(Enum):
    FLAT = 0
    LONG = 1
    SHORT = 2

class Trade:
    def __init__(self, action: str, entry_price: float, stop_loss: float, 
                 take_profit: float, size: float, timestamp: datetime):
        self.action = action  # 'BUY' or 'SELL'
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.size = size
        self.timestamp = timestamp
        self.exit_price = None
        self.exit_timestamp = None
        self.pnl = 0.0
        self.is_closed = False
        self.exit_reason = None  # 'SL', 'TP', 'EOD', 'MANUAL'

class ForexTradingEnv(gym.Env):
    """
    Forex Trading Environment with realistic ask/bid execution
    """
    
    def __init__(self, data: pd.DataFrame, config=None):
        super().__init__()
        
        self.config = config or CONFIG
        self.data = data.copy()
        self.original_data_len = len(data)
        
        # Environment setup
        self.lookback_window = self.config.lookback_window
        self.max_steps = len(data) - self.lookback_window
        self.current_step = 0
        
        # Account setup
        self.initial_balance = self.config.initial_balance
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        # Position and trade management
        self.position = Position.FLAT
        self.current_trade = None
        self.trades_history = []
        self.no_trade_steps = 0
        
        # Daily tracking for intraday focus
        self.current_date = None
        self.daily_trades = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        # Action and observation spaces
        n_features = len(self.data.columns)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.lookback_window, n_features),
            dtype=np.float32
        )
        
        # Action space: [action_type, stop_loss_pct, take_profit_pct]
        # action_type: 0=HOLD, 1=BUY, 2=SELL
        # stop_loss_pct: 0.001 to 0.02 (0.1% to 2%)
        # take_profit_pct: 0.005 to 0.05 (0.5% to 5%)
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, 0.001, 0.005], dtype=np.float32),
            high=np.array([2.0, 0.02, 0.05], dtype=np.float32),
            dtype=np.float32
        )
        
        logger.info(f"Environment initialized with {len(data)} samples")
    
    def reset(self, seed=None, **kwargs):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.peak_balance = self.initial_balance
        self.max_drawdown = 0.0
        
        self.position = Position.FLAT
        self.current_trade = None
        self.trades_history = []
        self.daily_trades = []
        self.no_trade_steps = 0
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        self.current_date = None
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment"""
        
        # Parse action
        action_type = int(round(action[0]))
        stop_loss_pct = float(action[1])
        take_profit_pct = float(action[2])
        
        # Ensure action is valid
        action_type = np.clip(action_type, 0, 2)
        stop_loss_pct = np.clip(stop_loss_pct, 0.001, self.config.max_stop_loss_pct)
        take_profit_pct = np.clip(take_profit_pct, 0.005, 0.05)
        
        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_timestamp = current_data.name
        current_date = current_timestamp.date()
        
        # Check for end of day - close any open positions
        if self.current_date is not None and current_date != self.current_date:
            if self.current_trade is not None:
                self._close_trade_eod(current_data)
            self.daily_trades = []
        
        self.current_date = current_date
        
        # Check existing trade for SL/TP hits
        if self.current_trade is not None:
            self._check_sl_tp(current_data)
        
        # Calculate reward
        reward = self._calculate_reward(action_type, current_data)
        
        # Execute new action if no current position
        if self.position == Position.FLAT and action_type != Action.HOLD.value:
            self._execute_trade(action_type, stop_loss_pct, take_profit_pct, current_data)
        
        # Update counters
        if action_type == Action.HOLD.value:
            self.no_trade_steps += 1
        else:
            self.no_trade_steps = 0
        
        # Update equity and drawdown
        self._update_equity(current_data)
        self._update_drawdown()
        
        # Move to next step
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation window"""
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        obs = self.data.iloc[start_idx:end_idx].values
        
        # Normalize observation (simple approach)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)
    
    def _execute_trade(self, action_type: int, stop_loss_pct: float, 
                      take_profit_pct: float, current_data: pd.Series):
        """Execute a trade with realistic ask/bid execution"""
        
        if action_type == Action.BUY.value:
            # Buy at ask price
            entry_price = current_data['ask_close']
            action_str = 'BUY'
            self.position = Position.LONG
            
            # Calculate SL/TP levels
            stop_loss = entry_price * (1 - stop_loss_pct)
            take_profit = entry_price * (1 + take_profit_pct)
            
        elif action_type == Action.SELL.value:
            # Sell at bid price
            entry_price = current_data['bid_close']
            action_str = 'SELL'
            self.position = Position.SHORT
            
            # Calculate SL/TP levels
            stop_loss = entry_price * (1 + stop_loss_pct)
            take_profit = entry_price * (1 - take_profit_pct)
        
        else:
            return  # No action
        
        # Calculate position size based on risk
        risk_amount = self.balance * stop_loss_pct
        pip_value = self._calculate_pip_value(current_data)
        stop_loss_pips = abs(entry_price - stop_loss) / pip_value
        
        # Position size calculation (simplified)
        position_size = min(risk_amount / (stop_loss_pips * pip_value), 
                           self.balance * 0.1)  # Max 10% of balance per trade
        
        # Create trade
        self.current_trade = Trade(
            action=action_str,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            size=position_size,
            timestamp=current_data.name
        )
        
        self.total_trades += 1
        logger.debug(f"Opened {action_str} trade at {entry_price}, SL: {stop_loss:.5f}, TP: {take_profit:.5f}")
    
    def _check_sl_tp(self, current_data: pd.Series):
        """Check if current trade hits SL or TP"""
        if self.current_trade is None:
            return
        
        current_bid = current_data['bid_close']
        current_ask = current_data['ask_close']
        
        if self.position == Position.LONG:
            # For long positions, exit at bid price
            if current_bid <= self.current_trade.stop_loss:
                self._close_trade(current_bid, 'SL', current_data.name)
            elif current_bid >= self.current_trade.take_profit:
                self._close_trade(current_bid, 'TP', current_data.name)
                
        elif self.position == Position.SHORT:
            # For short positions, exit at ask price
            if current_ask >= self.current_trade.stop_loss:
                self._close_trade(current_ask, 'SL', current_data.name)
            elif current_ask <= self.current_trade.take_profit:
                self._close_trade(current_ask, 'TP', current_data.name)
    
    def _close_trade_eod(self, current_data: pd.Series):
        """Close trade at end of day"""
        if self.current_trade is None:
            return
        
        if self.position == Position.LONG:
            exit_price = current_data['bid_close']
        else:
            exit_price = current_data['ask_close']
        
        self._close_trade(exit_price, 'EOD', current_data.name)
    
    def _close_trade(self, exit_price: float, reason: str, timestamp):
        """Close current trade and calculate PnL"""
        if self.current_trade is None:
            return
        
        trade = self.current_trade
        trade.exit_price = exit_price
        trade.exit_timestamp = timestamp
        trade.exit_reason = reason
        
        # Calculate PnL
        if trade.action == 'BUY':
            trade.pnl = (exit_price - trade.entry_price) * trade.size
        else:  # SELL
            trade.pnl = (trade.entry_price - exit_price) * trade.size
        
        # Update balance and statistics
        self.balance += trade.pnl
        self.total_pnl += trade.pnl
        
        if trade.pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Add to history
        trade.is_closed = True
        self.trades_history.append(trade)
        self.daily_trades.append(trade)
        
        # Reset position
        self.position = Position.FLAT
        self.current_trade = None
        
        logger.debug(f"Closed {trade.action} trade: PnL = {trade.pnl:.2f}, Reason: {reason}")
    
    def _calculate_reward(self, action_type: int, current_data: pd.Series) -> float:
        """Calculate reward for the current action"""
        reward = 0.0
        
        # Reward/penalty for trading actions
        if self.current_trade is not None and self.current_trade.is_closed:
            # Reward for profitable trades, penalty for losses
            if self.current_trade.pnl > 0:
                reward += self.current_trade.pnl / self.initial_balance * 100  # Scale reward
            else:
                reward += self.current_trade.pnl / self.initial_balance * 100  # Negative reward for losses
        
        # Exponential penalty for not trading
        if action_type == Action.HOLD.value:
            if self.no_trade_steps > 0:
                penalty = self.config.no_trade_penalty_base * (
                    self.config.no_trade_penalty_exp ** min(self.no_trade_steps, 
                                                           self.config.max_no_trade_steps)
                )
                reward -= penalty
        
        # Bonus for staying within session hours (major sessions preferred)
        session_bonus = current_data.get('session_weight', 1.0) * 0.001
        reward += session_bonus
        
        # Penalty for excessive drawdown
        if self.max_drawdown > 0.1:  # More than 10% drawdown
            reward -= (self.max_drawdown - 0.1) * 10
        
        # Major bonus for doubling account
        if self.balance >= self.initial_balance * self.config.target_multiplier:
            reward += 100  # Large bonus for reaching target
        
        return reward
    
    def _update_equity(self, current_data: pd.Series):
        """Update current equity including unrealized PnL"""
        unrealized_pnl = 0.0
        
        if self.current_trade is not None:
            if self.position == Position.LONG:
                unrealized_pnl = (current_data['bid_close'] - self.current_trade.entry_price) * self.current_trade.size
            elif self.position == Position.SHORT:
                unrealized_pnl = (self.current_trade.entry_price - current_data['ask_close']) * self.current_trade.size
        
        self.equity = self.balance + unrealized_pnl
        
        if self.equity > self.peak_balance:
            self.peak_balance = self.equity
    
    def _update_drawdown(self):
        """Update maximum drawdown"""
        if self.peak_balance > 0:
            current_drawdown = (self.peak_balance - self.equity) / self.peak_balance
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
    
    def _calculate_pip_value(self, current_data: pd.Series) -> float:
        """Calculate pip value for position sizing"""
        # Simplified pip value calculation
        # For EUR/USD, 1 pip = 0.0001
        return 0.0001
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if account is blown (balance < 10% of initial)
        if self.balance < self.initial_balance * 0.1:
            return True
        
        # Terminate if target reached (account doubled)
        if self.balance >= self.initial_balance * self.config.target_multiplier:
            return True
        
        return False
    
    def _is_truncated(self) -> bool:
        """Check if episode should be truncated (reached end of data)"""
        return self.current_step >= self.max_steps
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment information"""
        win_rate = self.winning_trades / max(self.total_trades, 1)
        
        return {
            'balance': self.balance,
            'equity': self.equity,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'max_drawdown': self.max_drawdown,
            'current_position': self.position.name,
            'no_trade_steps': self.no_trade_steps,
            'peak_balance': self.peak_balance,
            'returns': (self.balance - self.initial_balance) / self.initial_balance,
            'daily_trades_count': len(self.daily_trades)
        }

if __name__ == "__main__":
    # Test the environment
    from data_fetcher import OandaDataFetcher
    from technical_indicators import TechnicalIndicators
    
    # Fetch and prepare data
    fetcher = OandaDataFetcher()
    data = fetcher.fetch_candles(count=500)
    
    if data is not None:
        indicators = TechnicalIndicators()
        enhanced_data = indicators.add_all_indicators(data)
        
        # Create and test environment
        env = ForexTradingEnv(enhanced_data)
        obs, info = env.reset()
        
        print(f"Environment created successfully!")
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Initial info: {info}")
        
        # Test a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Reward={reward:.4f}, Balance={info['balance']:.2f}, Trades={info['total_trades']}")
            
            if terminated or truncated:
                break
    else:
        print("Failed to fetch data for testing")