"""
Technical Indicators Module for RL Trading Agent
Adds comprehensive technical analysis features
"""
import pandas as pd
import numpy as np
import ta
from typing import Dict, List
import pytz
from datetime import datetime
from config import CONFIG
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self, config=None):
        self.config = config or CONFIG
    
    def add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trading session features (Tokyo, London, NY, Sydney)
        """
        df = df.copy()
        
        # Convert to UTC if not already
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        
        # Extract hour from UTC timestamp
        df['hour_utc'] = df.index.hour
        
        # Session indicators
        for session, (start_hour, end_hour) in self.config.sessions.items():
            if start_hour <= end_hour:
                df[f'{session}_session'] = ((df['hour_utc'] >= start_hour) & 
                                          (df['hour_utc'] < end_hour)).astype(int)
            else:  # Session crosses midnight (like Sydney)
                df[f'{session}_session'] = ((df['hour_utc'] >= start_hour) | 
                                          (df['hour_utc'] < end_hour)).astype(int)
        
        # Session overlaps
        df['london_ny_overlap'] = (df['london_session'] & df['ny_session']).astype(int)
        df['tokyo_london_overlap'] = (df['tokyo_session'] & df['london_session']).astype(int)
        df['ny_tokyo_overlap'] = (df['ny_session'] & df['tokyo_session']).astype(int)
        
        # Major sessions (London + NY)
        df['major_sessions'] = (df['london_session'] | df['ny_session']).astype(int)
        
        # Session volume indicator (major sessions tend to have higher volume)
        df['session_weight'] = (df['london_session'] * 1.5 + 
                               df['ny_session'] * 1.5 + 
                               df['tokyo_session'] * 1.0 + 
                               df['sydney_session'] * 0.8 +
                               df['london_ny_overlap'] * 0.5)
        
        return df
    
    def add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic technical indicators
        """
        df = df.copy()
        
        # Handle different column naming conventions
        if 'mid_open' in df.columns:
            # New format (from data_fetcher)
            open_price = df['mid_open']
            high = df['mid_high']
            low = df['mid_low']  
            close = df['mid_close']
        elif 'open' in df.columns:
            # Existing format (from part3 data)
            open_price = df['open']
            high = df['high']
            low = df['low']  
            close = df['close']
        else:
            raise ValueError("Could not find price columns (expected 'mid_open' or 'open')")
        
        volume = df['volume']
        
        # Moving averages
        for period in self.config.sma_periods:
            df[f'sma_{period}'] = ta.trend.sma_indicator(close, window=period)
            
        for period in self.config.ema_periods:
            df[f'ema_{period}'] = ta.trend.ema_indicator(close, window=period)
        
        # RSI
        df['rsi'] = ta.momentum.rsi(close, window=self.config.rsi_period)
        
        # MACD
        macd = ta.trend.MACD(close, 
                            window_slow=self.config.macd_slow,
                            window_fast=self.config.macd_fast,
                            window_sign=self.config.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(high, low, close, 
                                                    window=self.config.atr_period)
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close)
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_percent'] = (close - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        
        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(close, volume)
        df['vwap'] = ta.volume.volume_weighted_average_price(high, low, close, volume)
        
        return df
    
    def add_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add advanced technical indicators
        """
        df = df.copy()
        
        # Handle different column naming conventions
        if 'mid_open' in df.columns:
            high = df['mid_high']
            low = df['mid_low']
            close = df['mid_close']
        elif 'open' in df.columns:
            high = df['high']
            low = df['low']
            close = df['close']
        else:
            raise ValueError("Could not find price columns")
        
        volume = df['volume']
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(high, low, close)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ADX (Average Directional Index)
        df['adx'] = ta.trend.adx(high, low, close)
        df['adx_pos'] = ta.trend.adx_pos(high, low, close)
        df['adx_neg'] = ta.trend.adx_neg(high, low, close)
        
        # Commodity Channel Index
        df['cci'] = ta.trend.cci(high, low, close)
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(high, low, close)
        
        # Aroon
        try:
            aroon = ta.trend.AroonIndicator(high=high, low=low, window=14)
            df['aroon_up'] = aroon.aroon_up()
            df['aroon_down'] = aroon.aroon_down()
        except Exception as e:
            logger.warning(f"Could not calculate Aroon indicator: {e}")
            df['aroon_up'] = 50.0  # Neutral value
            df['aroon_down'] = 50.0  # Neutral value
        
        return df
    
    def add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add pivot points and support/resistance levels
        """
        df = df.copy()
        
        # Calculate daily pivots using previous day's data
        df['date'] = df.index.date
        
        daily_data = df.groupby('date').agg({
            'mid_high': 'max',
            'mid_low': 'min',
            'mid_close': 'last'
        }).shift(1)  # Use previous day's data
        
        # Standard pivot points
        daily_data['pivot'] = (daily_data['mid_high'] + 
                              daily_data['mid_low'] + 
                              daily_data['mid_close']) / 3
        
        daily_data['r1'] = 2 * daily_data['pivot'] - daily_data['mid_low']
        daily_data['s1'] = 2 * daily_data['pivot'] - daily_data['mid_high']
        daily_data['r2'] = daily_data['pivot'] + (daily_data['mid_high'] - daily_data['mid_low'])
        daily_data['s2'] = daily_data['pivot'] - (daily_data['mid_high'] - daily_data['mid_low'])
        daily_data['r3'] = daily_data['mid_high'] + 2 * (daily_data['pivot'] - daily_data['mid_low'])
        daily_data['s3'] = daily_data['mid_low'] - 2 * (daily_data['mid_high'] - daily_data['pivot'])
        
        # Merge back to main dataframe
        df = df.merge(daily_data[['pivot', 'r1', 's1', 'r2', 's2', 'r3', 's3']], 
                     left_on='date', right_index=True, how='left')
        
        # Calculate distance to pivot levels
        close = df['mid_close']
        df['dist_to_pivot'] = (close - df['pivot']) / df['pivot']
        df['dist_to_r1'] = (close - df['r1']) / df['r1']
        df['dist_to_s1'] = (close - df['s1']) / df['s1']
        
        df.drop('date', axis=1, inplace=True)
        
        return df
    
    def add_fibonacci_levels(self, df: pd.DataFrame, lookback_period: int = 50) -> pd.DataFrame:
        """
        Add Fibonacci retracement levels
        """
        df = df.copy()
        
        high = df['mid_high']
        low = df['mid_low']
        close = df['mid_close']
        
        # Calculate rolling max and min
        rolling_max = high.rolling(window=lookback_period).max()
        rolling_min = low.rolling(window=lookback_period).min()
        
        # Fibonacci ratios
        fib_ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        
        for ratio in fib_ratios:
            # For uptrend (price closer to high)
            df[f'fib_{int(ratio*1000)}_up'] = rolling_max - (rolling_max - rolling_min) * ratio
            # For downtrend (price closer to low)
            df[f'fib_{int(ratio*1000)}_down'] = rolling_min + (rolling_max - rolling_min) * ratio
        
        # Distance to key Fibonacci levels
        df['dist_to_fib_618'] = np.minimum(
            abs(close - df['fib_618_up']),
            abs(close - df['fib_618_down'])
        ) / close
        
        df['dist_to_fib_382'] = np.minimum(
            abs(close - df['fib_382_up']),
            abs(close - df['fib_382_down'])
        ) / close
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features
        """
        df = df.copy()
        
        # Price changes
        df['mid_close_pct_change'] = df['mid_close'].pct_change()
        df['bid_ask_spread_pct'] = df['spread_close'] / df['mid_close']
        
        # Volatility measures
        df['price_range'] = (df['mid_high'] - df['mid_low']) / df['mid_close']
        df['true_range'] = np.maximum(
            df['mid_high'] - df['mid_low'],
            np.maximum(
                abs(df['mid_high'] - df['mid_close'].shift(1)),
                abs(df['mid_low'] - df['mid_close'].shift(1))
            )
        )
        
        # Candle patterns (basic)
        df['is_doji'] = (abs(df['mid_close'] - df['mid_open']) / 
                        (df['mid_high'] - df['mid_low']) < 0.1).astype(int)
        df['is_hammer'] = ((df['mid_close'] > df['mid_open']) & 
                          ((df['mid_open'] - df['mid_low']) > 2 * (df['mid_close'] - df['mid_open']))).astype(int)
        
        # Moving average crossovers
        for i, short_period in enumerate([21, 50]):
            for long_period in [50, 200]:
                if short_period < long_period:
                    short_ma = df[f'sma_{short_period}']
                    long_ma = df[f'sma_{long_period}']
                    df[f'ma_{short_period}_{long_period}_cross'] = (
                        (short_ma > long_ma).astype(int) - 
                        (short_ma < long_ma).astype(int)
                    )
        
        return df
    
    def add_all_indicators(self, df: pd.DataFrame, save_enhanced: bool = True) -> pd.DataFrame:
        """
        Add all technical indicators and optionally save the enhanced dataframe
        """
        logger.info("Adding session features...")
        enhanced_df = self.add_session_features(df.copy())
        
        logger.info("Adding basic indicators...")
        enhanced_df = self.add_basic_indicators(enhanced_df)
        
        logger.info("Adding advanced indicators...")
        enhanced_df = self.add_advanced_indicators(enhanced_df)
        
        logger.info("Adding pivot points...")
        enhanced_df = self.add_pivot_points(enhanced_df)
        
        logger.info("Adding Fibonacci levels...")
        enhanced_df = self.add_fibonacci_levels(enhanced_df)
        
        logger.info("Adding price features...")
        enhanced_df = self.add_price_features(enhanced_df)
        
        # Forward fill any NaN values
        enhanced_df = enhanced_df.ffill()
        
        # Drop any remaining NaN rows (usually at the beginning)
        enhanced_df = enhanced_df.dropna()
        
        logger.info(f"Final enhanced dataset shape: {enhanced_df.shape}")
        
        # Save enhanced dataframe if requested
        if save_enhanced:
            self.save_enhanced_data(enhanced_df)
        
        return enhanced_df
    
    def save_enhanced_data(self, df: pd.DataFrame, filename: str = None):
        """
        Save enhanced dataframe with technical indicators to pickle file
        """
        import os
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./data/{self.config.instrument}_{self.config.timeframe}_TA_{timestamp}.pkl"
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Save enhanced dataframe
        df.to_pickle(filename)
        logger.info(f"📊 Technical Analysis data saved to {filename}")
        logger.info(f"   - Enhanced columns: {len(df.columns)}")
        logger.info(f"   - Rows: {len(df):,} candles")
        logger.info(f"   - Date range: {df.index[0]} to {df.index[-1]}")
        
        return filename

if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    parser = argparse.ArgumentParser(description='Add Technical Indicators to Raw Data')
    parser.add_argument('input_file', nargs='?', help='Input raw data pickle file')
    parser.add_argument('--output', help='Output filename (optional)')
    
    args = parser.parse_args()
    
    if args.input_file:
        # Process specific file
        print(f"📊 Processing raw data file: {args.input_file}")
        
        if not os.path.exists(args.input_file):
            print(f"❌ File not found: {args.input_file}")
            sys.exit(1)
        
        data = pd.read_pickle(args.input_file)
        print(f"✅ Loaded: {len(data):,} candles, {len(data.columns)} columns")
        
        indicators = TechnicalIndicators()
        enhanced_data = indicators.add_all_indicators(data, save_enhanced=True)
        
        print(f"🎯 Processing complete!")
        print(f"   📊 Original: {len(data.columns)} columns")
        print(f"   🔬 Enhanced: {len(enhanced_data.columns)} columns")
        print(f"   ⚙️ Added: {len(enhanced_data.columns) - len(data.columns)} technical indicators")
    
    else:
        # Test mode - fetch sample data
        print("🧪 Testing technical indicators with sample data...")
        from data_fetcher import OandaDataFetcher
        
        fetcher = OandaDataFetcher()
        data = fetcher.fetch_candles(count=1000)
        
        if data is not None:
            indicators = TechnicalIndicators()
            enhanced_data = indicators.add_all_indicators(data, save_enhanced=True)
            
            print("Enhanced data columns:")
            for col in sorted(enhanced_data.columns):
                print(f"  {col}")
            print(f"Shape: {enhanced_data.shape}")
            print(enhanced_data.head())