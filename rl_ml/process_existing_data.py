"""
Process existing EUR_USD data and add technical indicators
Creates enhanced dataset for RL training
"""
import pandas as pd
import numpy as np
import ta
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_compatible_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators compatible with existing data format
    """
    logger.info("Processing existing data format...")
    enhanced_df = df.copy()
    
    # Price columns (already available: open, high, low, close)
    high = df['high']
    low = df['low'] 
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    
    logger.info("Adding basic indicators...")
    
    # Moving Averages (add more periods)
    for period in [5, 10, 21, 50, 100, 200]:
        if f'sma_{period}' not in enhanced_df.columns:
            enhanced_df[f'sma_{period}'] = ta.trend.sma_indicator(close, window=period)
        if period <= 50:  # EMA for shorter periods
            enhanced_df[f'ema_{period}'] = ta.trend.ema_indicator(close, window=period)
    
    # RSI (add if not exists or add different periods)
    if 'rsi' not in enhanced_df.columns:
        enhanced_df['rsi'] = ta.momentum.rsi(close, window=14)
    enhanced_df['rsi_9'] = ta.momentum.rsi(close, window=9)
    enhanced_df['rsi_21'] = ta.momentum.rsi(close, window=21)
    
    # MACD
    macd = ta.trend.MACD(close)
    enhanced_df['macd'] = macd.macd()
    enhanced_df['macd_signal'] = macd.macd_signal()
    enhanced_df['macd_histogram'] = macd.macd_diff()
    
    # ATR
    enhanced_df['atr'] = ta.volatility.average_true_range(high, low, close, window=14)
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close)
    enhanced_df['bb_upper'] = bb.bollinger_hband()
    enhanced_df['bb_middle'] = bb.bollinger_mavg() 
    enhanced_df['bb_lower'] = bb.bollinger_lband()
    enhanced_df['bb_width'] = bb.bollinger_wband()
    enhanced_df['bb_percent'] = bb.bollinger_pband()
    
    logger.info("Adding advanced indicators...")
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(high, low, close)
    enhanced_df['stoch_k'] = stoch.stoch()
    enhanced_df['stoch_d'] = stoch.stoch_signal()
    
    # ADX
    enhanced_df['adx'] = ta.trend.adx(high, low, close)
    enhanced_df['adx_pos'] = ta.trend.adx_pos(high, low, close)
    enhanced_df['adx_neg'] = ta.trend.adx_neg(high, low, close)
    
    # CCI
    enhanced_df['cci'] = ta.trend.cci(high, low, close)
    
    # Williams %R
    enhanced_df['williams_r'] = ta.momentum.williams_r(high, low, close)
    
    # OBV (update existing if needed)
    if 'obv' not in enhanced_df.columns:
        enhanced_df['obv'] = ta.volume.on_balance_volume(close, volume)
    
    # VWAP (update existing if needed)
    if 'vwap' not in enhanced_df.columns:
        enhanced_df['vwap'] = ta.volume.volume_weighted_average_price(high, low, close, volume)
    
    # Session features
    logger.info("Adding session features...")
    enhanced_df['hour_utc'] = enhanced_df.index.hour
    
    # Trading sessions (UTC hours)
    enhanced_df['tokyo_session'] = ((enhanced_df['hour_utc'] >= 23) | (enhanced_df['hour_utc'] < 8)).astype(int)
    enhanced_df['london_session'] = ((enhanced_df['hour_utc'] >= 7) & (enhanced_df['hour_utc'] < 16)).astype(int)
    enhanced_df['ny_session'] = ((enhanced_df['hour_utc'] >= 12) & (enhanced_df['hour_utc'] < 21)).astype(int)
    enhanced_df['sydney_session'] = ((enhanced_df['hour_utc'] >= 21) | (enhanced_df['hour_utc'] < 6)).astype(int)
    
    # Session overlaps
    enhanced_df['london_ny_overlap'] = (enhanced_df['london_session'] & enhanced_df['ny_session']).astype(int)
    enhanced_df['tokyo_london_overlap'] = (enhanced_df['tokyo_session'] & enhanced_df['london_session']).astype(int)
    enhanced_df['major_sessions'] = (enhanced_df['london_session'] | enhanced_df['ny_session']).astype(int)
    
    # Price action features
    logger.info("Adding price action features...")
    
    enhanced_df['price_change'] = close.pct_change()
    enhanced_df['price_range'] = (high - low) / close
    enhanced_df['upper_shadow'] = (high - np.maximum(open_price, close)) / close
    enhanced_df['lower_shadow'] = (np.minimum(open_price, close) - low) / close
    enhanced_df['body_size'] = abs(close - open_price) / close
    
    # Candle patterns
    enhanced_df['is_doji'] = (abs(close - open_price) / (high - low) < 0.1).astype(int)
    enhanced_df['is_hammer'] = ((close > open_price) & 
                               ((open_price - low) > 2 * (close - open_price))).astype(int)
    
    # Moving average crosses
    enhanced_df['sma_21_50_cross'] = ((enhanced_df['sma_21'] > enhanced_df['sma_50']).astype(int) - 
                                     (enhanced_df['sma_21'] < enhanced_df['sma_50']).astype(int))
    
    # Momentum indicators
    enhanced_df['momentum_10'] = close / close.shift(10) - 1
    enhanced_df['momentum_20'] = close / close.shift(20) - 1
    
    # Volatility
    enhanced_df['volatility_10'] = close.rolling(10).std()
    enhanced_df['volatility_20'] = close.rolling(20).std()
    
    # Clean up
    enhanced_df = enhanced_df.ffill().dropna()
    
    logger.info(f"Enhanced dataset: {len(enhanced_df.columns)} columns, {len(enhanced_df)} rows")
    
    return enhanced_df

def main():
    """Process existing data and create enhanced version"""
    
    print("📊 Processing existing EUR_USD data for RL training")
    print("=" * 60)
    
    # Load existing data
    input_file = "../part3/data/EUR_USD/EUR_USD_M5_with_bid_ask.pkl"
    print(f"Loading: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    data = pd.read_pickle(input_file)
    print(f"✅ Loaded: {len(data):,} candles, {len(data.columns)} columns")
    print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
    
    # Process with indicators
    print(f"\n⚙️ Adding technical indicators...")
    enhanced_data = add_compatible_indicators(data)
    
    # Save enhanced data
    os.makedirs("rl_ml/data", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"rl_ml/data/EUR_USD_M5_enhanced_{timestamp}.pkl"
    
    enhanced_data.to_pickle(output_file)
    
    print(f"\n🎯 Processing Complete!")
    print(f"📊 Original: {len(data.columns)} columns")
    print(f"🔬 Enhanced: {len(enhanced_data.columns)} columns") 
    print(f"⚙️ Added: {len(enhanced_data.columns) - len(data.columns)} technical indicators")
    print(f"💾 Saved: {output_file}")
    
    # Show sample of new columns
    new_columns = [col for col in enhanced_data.columns if col not in data.columns]
    print(f"\n📋 New Technical Indicators:")
    for i, col in enumerate(sorted(new_columns)):
        if i % 4 == 0:
            print()
        print(f"  {col:<20}", end="")
    print(f"\n\n✅ Ready for RL training!")

if __name__ == "__main__":
    main()