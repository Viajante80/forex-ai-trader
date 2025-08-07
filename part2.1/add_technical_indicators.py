import pandas as pd
import os
import glob
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolumeIndicator
from ta.utils import dropna
import warnings
warnings.filterwarnings('ignore')

# Directory settings
input_dir = "../oanda_historical_data"
output_dir = "../trading_ready_data"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define currency pairs and timeframes (same as in the data downloader)
currency_pairs = ["EUR_USD", "GBP_USD", "EUR_GBP"]
timeframes = ["M5", "M15", "M30", "H1", "H4", "D", "W"]

# Moving average periods requested
MA_PERIODS = [50, 80, 100, 200]


def _compute_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Compute classic daily pivot points on normalized prices and align to intraday rows.
    For non-daily data, we compute previous day's pivots and forward-fill.
    """
    if df.index.tz is None:
        # assume UTC if naive
        df = df.tz_localize('UTC')
    # Daily resample of norm prices
    daily = df[['norm_high', 'norm_low', 'norm_close']].resample('D').agg({
        'norm_high': 'max', 'norm_low': 'min', 'norm_close': 'last'
    }).dropna()
    piv = pd.DataFrame(index=daily.index)
    H, L, C = daily['norm_high'], daily['norm_low'], daily['norm_close']
    P = (H + L + C) / 3
    R1 = 2*P - L
    S1 = 2*P - H
    R2 = P + (H - L)
    S2 = P - (H - L)
    piv['pivot_p'] = P.shift(1)
    piv['pivot_r1'] = R1.shift(1)
    piv['pivot_s1'] = S1.shift(1)
    piv['pivot_r2'] = R2.shift(1)
    piv['pivot_s2'] = S2.shift(1)
    # Map back to original index by forward filling
    piv = piv.reindex(df.index, method='ffill')
    return piv


def _compute_fibonacci_levels(df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
    """Compute rolling Fibonacci retracement levels based on last `window` bars.
    Returns columns fib_0, fib_0236, fib_0382, fib_0500, fib_0618, fib_0786, fib_1.
    """
    highs = df['norm_high'].rolling(window)
    lows = df['norm_low'].rolling(window)
    H = highs.max()
    L = lows.min()
    rng = H - L
    levels = pd.DataFrame(index=df.index)
    levels['fib_0'] = L
    levels['fib_1'] = H
    for ratio, name in [(0.236, 'fib_0236'), (0.382, 'fib_0382'), (0.5, 'fib_0500'), (0.618, 'fib_0618'), (0.786, 'fib_0786')]:
        levels[name] = L + rng * ratio
    return levels


def _compute_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple rolling support/resistance levels using min/max windows.
    Produces sr_support_20, sr_resistance_20, sr_support_50, sr_resistance_50.
    """
    sr = pd.DataFrame(index=df.index)
    for w in [20, 50]:
        sr[f'sr_support_{w}'] = df['norm_low'].rolling(w).min()
        sr[f'sr_resistance_{w}'] = df['norm_high'].rolling(w).max()
    return sr


def _compute_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Basic candlestick pattern recognition on normalized prices.
    Adds bullish_engulfing, bearish_engulfing, hammer, shooting_star (boolean ints).
    """
    p = pd.DataFrame(index=df.index)
    o, h, l, c = df['norm_open'], df['norm_high'], df['norm_low'], df['norm_close']
    prev_o, prev_c = o.shift(1), c.shift(1)
    body = (c - o).abs()
    range_ = h - l
    upper_wick = h - c.where(c >= o, o)
    lower_wick = o.where(c >= o, c) - l
    # Engulfing
    p['bullish_engulfing'] = ((prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)).astype(int)
    p['bearish_engulfing'] = ((prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)).astype(int)
    # Hammer: small body, long lower wick
    p['hammer'] = ((body <= 0.3*range_) & (lower_wick >= 0.6*range_) & (upper_wick <= 0.2*range_)).astype(int)
    # Shooting star: small body, long upper wick
    p['shooting_star'] = ((body <= 0.3*range_) & (upper_wick >= 0.6*range_) & (lower_wick <= 0.2*range_)).astype(int)
    return p


def add_custom_indicators(df):
    """
    Add custom technical indicators to the dataframe
    Uses normalized price data (norm_open, norm_high, norm_low, norm_close)
    """
    # Make a copy to avoid modifying the original
    df_indicators = df.copy()
    
    # Ensure we have the required columns
    required_cols = ['norm_open', 'norm_high', 'norm_low', 'norm_close', 'volume']
    missing_cols = [col for col in required_cols if col not in df_indicators.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        return df_indicators
    
    try:
        # Trend Indicators - Simple Moving Averages (requested periods)
        for w in MA_PERIODS:
            df_indicators[f'sma_{w}'] = SMAIndicator(close=df_indicators['norm_close'], window=w).sma_indicator()
        
        # Exponential Moving Averages (requested periods)
        for w in MA_PERIODS:
            df_indicators[f'ema_{w}'] = EMAIndicator(close=df_indicators['norm_close'], window=w).ema_indicator()
        
        # MACD
        macd = MACD(close=df_indicators['norm_close'])
        df_indicators['macd'] = macd.macd()
        df_indicators['macd_signal'] = macd.macd_signal()
        df_indicators['macd_diff'] = macd.macd_diff()
        
        # ADX (Average Directional Index)
        adx = ADXIndicator(high=df_indicators['norm_high'], low=df_indicators['norm_low'], close=df_indicators['norm_close'])
        df_indicators['adx'] = adx.adx()
        df_indicators['adx_pos'] = adx.adx_pos()
        df_indicators['adx_neg'] = adx.adx_neg()
        
        # Momentum Indicators
        # RSI
        df_indicators['rsi'] = RSIIndicator(close=df_indicators['norm_close']).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df_indicators['norm_high'], low=df_indicators['norm_low'], close=df_indicators['norm_close'])
        df_indicators['stoch_k'] = stoch.stoch()
        df_indicators['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df_indicators['williams_r'] = WilliamsRIndicator(high=df_indicators['norm_high'], low=df_indicators['norm_low'], close=df_indicators['norm_close']).williams_r()
        
        # Volatility Indicators
        # Bollinger Bands
        bb = BollingerBands(close=df_indicators['norm_close'])
        df_indicators['bb_upper'] = bb.bollinger_hband()
        df_indicators['bb_middle'] = bb.bollinger_mavg()
        df_indicators['bb_lower'] = bb.bollinger_lband()
        df_indicators['bb_width'] = bb.bollinger_wband()
        df_indicators['bb_percent'] = bb.bollinger_pband()
        
        # Average True Range
        df_indicators['atr'] = AverageTrueRange(high=df_indicators['norm_high'], low=df_indicators['norm_low'], close=df_indicators['norm_close']).average_true_range()
        
        # Volume Indicators
        # Volume Weighted Average Price
        df_indicators['vwap'] = VolumeWeightedAveragePrice(high=df_indicators['norm_high'], low=df_indicators['norm_low'], close=df_indicators['norm_close'], volume=df_indicators['volume']).volume_weighted_average_price()
        # On-Balance Volume
        df_indicators['obv'] = OnBalanceVolumeIndicator(close=df_indicators['norm_close'], volume=df_indicators['volume']).on_balance_volume()
        
        # Ichimoku Cloud (uses high/low only)
        ichi = IchimokuIndicator(high=df_indicators['norm_high'], low=df_indicators['norm_low'])
        df_indicators['ichimoku_conv'] = ichi.ichimoku_conversion_line()
        df_indicators['ichimoku_base'] = ichi.ichimoku_base_line()
        df_indicators['ichimoku_a'] = ichi.ichimoku_a()
        df_indicators['ichimoku_b'] = ichi.ichimoku_b()
        # Lagging span (close shifted 26)
        df_indicators['ichimoku_lagging'] = df_indicators['norm_close'].shift(26)
        
        # Support/Resistance (rolling)
        sr = _compute_support_resistance(df_indicators)
        df_indicators = pd.concat([df_indicators, sr], axis=1)
        
        # Pivot Points (daily)
        piv = _compute_pivot_points(df_indicators)
        df_indicators = pd.concat([df_indicators, piv], axis=1)
        
        # Fibonacci levels (rolling window)
        fib = _compute_fibonacci_levels(df_indicators, window=100)
        df_indicators = pd.concat([df_indicators, fib], axis=1)
        
        # Custom Indicators
        # Price position relative to moving averages (selected periods)
        for w in [50, 200]:
            ma_col = f'sma_{w}'
            if ma_col in df_indicators:
                df_indicators[f'price_vs_sma_{w}'] = (df_indicators['norm_close'] - df_indicators[ma_col]) / df_indicators[ma_col] * 100
        for w in [50, 200]:
            ma_col = f'ema_{w}'
            if ma_col in df_indicators:
                df_indicators[f'price_vs_ema_{w}'] = (df_indicators['norm_close'] - df_indicators[ma_col]) / df_indicators[ma_col] * 100
        
        # Volatility ratio
        df_indicators['volatility_ratio'] = df_indicators['atr'] / df_indicators['norm_close'] * 100
        
        # Pattern recognition
        patterns = _compute_patterns(df_indicators)
        df_indicators = pd.concat([df_indicators, patterns], axis=1)
        
        print(f"  ✓ Added {len(df_indicators.columns) - len(df.columns)} technical indicators")
        
    except Exception as e:
        print(f"  ✗ Error adding indicators: {e}")
    
    return df_indicators


def add_all_ta_indicators(df):
    """
    Add all available TA indicators to the dataframe
    """
    try:
        # Rename columns to match TA library expectations
        df_ta = df.copy()
        df_ta = df_ta.rename(columns={
            'norm_open': 'open',
            'norm_high': 'high', 
            'norm_low': 'low',
            'norm_close': 'close'
        })
        
        # Add all TA indicators
        df_ta = add_all_ta_features(
            df_ta, 
            open="open", 
            high="high", 
            low="low", 
            close="close", 
            volume="volume", 
            fillna=True
        )
        
        # Rename back to original column names
        df_ta = df_ta.rename(columns={
            'open': 'norm_open',
            'high': 'norm_high',
            'low': 'norm_low', 
            'close': 'norm_close'
        })
        
        print(f"  ✓ Added all TA indicators ({len(df_ta.columns) - len(df.columns)} new columns)")
        return df_ta
        
    except Exception as e:
        print(f"  ✗ Error adding all TA indicators: {e}")
        return df

def process_file(file_path, output_dir):
    """
    Process a single file and add technical indicators
    """
    try:
        # Load the data
        print(f"Processing: {os.path.basename(file_path)}")
        df = pd.read_pickle(file_path)
        
        if df.empty:
            print(f"  ⚠️  Empty dataset, skipping")
            return
        
        print(f"  📊 Original shape: {df.shape}")
        
        # Clean NaN values
        df_clean = dropna(df)
        if len(df_clean) < 50:  # Need minimum data for indicators
            print(f"  ⚠️  Insufficient data after cleaning ({len(df_clean)} rows), skipping")
            return
        
        print(f"  🧹 Cleaned shape: {df_clean.shape}")
        
        # Add custom indicators
        df_with_indicators = add_custom_indicators(df_clean)
        
        # Add all TA indicators (optional - can be commented out if too many indicators)
        # df_with_indicators = add_all_ta_indicators(df_with_indicators)
        
        # Remove rows with NaN values from indicators
        df_final = df_with_indicators.dropna()
        
        if df_final.empty:
            print(f"  ⚠️  No data remaining after adding indicators")
            return
        
        print(f"  📈 Final shape: {df_final.shape}")
        
        # Create output filename
        filename = os.path.basename(file_path)
        output_filename = filename.replace('_normalized.pkl', '_with_indicators.pkl')
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the enhanced data
        df_final.to_pickle(output_path)
        
        # Save a sample CSV for inspection
        sample_filename = output_filename.replace('.pkl', '_sample.csv')
        sample_path = os.path.join(output_dir, sample_filename)
        df_final.head(100).to_csv(sample_path)
        
        # Calculate file size
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        
        print(f"  💾 Saved: {output_filename} ({file_size_mb:.2f} MB)")
        print(f"  📄 Sample: {sample_filename}")
        print(f"  📊 Indicators added: {len(df_final.columns) - len(df.columns)}")
        print()
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        print()

def main():
    """
    Main function to process all historical data files
    """
    print("🚀 Starting Technical Indicators Addition")
    print("=" * 50)
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory '{input_dir}' not found!")
        print("Please run the data collection script first (part1/get_historical_data.py)")
        return
    
    # Process each currency pair
    for pair in currency_pairs:
        print(f"\n📈 Processing {pair}:")
        print("-" * 30)
        
        pair_input_dir = os.path.join(input_dir, pair)
        pair_output_dir = os.path.join(output_dir, pair)
        
        # Create pair-specific output directory
        os.makedirs(pair_output_dir, exist_ok=True)
        
        if not os.path.exists(pair_input_dir):
            print(f"  ⚠️  No data found for {pair}, skipping")
            continue
        
        # Process each timeframe
        for timeframe in timeframes:
            # Look for normalized data files
            pattern = os.path.join(pair_input_dir, f"{pair}_{timeframe}_normalized.pkl")
            files = glob.glob(pattern)
            
            for file_path in files:
                process_file(file_path, pair_output_dir)
    
    print("\n✅ Technical Indicators Addition Complete!")
    print(f"📁 Output directory: {output_dir}")
    print("\n📊 Summary of added indicators:")
    print("  • Trend: SMA, EMA, MACD, ADX")
    print("  • Momentum: RSI, Stochastic, Williams %R")
    print("  • Volatility: Bollinger Bands, ATR")
    print("  • Volume: VWAP")
    print("  • Custom: Price vs MA ratios, BB position, RSI zones")

if __name__ == "__main__":
    main()
