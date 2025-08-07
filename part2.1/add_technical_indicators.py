import pandas as pd
import os
import glob
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice
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
        # Trend Indicators
        # Simple Moving Averages
        df_indicators['sma_10'] = SMAIndicator(close=df_indicators['norm_close'], window=10).sma_indicator()
        df_indicators['sma_20'] = SMAIndicator(close=df_indicators['norm_close'], window=20).sma_indicator()
        df_indicators['sma_50'] = SMAIndicator(close=df_indicators['norm_close'], window=50).sma_indicator()
        df_indicators['sma_200'] = SMAIndicator(close=df_indicators['norm_close'], window=200).sma_indicator()
        
        # Exponential Moving Averages
        df_indicators['ema_10'] = EMAIndicator(close=df_indicators['norm_close'], window=10).ema_indicator()
        df_indicators['ema_20'] = EMAIndicator(close=df_indicators['norm_close'], window=20).ema_indicator()
        df_indicators['ema_50'] = EMAIndicator(close=df_indicators['norm_close'], window=50).ema_indicator()
        
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
        
        # Custom Indicators
        # Price position relative to moving averages
        df_indicators['price_vs_sma_20'] = (df_indicators['norm_close'] - df_indicators['sma_20']) / df_indicators['sma_20'] * 100
        df_indicators['price_vs_sma_50'] = (df_indicators['norm_close'] - df_indicators['sma_50']) / df_indicators['sma_50'] * 100
        df_indicators['price_vs_ema_20'] = (df_indicators['norm_close'] - df_indicators['ema_20']) / df_indicators['ema_20'] * 100
        
        # Moving average crossovers
        df_indicators['sma_10_vs_sma_20'] = (df_indicators['sma_10'] - df_indicators['sma_20']) / df_indicators['sma_20'] * 100
        df_indicators['ema_10_vs_ema_20'] = (df_indicators['ema_10'] - df_indicators['ema_20']) / df_indicators['ema_20'] * 100
        
        # Bollinger Band position
        df_indicators['bb_position'] = (df_indicators['norm_close'] - df_indicators['bb_lower']) / (df_indicators['bb_upper'] - df_indicators['bb_lower'])
        
        # RSI zones
        df_indicators['rsi_oversold'] = (df_indicators['rsi'] < 30).astype(int)
        df_indicators['rsi_overbought'] = (df_indicators['rsi'] > 70).astype(int)
        
        # Volatility ratio
        df_indicators['volatility_ratio'] = df_indicators['atr'] / df_indicators['norm_close'] * 100
        
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
