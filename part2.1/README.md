# Forex AI Trader - Part 2.1: Technical Indicators

This module adds comprehensive technical indicators to the historical forex data collected in Part 1, transforming raw price data into ML-ready features using the [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/latest/).

## Overview

The `add_technical_indicators.py` script processes the normalized historical data from Part 1 and adds a comprehensive set of technical indicators. These indicators are essential for machine learning models to identify patterns, trends, and trading opportunities in the forex market.

## Features

### 📊 **Comprehensive Indicator Suite**
- **Trend Indicators**: SMA, EMA, MACD, ADX, Ichimoku Cloud
- **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R
- **Volatility Indicators**: Bollinger Bands, Average True Range (ATR)
- **Levels**: Support/Resistance (rolling), Pivot Points (daily), Fibonacci Retracements (rolling)
- **Volume Indicators**: VWAP, On-Balance Volume (OBV)
- **Patterns**: Bullish/Bearish Engulfing, Hammer, Shooting Star
- **Custom Metrics**: Price vs MA ratios, volatility ratios, BB position

### 🔧 **Advanced Processing**
- **Normalized Data Support**: Uses `norm_open`, `norm_high`, `norm_low`, `norm_close` (in pips) for consistency across pairs
- **Automatic Data Cleaning**: Removes NaN values and ensures data quality
- **Batch Processing**: Processes all currency pairs and timeframes automatically
- **Error Handling**: Robust error handling with detailed logging

### 💾 **Flexible Output**
- **Pickle Format**: Efficient binary storage for large datasets
- **CSV Samples**: Human-readable samples for data inspection
- **Organized Structure**: Maintains the same directory structure as input

## Technical Indicators Added

### Trend Indicators
- **SMA (Simple Moving Average)**: 50, 80, 100, 200 periods → `sma_50`, `sma_80`, `sma_100`, `sma_200`
- **EMA (Exponential Moving Average)**: 50, 80, 100, 200 periods → `ema_50`, `ema_80`, `ema_100`, `ema_200`
- **MACD**: `macd`, `macd_signal`, `macd_diff`
- **ADX**: `adx`, `adx_pos`, `adx_neg`
- **Ichimoku Cloud**: `ichimoku_conv`, `ichimoku_base`, `ichimoku_a`, `ichimoku_b`, `ichimoku_lagging`

### Momentum Indicators
- **RSI**: `rsi` (default 14)
- **Stochastic Oscillator**: `%K` and `%D` → `stoch_k`, `stoch_d`
- **Williams %R**: `williams_r`

### Volatility Indicators
- **Bollinger Bands**: `bb_upper`, `bb_middle`, `bb_lower`, `bb_width`, `bb_percent`
- **ATR**: `atr`

### Levels & Price Structures
- **Support/Resistance (rolling)**: `sr_support_20`, `sr_resistance_20`, `sr_support_50`, `sr_resistance_50`
- **Pivot Points (classic, daily)**: `pivot_p`, `pivot_r1`, `pivot_s1`, `pivot_r2`, `pivot_s2`
- **Fibonacci Retracements (rolling window)**: `fib_0`, `fib_0236`, `fib_0382`, `fib_0500`, `fib_0618`, `fib_0786`, `fib_1`

### Volume Indicators
- **VWAP**: `vwap`
- **On-Balance Volume**: `obv`

### Pattern Recognition
- **Candlestick Patterns**: `bullish_engulfing`, `bearish_engulfing`, `hammer`, `shooting_star`

## Setup Instructions

### 1. Prerequisites
Ensure you have completed Part 1 and have the historical data available:
```bash
# Run Part 1 first if you haven't already
cd part1
uv run get_historical_data.py
```

### 2. Install Dependencies
```bash
cd part2.1
uv sync
```

### 3. Run the Script
```bash
uv run add_technical_indicators.py
```

### 4. Launch the Dashboard (Optional)
```bash
uv run streamlit run dashboard.py
```

## Configuration

### Currency Pairs
Modify the `currency_pairs` list in the script:
```python
currency_pairs = ["EUR_USD", "GBP_USD", "EUR_GBP"]
```

### Timeframes
Modify the `timeframes` list in the script:
```python
timeframes = ["M5", "M15", "M30", "H1", "H4", "D", "W"]
```

### Input/Output Directories
```python
input_dir = "../oanda_historical_data"    # From Part 1 (relative to part2.1)
output_dir = "../trading_ready_data"      # New output directory (relative to part2.1)
```

## Output Structure

The script creates the following directory structure:

```
trading_ready_data/
├── EUR_USD/
│   ├── EUR_USD_M5_with_indicators.pkl
│   ├── EUR_USD_M5_with_indicators_sample.csv
│   ├── EUR_USD_M15_with_indicators.pkl
│   ├── EUR_USD_M15_with_indicators_sample.csv
│   └── ...
├── GBP_USD/
│   └── ...
└── EUR_GBP/
    └── ...
```

### File Formats
- **`.pkl` files**: Complete datasets with all indicators in pickle format
- **`.csv` files**: Sample data (first 100 rows) for manual inspection

## Normalized vs Original Prices

- All price-derived indicators are computed using normalized prices in pips (`norm_*`). This ensures cross-pair comparability (e.g., EURUSD vs USDJPY) and makes magnitudes directly interpretable in pips.
- Volume-based metrics (e.g., OBV) use the original volume column.
- The dashboard plots prices in pips and overlays indicators on the same scale.

## Data Quality Features

### 🛡️ **Error Handling**
- Automatic detection of missing columns
- Graceful handling of insufficient data
- Detailed error logging for troubleshooting

### 🔄 **Data Integrity**
- Automatic NaN removal and data cleaning
- Minimum data requirements (50 rows) for indicator calculation
- Preservation of original data structure

### 📈 **Performance Optimization**
- Efficient pandas operations
- Memory-conscious processing
- Batch processing for multiple files

## Advanced Usage

### Adding All TA Indicators
The script includes an optional function to add all available TA indicators:

```python
# Uncomment this line in the process_file function
df_with_indicators = add_all_ta_indicators(df_with_indicators)
```

**Note**: This adds 100+ additional indicators and significantly increases file size.

### Custom Indicator Development
You can easily add custom indicators by modifying the `add_custom_indicators` function:

```python
# Example: Add a custom momentum indicator
df_indicators['custom_momentum'] = (
    df_indicators['norm_close'] - df_indicators['norm_close'].shift(5)
) / df_indicators['norm_close'].shift(5) * 100
```

## Troubleshooting

### Common Issues

1. **Missing Input Data**
   ```
   ❌ Input directory 'oanda_historical_data' not found!
   ```
   **Solution**: Run Part 1 data collection first.

2. **Insufficient Data**
   ```
   ⚠️  Insufficient data after cleaning (X rows), skipping
   ```
   **Solution**: Ensure you have at least 50 rows of clean data.

3. **Missing Columns**
   ```
   Warning: Missing columns: ['norm_open', 'norm_high']
   ```
   **Solution**: Check that Part 1 generated normalized data correctly.

### Performance Tips

- **Large datasets**: Monitor memory usage for very large files
- **Processing time**: Higher timeframes (D, W) process faster than lower ones (M5, M15)
- **Storage space**: Each file with indicators is larger than the original

## Integration with ML Models

The enhanced datasets are now ready for machine learning:

```python
import pandas as pd

# Load the enhanced data
df = pd.read_pickle('trading_ready_data/EUR_USD/EUR_USD_H1_with_indicators.pkl')

# Example feature selection (indicators only)
feature_columns = [
    'sma_50','sma_200','ema_50','ema_200','rsi','macd','macd_signal','bb_width',
    'stoch_k','stoch_d','atr','adx','adx_pos','adx_neg','ichimoku_a','ichimoku_b',
    'sr_support_20','sr_resistance_20','pivot_p','fib_0618','obv','vwap'
]
X = df[feature_columns].dropna()
```

## Interactive Dashboard

The module includes a comprehensive Streamlit dashboard for data visualization:

### Features
- **Interactive Candlestick Charts (pips)**: Price data with volume
- **Overlay Indicators (selectable)**: SMA/EMA (periods 50/80/100/200), Bollinger Bands, Ichimoku, Support/Resistance, Fibonacci, Pivot Points, Patterns
- **Separate Indicator Charts (selectable)**: RSI, MACD, Stochastic, ATR, ADX, OBV, VWAP
- **Date Range Selection**: Capped at yesterday (UTC)
- **Data Summary Metrics** and **Interactive Table**
- **Download Options**: Export filtered data as CSV

### Usage
```bash
cd part2.1
uv run streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Dependencies

- `pandas>=2.3.0`: Data manipulation and analysis
- `numpy>=2.3.1`: Numerical computations
- `ta>=0.10.2`: Technical analysis library
- `streamlit>=1.28.0`: Interactive web dashboard
- `plotly>=5.17.0`: Interactive charts and visualizations
- `python-dotenv>=0.9.9`: Environment variable management

---

**Note**: The technical indicators are calculated using the [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/latest/), supplemented with custom implementations for levels and patterns. Normalized prices in pips ensure consistent interpretation across currency pairs. 