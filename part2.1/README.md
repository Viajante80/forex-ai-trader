# Forex AI Trader - Part 2.1: Technical Indicators

This module adds comprehensive technical indicators to the historical forex data collected in Part 1, transforming raw price data into ML-ready features using the [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/latest/).

## Overview

The `add_technical_indicators.py` script processes the normalized historical data from Part 1 and adds a comprehensive set of technical indicators. These indicators are essential for machine learning models to identify patterns, trends, and trading opportunities in the forex market.

## Features

### 📊 **Comprehensive Indicator Suite**
- **Trend Indicators**: SMA, EMA, MACD, ADX
- **Momentum Indicators**: RSI, Stochastic Oscillator, Williams %R
- **Volatility Indicators**: Bollinger Bands, Average True Range (ATR)
- **Volume Indicators**: Volume Weighted Average Price (VWAP)
- **Custom Indicators**: Price vs MA ratios, BB position, RSI zones

### 🔧 **Advanced Processing**
- **Normalized Data Support**: Uses `norm_open`, `norm_high`, `norm_low`, `norm_close` for consistent scaling
- **Automatic Data Cleaning**: Removes NaN values and ensures data quality
- **Batch Processing**: Processes all currency pairs and timeframes automatically
- **Error Handling**: Robust error handling with detailed logging

### 💾 **Flexible Output**
- **Pickle Format**: Efficient binary storage for large datasets
- **CSV Samples**: Human-readable samples for data inspection
- **Organized Structure**: Maintains the same directory structure as input

## Technical Indicators Added

### Trend Indicators
- **SMA (Simple Moving Average)**: 10, 20, 50, 200 periods
- **EMA (Exponential Moving Average)**: 10, 20, 50 periods
- **MACD**: MACD line, signal line, and histogram
- **ADX**: Average Directional Index with positive/negative components

### Momentum Indicators
- **RSI**: Relative Strength Index (14 periods)
- **Stochastic Oscillator**: %K and %D lines
- **Williams %R**: Williams Percent Range

### Volatility Indicators
- **Bollinger Bands**: Upper, middle, lower bands with width and percent
- **ATR**: Average True Range for volatility measurement

### Volume Indicators
- **VWAP**: Volume Weighted Average Price

### Custom Indicators
- **Price vs MA Ratios**: Percentage difference from moving averages
- **MA Crossovers**: Relationships between different moving averages
- **BB Position**: Price position within Bollinger Bands (0-1 scale)
- **RSI Zones**: Binary indicators for oversold (<30) and overbought (>70)
- **Volatility Ratio**: ATR as percentage of price

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

# Features for ML (exclude original price columns if desired)
feature_columns = [col for col in df.columns if col.startswith(('sma_', 'ema_', 'rsi', 'bb_'))]
X = df[feature_columns].dropna()
```

## Next Steps

This enhanced dataset is ready for:
1. **Part 2.2**: Advanced feature engineering and pattern recognition
2. **Part 3**: Machine learning model development
3. **Part 4**: Reinforcement learning agent training

## Interactive Dashboard

The module includes a comprehensive Streamlit dashboard for data visualization:

### Features
- **Interactive Candlestick Charts**: Price data with volume
- **Technical Indicators Overlay**: Select which indicators to display on price charts
- **Separate Indicator Charts**: RSI, Stochastic, ATR, ADX in dedicated subplots
- **Date Range Selection**: Filter data by specific time periods
- **Data Summary Metrics**: Key statistics and insights
- **Interactive Data Table**: Browse and filter data
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

## License

This project is part of the Forex AI Trader system. See the main project README for licensing information.

---

**Note**: The technical indicators are calculated using the [Technical Analysis Library in Python](https://technical-analysis-library-in-python.readthedocs.io/en/latest/), which provides a comprehensive suite of financial indicators for time series analysis. 