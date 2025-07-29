# Forex AI Trader - Part 1: Historical Data Collection

This module provides comprehensive historical forex data collection from OANDA's API, with advanced data normalization and feature engineering for machine learning and reinforcement learning applications.

## Overview

The `get_historical_data.py` script downloads historical forex data from OANDA's API and processes it with advanced normalization techniques. It's designed to create high-quality datasets for AI trading systems, with features that make the data suitable for machine learning and reinforcement learning models.

## Features

### 🔄 **Multi-Timeframe Support**
- Downloads data across multiple timeframes: M5, M15, M30, H1, H4, D, W
- Handles OANDA's API pagination limits automatically
- Respects API rate limits to prevent service interruptions

### 📊 **Advanced Data Normalization**
- **Pip-based normalization**: Converts all prices to pip values for consistent scaling
- **Instrument-specific handling**: Automatically detects pip locations for different currency pairs
- **Normalized features**: Creates both original and normalized price columns

### 🎯 **Feature Engineering**
- **Returns calculation**: Percentage changes in both original and normalized prices
- **Volatility metrics**: Rolling standard deviation of returns
- **Candlestick analysis**: Body size, upper/lower shadows (wicks) in pip values
- **Range analysis**: High-low ranges and open-close movements in pips

### 💾 **Data Storage**
- **Pickle format**: Efficient binary storage for large datasets
- **CSV samples**: Human-readable samples for data inspection
- **Organized structure**: Separate directories for each currency pair

## Supported Currency Pairs

Currently configured for major forex pairs:
- `EUR_USD` (Euro/US Dollar)
- `GBP_USD` (British Pound/US Dollar)  
- `EUR_GBP` (Euro/British Pound)

*Note: The code is easily extensible to include more pairs by modifying the `major_pairs` list.*

## Data Structure

Each downloaded dataset includes the following columns:

### Price Data
- `open`, `high`, `low`, `close`: Original price values
- `norm_open`, `norm_high`, `norm_low`, `norm_close`: Normalized pip values
- `volume`: Trading volume

### Calculated Features
- `return`: Percentage change in original prices
- `norm_return`: Percentage change in normalized prices
- `pip_range`: High-low range in pips
- `pip_move`: Open-close movement in pips
- `volatility_10`: 10-period rolling volatility
- `body_size`: Absolute body size in pips
- `upper_shadow`: Upper wick length in pips
- `lower_shadow`: Lower wick length in pips

## Setup Instructions

### 1. Environment Setup

Create a `.env` file in the project root with your OANDA credentials:

```bash
OANDA_API_KEY=your_api_key_here
OANDA_ACCOUNT_ID=your_account_id_here
```

### 2. Install Dependencies

The project uses `uv` for dependency management. Install dependencies with:

```bash
uv sync
```

### 3. OANDA Account Setup

1. Create an OANDA account at [oanda.com](https://www.oanda.com)
2. Generate an API key from your account dashboard
3. Note your account ID (found in account settings)

## Usage

### Basic Execution

```bash
cd part1
uv run get_historical_data.py
```

### Configuration Options

You can modify the following variables in the script:

```python
# Currency pairs to download
major_pairs = ["EUR_USD", "GBP_USD", "EUR_GBP"]

# Timeframes to collect
timeframes = ["M5", "M15", "M30", "H1", "H4", "D", "W"]

# Start date for historical data
start_date = datetime(2016, 1, 1)
```

## Output Structure

The script creates the following directory structure:

```
oanda_historical_data/
├── EUR_USD/
│   ├── EUR_USD_M5_normalized.pkl
│   ├── EUR_USD_M5_sample.csv
│   ├── EUR_USD_M15_normalized.pkl
│   ├── EUR_USD_M15_sample.csv
│   └── ...
├── GBP_USD/
│   └── ...
└── EUR_GBP/
    └── ...
```

### File Formats

- **`.pkl` files**: Complete datasets in pickle format for efficient loading
- **`.csv` files**: Sample data (first 100 rows) for manual inspection

## Data Quality Features

### 🛡️ **Error Handling**
- Automatic retry logic for API failures
- Graceful handling of rate limits
- Comprehensive error logging

### 🔄 **Data Integrity**
- Duplicate removal at pagination boundaries
- Proper datetime indexing
- Chronological sorting

### 📈 **Performance Optimization**
- Efficient pandas operations
- Memory-conscious data processing
- Optimized API request patterns

## API Rate Limiting

The script implements intelligent rate limiting:
- Pauses between API requests to respect OANDA's limits
- Longer delays on errors to prevent service disruption
- Batch processing to minimize API calls

## Troubleshooting

### Common Issues

1. **Missing Credentials**
   ```
   ValueError: Missing OANDA credentials. Ensure OANDA_API_KEY and OANDA_ACCOUNT_ID are in your .env file.
   ```
   **Solution**: Check your `.env` file and ensure credentials are correct.

2. **API Rate Limits**
   ```
   Error: HTTP 429 - Too Many Requests
   ```
   **Solution**: The script automatically handles this, but you can increase delays if needed.

3. **No Data Available**
   ```
   No data available for [PAIR] [TIMEFRAME]
   ```
   **Solution**: Check if the instrument is available in your OANDA account.

### Performance Tips

- **Large datasets**: The script can download years of data. Monitor disk space.
- **Memory usage**: For very large datasets, consider processing in chunks.
- **Network stability**: Ensure stable internet connection for large downloads.

## Next Steps

This data collection module is designed to feed into:

1. **Part 2**: Technical analysis and feature engineering
2. **Part 3**: Machine learning model development
3. **Part 4**: Reinforcement learning trading agent
5. **Part 5**: Backtesting and performance evaluation

## Dependencies

- `pandas>=2.3.0`: Data manipulation and analysis
- `numpy>=2.3.1`: Numerical computations
- `oandapyv20>=0.7.2`: OANDA API client
- `python-dotenv>=0.9.9`: Environment variable management
- `requests>=2.32.4`: HTTP requests

## License

This project is part of the Forex AI Trader system. See the main project README for licensing information.

---

**Note**: This script requires an active OANDA account with API access. Demo accounts are sufficient for testing, but live accounts may provide more comprehensive data access. 