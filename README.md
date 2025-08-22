# Forex AI Trader

*A comprehensive algorithmic trading system built with Python and OANDA API*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-orange.svg)](https://docs.astral.sh/uv/)

## 🚀 Overview

A complete algorithmic trading system designed for forex markets, featuring robust data collection, advanced feature engineering, machine learning models, and reinforcement learning agents. This project provides a production-ready foundation for building profitable trading algorithms.

The system is built with scalability and reliability in mind, incorporating best practices for data handling, model development, and live trading deployment.

## 🎯 Why Algorithmic Trading?

Algorithmic trading provides several key advantages over manual trading:

- **Emotion-free decision making**: Eliminates human bias and emotional trading
- **24/7 market monitoring**: Continuous market analysis and execution
- **Backtesting capabilities**: Test strategies on years of historical data
- **Consistent execution**: Every trade follows predefined criteria
- **Scalability**: Handle multiple instruments and strategies simultaneously
- **Risk management**: Automated position sizing and stop-loss execution

## 📚 Project Structure

This project is organized into parts, each building upon the previous:

```
forex-ai-trader/
├── part1/                    # Historical Data Collection
│   ├── get_historical_data.py
│   ├── README.md
│   └── pyproject.toml
├── part2.1/                  # Technical Indicators & Dashboard (Streamlit)
│   ├── add_technical_indicators.py
│   ├── dashboard.py
│   ├── README.md
│   └── pyproject.toml
├── part2.2/                  # Strategy Backtesting Engine
│   ├── backtest_engine.py
│   ├── dashboard.py
│   ├── README.md
│   └── pyproject.toml
├── part2.3/                  # Trading Simulator (Coming Soon)
└── README.md                 # This file
```

## 🛠️ Quick Start

### Prerequisites

- Python 3.8 or higher
- OANDA account (free demo account works)
- Git

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Viajante80/forex-ai-trader
   cd forex-ai-trader
   ```

2. **Install uv** (the fastest Python package manager):
   
   **On macOS:**
   ```bash
   brew install astral-sh/uv/uv
   ```
   
   **On Linux/macOS (universal installer):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   
   **On Windows (PowerShell):**
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.sh | iex"
   ```

3. **Set up OANDA credentials**:
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env with your actual credentials
   code .env  # or use your preferred editor
   ```

   Your `.env` file should look like this:
   ```
   # Oanda API credentials
   OANDA_API_KEY=your_actual_api_key_here
   OANDA_ACCOUNT_ID=your_actual_account_id_here
   ```

4. **Run Part 1 - Data Collection**:
   ```bash
   cd part1
   uv run get_historical_data.py
   ```

## 📊 Part 1: Historical Data Collection

The foundation of any successful trading algorithm is quality data. Part 1 focuses on building a robust data collection system using the OANDA API.

### Key Features

- **Multi-timeframe support**: M5, M15, M30, H1, H4, D, W
- **Advanced data normalization**: Prices normalized to pips (uses dynamic `pipLocation`; JPY handled automatically)
- **Production-ready**: Error handling, rate limiting, and comprehensive logging
- **Feature engineering**: Volatility, returns, and candlestick analysis

### Running Part 1

```bash
cd part1
uv run get_historical_data.py
```

Outputs:
- Downloaded historical data per pair/timeframe
- Prices normalized to pips for cross-pair comparability
- Saved as pickle with sample CSVs

📖 **[Read the detailed Part 1 documentation →](part1/README.md)**

## 🔧 Technical Foundation

The system is built on several core principles that ensure reliability and scalability:

### Core Principles

1. **Quality Data is Non-Negotiable**: Garbage in, garbage out applies doubly to trading
2. **Normalization Matters**: Raw prices are useless for cross-instrument analysis
3. **Build for Production**: Even experimental code should be robust and maintainable
4. **Start Simple**: Master data collection before moving to complex strategies



## 📈 Part 2.1: Technical Indicators & Dashboard

Turn normalized historical data into ML‑ready features and visualize them interactively.

### Key Features
- Indicators: SMA/EMA (50/80/100/200), MACD, RSI, Stochastic, Bollinger Bands, ATR, ADX, Ichimoku
- Levels: Support/Resistance, Pivot Points, Fibonacci retracements
- Volume: VWAP, OBV
- Patterns: Engulfing, Hammer, Shooting Star
- Streamlit Dashboard: Candlestick (pips) with selectable overlays and separate indicator charts, date range capped at yesterday

### Run
```bash
cd part2.1
uv sync
uv run add_technical_indicators.py
uv run streamlit run dashboard.py
```

📖 **[Read the detailed Part 2.1 documentation →](part2.1/README.md)**

## 🧪 Part 2.2: Strategy Backtesting

A comprehensive backtesting engine that evaluates individual indicators and combinations (2 and 3) under a consistent rule-based execution model. This module provides fast, reproducible backtesting workflows over the normalized (pips) datasets produced in Part 1 and enriched in Part 2.1.

### Key Features

- **Multi-timeframe testing**: M5, M15, M30, H1, H4, D, W
- **Comprehensive indicator coverage**: RSI, MACD, Stochastic, Bollinger Bands, ADX, Ichimoku, MAs, VWAP, OBV
- **Consistent execution rules**: Standardized entry/exit logic across all strategies
- **Results dashboard**: Interactive Streamlit interface for strategy analysis
- **Performance metrics**: Win rate, total pips, profit factor, average pips per trade

### Entry and Risk Rules (in pips)

#### Long Trades
- **Entry**: When combined signals are bullish AND price trades above previous candle high + 1 pip
- **Stop Loss**: Previous candle low
- **Take Profit**: 2:1 reward-to-risk ratio → TP = entry + 2 × (entry − SL)

#### Short Trades  
- **Entry**: When combined signals are bearish AND price trades below previous candle low − 1 pip
- **Stop Loss**: Previous candle high
- **Take Profit**: 2:1 reward-to-risk ratio → TP = entry − 2 × (SL − entry)

#### Explicit Formulas
- `entry_long = prev_high + 1 pip`
- `sl_long = prev_low`
- `tp_long = entry_long + 2 × (entry_long − sl_long)`
- `entry_short = prev_low − 1 pip`
- `sl_short = prev_high`
- `tp_short = entry_short − 2 × (sl_short − entry_short)`

#### Example (pips)
If `prev_high = 11234` and `prev_low = 11210`:
- **Long**: `entry = 11235`, `SL = 11210`, `risk = 25` → `TP = 11235 + 2×25 = 11285`
- **Short**: `entry = 11209`, `SL = 11234`, `risk = 25` → `TP = 11209 − 2×25 = 11159`

### What It Tests

#### Single Indicator Strategies
Tests individual indicators from this comprehensive set:
`rsi, macd, macd_signal, stoch_k, stoch_d, bb_percent, bb_width, atr, adx, adx_pos, adx_neg, ichimoku_a, ichimoku_b, obv, vwap, sma_50, sma_80, sma_100, sma_200, ema_50, ema_80, ema_100, ema_200`

#### Multi-Indicator Combinations
- **2-indicator combinations**: Tests strategic pairs of indicators
- **3-indicator combinations**: Tests triple indicator strategies

#### Signal Logic Summary
- **RSI**: Long if <30; Short if >70
- **MACD**: Long if `macd > macd_signal`; Short otherwise
- **Stochastic**: Long if `%K<20 & %K>%D`; Short if `%K>80 & %K<%D`
- **Bollinger Bands**: Long if `%B<0.1`; Short if `%B>0.9`
- **ADX/DI**: Long if `+DI>-DI & ADX>20`; Short if `-DI>+DI & ADX>20`
- **Ichimoku**: Long if close above both Span A & B; Short if below both
- **MAs/VWAP**: Long if close > MA/VWAP; Short if close < MA/VWAP
- **OBV**: Slope up → Long; Slope down → Short
- **Combined signal**: Average of votes (>0 bullish, <0 bearish, 0 neutral)

### Run Backtests

```bash
cd part2.2
uv sync
uv run backtest_engine.py
```

### Results Dashboard

```bash
uv run streamlit run dashboard.py
```

### Outputs

Results are saved to `../backtest_strategies/`:
- `results_single.pkl` – Single-indicator summary
- `results_combo2.pkl` – 2-indicator combo summary  
- `results_combo3.pkl` – 3-indicator combo summary
- Per-pair trade logs: `{PAIR}/{PAIR}_{TF}_{combo}_trades.pkl`

### Dashboard Features

- Loads `.pkl` result files
- Filter by timeframe and minimum trade count
- Sort by win rate, total pips, profit factor, or average pips
- View top strategy rows and bar charts
- Summary metrics aggregation

📖 **[Read the detailed Part 2.2 documentation →](part2.2/README.md)**

## 🚧 Coming Soon

### Part 2.3: Trading Simulator
Interactive trading simulator for strategy testing and validation.

## 📈 Supported Currency Pairs

Currently configured for major forex pairs:
- `EUR_USD` (Euro/US Dollar)
- `GBP_USD` (British Pound/US Dollar)  
- `EUR_GBP` (Euro/British Pound)

*Easily extensible to include more pairs by modifying the `major_pairs` list.*

## 🛡️ Production-Ready Features

The system is designed with production deployment in mind:

- **Environment variables**: API keys safely stored in `.env` files
- **Organised data storage**: Each currency pair gets its own directory
- **Multiple output formats**: Both pickle (for Python) and CSV samples (for human analysis)
- **Comprehensive logging**: Track progress and catch issues early
- **Error handling**: Automatic retry logic and rate limiting
- **Data integrity**: Duplicate removal and chronological sorting

## 🤝 Contributing

This project is open for contributions and improvements:

- Fork the repository
- Submit issues and feature requests
- Contribute improvements through pull requests
- Share your own trading strategies and insights

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OANDA](https://www.oanda.com/) for providing the forex data API
- [uv](https://docs.astral.sh/uv/) for fast Python package management
- The open-source community for the amazing tools that made this possible

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Part 1 documentation](part1/README.md) for detailed setup instructions
2. Review the troubleshooting section in each part's README
3. Open an issue on GitHub with detailed error information

---

*Ready to start building your own algorithmic trading system? Clone the repository and begin with data collection today.*

**Stay tuned for Parts 2-5, featuring advanced feature engineering, machine learning models, and live trading deployment.** 