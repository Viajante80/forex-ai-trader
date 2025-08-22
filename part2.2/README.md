# Forex AI Trader - Part 2.2: Strategy Backtesting

This module provides a fast, reproducible backtesting workflow over the normalized (pips) datasets produced in Part 1 and enriched in Part 2.1. It evaluates individual indicators and combinations (2 and 3) under a consistent rule-based execution model and includes a results dashboard.

## Overview

- Data source: `../trading_ready_data/{PAIR}/{PAIR}_{TF}_with_indicators.pkl`
- Execution rules (pips):
  - Long entry: when the combined signal is bullish and the current bar trades above the previous bar high + 1 pip.
    - Stop loss: previous bar low
    - Take profit: RR = 2.0 × risk (2:1 reward-to-risk)
  - Short entry: when the combined signal is bearish and the current bar trades below the previous bar low − 1 pip.
    - Stop loss: previous bar high
    - Take profit: RR = 2.0 × risk
- Indicators are computed on normalized prices (pips); volume metrics use raw volume.

## What’s Included

- `backtest_engine.py`: core backtest runner
- `dashboard.py`: Streamlit dashboard for results analysis
- `pyproject.toml`: dependencies for `uv`

## Tested Signals

- Single-indicator tests from this set (if column exists):
  `rsi, macd, macd_signal, stoch_k, stoch_d, bb_percent, bb_width, atr, adx, adx_pos, adx_neg, ichimoku_a, ichimoku_b, obv, vwap, sma_50, sma_80, sma_100, sma_200, ema_50, ema_80, ema_100, ema_200`
- 2-indicator combinations (subset for speed)
- 3-indicator combinations (subset for speed)

Signal logic (summary):
- RSI: long if <30; short if >70
- MACD: long if `macd > macd_signal`; short otherwise
- Stochastic: long if `%K<20 & %K>%D`; short if `%K>80 & %K<%D`
- BB%: long if <0.1; short if >0.9
- ADX/DI: long if `+DI>-DI & ADX>20`; short if `-DI>+DI & ADX>20`
- Ichimoku: long if close above both Span A & B; short if below both
- MAs/VWAP: long if close > MA/VWAP; short if close < MA/VWAP
- OBV: slope up → long; slope down → short
- Combined signal = average of votes (>0 bullish, <0 bearish, 0 neutral)

## Setup

```bash
cd part2.2
uv sync
```

## Run Backtests

```bash
uv run backtest_engine.py
```

Defaults:
- Pairs: `EUR_USD`
- Timeframes: `M5, M15, M30, H1, H4, D, W`
- Entry buffer: `ENTRY_BUFFER_PIPS = 1.0`
- Take profit ratio: `RR = 2.0`

Outputs (created in `../backtest_strategies/`):
- `results_single.pkl` – single-indicator summary (PKL)
- `results_combo2.pkl` – 2-indicator combo summary (PKL)
- `results_combo3.pkl` – 3-indicator combo summary (PKL)
- Per-pair trade logs: `{PAIR}/{PAIR}_{TF}_{combo}_trades.pkl`

## Trade Execution Logic (pips)

### Entry triggers
- Compute previous candle high/low: `prev_high`, `prev_low`
- Long entry when BOTH conditions are true on the current bar:
  - Combined indicator signal > 0 (bullish)
  - High of current bar crosses `prev_high + 1 pip`
- Short entry when BOTH conditions are true:
  - Combined indicator signal < 0 (bearish)
  - Low of current bar crosses `prev_low − 1 pip`

### Stop loss and take profit
- Long trade:
  - `SL = prev_low`
  - `risk = entry − SL`
  - `TP = entry + RR × risk` (with `RR = 2.0` by default)
- Short trade:
  - `SL = prev_high`
  - `risk = SL − entry`
  - `TP = entry − RR × risk`

### Explicit formulas
- `entry_long = prev_high + 1`
- `sl_long = prev_low`
- `tp_long = entry_long + 2 × (entry_long − sl_long)`
- `entry_short = prev_low − 1`
- `sl_short = prev_high`
- `tp_short = entry_short − 2 × (sl_short − entry_short)`

### Example
If `prev_high = 11234` and `prev_low = 11210` (all in pips):
- Long: `entry = 11235`, `SL = 11210` → `risk = 25` → `TP = 11235 + 2×25 = 11285`
- Short: `entry = 11209`, `SL = 11234` → `risk = 25` → `TP = 11209 − 2×25 = 11159`

### Intrabar evaluation order
- For longs we check SL first, then TP within the same bar (`low <= SL` before `high >= TP`).
- For shorts we check SL first, then TP (`high >= SL` before `low <= TP`).
- If neither TP nor SL is hit intrabar, the position is closed at the bar’s close.
- This conservative ordering slightly biases toward hitting the stop first when both extremes are touched on the same bar.

## Results Dashboard

```bash
uv run streamlit run dashboard.py
```

Dashboard features:
- Loads `.pkl` result files
- Filter by timeframe and minimum trade count
- Sort by win rate, total pips, profit factor, or average pips
- View top strategy rows and bar charts
- Summary metrics aggregation

## Customization

Open `backtest_engine.py` and adjust:
- Data scope:
  - `pairs = ["EUR_USD"]`
  - `timeframes = ["M5","M15","M30","H1","H4","D","W"]`
- Risk/entry rules:
  - `ENTRY_BUFFER_PIPS` (default 1.0)
  - `RR` (default 2.0)
- Indicator universe:
  - `SINGLE_INDICATORS` list (add/remove to expand or narrow tests)
- Search breadth:
  - `max_combos` in `run_initial_tests` for 1/2/3-indicator scans
- Signal semantics:
  - Edit `_indicator_signal` to change voting/thresholds

## Performance Tips

- Start with shorter timeframes or fewer combos via `max_combos`.
- Ensure `trading_ready_data` contains the necessary indicator columns.
- Run backtests per timeframe if memory is limited.

## Notes

- All prices are in pips via normalized fields `norm_*`.
- Intrabar TP/SL checks assume high/low containment for the bar; full tick-level simulation is out of scope here.
- This module is intended for relative strategy comparison; forward-testing/walk-forward validation is recommended before live use.

## Logging

- The backtest engine writes a `backtest.log` file into `../backtest_strategies/` with entries like:
  - `Testing pair=EUR_USD timeframe=H1 indicators=rsi+macd`.
  - This helps track what timeframe and indicator combinations were tested.