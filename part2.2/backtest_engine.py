import os
import glob
import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Paths
DATA_DIR = "../trading_ready_data"
OUTPUT_DIR = "../backtest_strategies"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logging setup (console + file)
logger = logging.getLogger("backtest_engine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "backtest.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

# Trading rule parameters
ENTRY_BUFFER_PIPS = 1.0  # 1 pip above/below previous high/low
RR = 2.0  # take profit reward:risk ratio

# Indicator sets to test (columns expected in trading_ready_data)
SINGLE_INDICATORS = [
    'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'bb_percent', 'bb_width',
    'atr', 'adx', 'adx_pos', 'adx_neg', 'ichimoku_a', 'ichimoku_b', 'obv', 'vwap',
    'sma_50', 'sma_80', 'sma_100', 'sma_200', 'ema_50', 'ema_80', 'ema_100', 'ema_200'
]

# Build pairs of indicators (2-combo) and triples (3-combo) for tests

def _combinations(lst: List[str], k: int) -> List[Tuple[str, ...]]:
    from itertools import combinations
    return list(combinations(lst, k))

COMBOS_2 = _combinations(SINGLE_INDICATORS, 2)
COMBOS_3 = _combinations(SINGLE_INDICATORS, 3)

@dataclass
class Trade:
    direction: str  # 'long' or 'short'
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    result_pips: Optional[float] = None


def _indicator_signal(df: pd.DataFrame, indicators: Tuple[str, ...]) -> pd.Series:
    """Generate a basic combined signal from indicators.
    Rules (simple defaults):
    - RSI: long if rsi < 30, short if rsi > 70
    - MACD: long if macd > macd_signal, short otherwise
    - Stochastic: long if k<20 & k>d, short if k>80 & k<d
    - BB%: long if bb_percent < 0.1, short if > 0.9
    - ADX/DI: long if adx_pos > adx_neg and adx>20, short if adx_neg>adx_pos and adx>20
    - Ichimoku: long if close > ichimoku_a and > ichimoku_b; short if < both
    - MAs: long if close > MA, short if close < MA
    - VWAP: long if close > vwap, short if close < vwap
    - OBV: slope up → long; slope down → short
    Final signal: mean of component votes (>0 => long, <0 => short, 0 neutral)
    """
    votes = pd.Series(0.0, index=df.index)
    close = df['norm_close']

    for ind in indicators:
        if ind == 'rsi' and 'rsi' in df:
            v = np.where(df['rsi'] < 30, 1, np.where(df['rsi'] > 70, -1, 0))
            votes += v
        elif ind == 'macd' and {'macd','macd_signal'}.issubset(df.columns):
            v = np.where(df['macd'] > df['macd_signal'], 1, -1)
            votes += v
        elif ind == 'stoch_k' and {'stoch_k','stoch_d'}.issubset(df.columns):
            v = np.where((df['stoch_k'] < 20) & (df['stoch_k'] > df['stoch_d']), 1,
                         np.where((df['stoch_k'] > 80) & (df['stoch_k'] < df['stoch_d']), -1, 0))
            votes += v
        elif ind == 'bb_percent' and 'bb_percent' in df:
            v = np.where(df['bb_percent'] < 0.1, 1, np.where(df['bb_percent'] > 0.9, -1, 0))
            votes += v
        elif ind in {'adx','adx_pos','adx_neg'} and {'adx','adx_pos','adx_neg'}.issubset(df.columns):
            long = (df['adx_pos'] > df['adx_neg']) & (df['adx'] > 20)
            short = (df['adx_neg'] > df['adx_pos']) & (df['adx'] > 20)
            v = np.where(long, 1, np.where(short, -1, 0))
            votes += v
        elif ind in {'ichimoku_a','ichimoku_b'} and {'ichimoku_a','ichimoku_b'}.issubset(df.columns):
            long = (close > df['ichimoku_a']) & (close > df['ichimoku_b'])
            short = (close < df['ichimoku_a']) & (close < df['ichimoku_b'])
            v = np.where(long, 1, np.where(short, -1, 0))
            votes += v
        elif ind.startswith('sma_') and ind in df:
            v = np.where(close > df[ind], 1, np.where(close < df[ind], -1, 0))
            votes += v
        elif ind.startswith('ema_') and ind in df:
            v = np.where(close > df[ind], 1, np.where(close < df[ind], -1, 0))
            votes += v
        elif ind == 'vwap' and 'vwap' in df:
            v = np.where(close > df['vwap'], 1, np.where(close < df['vwap'], -1, 0))
            votes += v
        elif ind == 'obv' and 'obv' in df:
            obv_slope = df['obv'].diff()
            v = np.where(obv_slope > 0, 1, np.where(obv_slope < 0, -1, 0))
            votes += v
        # Others default to 0

    # Normalize by number of indicators used
    votes = votes / max(len(indicators), 1)
    return votes


def _simulate_trades(df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
    """Simulate trades using entry/exit rules described by the user.
    - Entry long: when signal>0 and price crosses prev_high + 1 pip
    - Entry short: when signal<0 and price crosses prev_low - 1 pip
    - SL: previous candle low (long) or high (short)
    - TP: RR * risk
    """
    prev_high = df['norm_high'].shift(1)
    prev_low = df['norm_low'].shift(1)

    long_trigger = prev_high + ENTRY_BUFFER_PIPS
    short_trigger = prev_low - ENTRY_BUFFER_PIPS

    results = []

    for ts, row in df.iterrows():
        high = row['norm_high']
        low = row['norm_low']
        close = row['norm_close']
        sig = signals.loc[ts]
        ph = prev_high.loc[ts]
        pl = prev_low.loc[ts]

        # Long entry
        if sig > 0 and not np.isnan(ph) and high >= long_trigger.loc[ts]:
            entry = long_trigger.loc[ts]
            sl = pl  # previous low
            risk = entry - sl
            tp = entry + RR * risk
            outcome_price = None
            # Check intrabar TP/SL hit (SL first, then TP)
            if low <= sl:
                outcome_price = sl
            if high >= tp and (outcome_price is None):
                outcome_price = tp
            if outcome_price is None:
                outcome_price = close
            result_pips = outcome_price - entry
            results.append((ts, 'long', entry, sl, tp, outcome_price, result_pips))
            continue

        # Short entry
        if sig < 0 and not np.isnan(pl) and low <= short_trigger.loc[ts]:
            entry = short_trigger.loc[ts]
            sl = ph  # previous high
            risk = sl - entry
            tp = entry - RR * risk
            outcome_price = None
            if high >= sl:
                outcome_price = sl
            if low <= tp and (outcome_price is None):
                outcome_price = tp
            if outcome_price is None:
                outcome_price = close
            result_pips = entry - outcome_price
            results.append((ts, 'short', entry, sl, tp, outcome_price, result_pips))

    trades = pd.DataFrame(results, columns=['time','direction','entry','sl','tp','exit','result_pips'])
    return trades


def _evaluate(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'avg_pips': 0.0,
            'total_pips': 0.0,
            'profit_factor': 0.0
        }
    wins = trades['result_pips'] > 0
    losses = trades['result_pips'] < 0
    total_pips = trades['result_pips'].sum()
    avg_pips = trades['result_pips'].mean()
    win_rate = wins.mean() * 100
    gross_win = trades.loc[wins, 'result_pips'].sum()
    gross_loss = -trades.loc[losses, 'result_pips'].sum()
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else np.inf
    return {
        'num_trades': int(len(trades)),
        'win_rate': float(win_rate),
        'avg_pips': float(avg_pips),
        'total_pips': float(total_pips),
        'profit_factor': float(profit_factor)
    }


def backtest_pair_timeframe(pair: str, timeframe: str,
                            combo: Tuple[str, ...],
                            input_dir: str = DATA_DIR,
                            output_dir: str = OUTPUT_DIR) -> Dict[str, float]:
    path = os.path.join(input_dir, pair, f"{pair}_{timeframe}_with_indicators.pkl")
    if not os.path.exists(path):
        return {'num_trades': 0, 'win_rate': 0.0, 'avg_pips': 0.0, 'total_pips': 0.0, 'profit_factor': 0.0}
    df = pd.read_pickle(path)
    df = df.copy()
    df = df.dropna().copy()
    sig = _indicator_signal(df, combo)
    trades = _simulate_trades(df, sig)
    metrics = _evaluate(trades)
    # Save trades (PKL)
    os.makedirs(os.path.join(output_dir, pair), exist_ok=True)
    trades_path = os.path.join(output_dir, pair, f"{pair}_{timeframe}_{'_'.join(combo)}_trades.pkl")
    trades.to_pickle(trades_path)
    return metrics


def run_tests_per_tf(pairs: List[str], timeframes: List[str],
                     combos: List[Tuple[str, ...]],
                     max_per_timeframe: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        for tf in timeframes:
            tested = 0
            for combo in combos:
                logger.info(f"Testing pair={pair} timeframe={tf} indicators={'+'.join(combo)}")
                m = backtest_pair_timeframe(pair, tf, combo)
                rows.append({
                    'pair': pair,
                    'timeframe': tf,
                    'combo': '+'.join(combo),
                    **m
                })
                tested += 1
                if max_per_timeframe is not None and tested >= max_per_timeframe:
                    break
    return pd.DataFrame(rows)


def main():
    logger.info("Running backtests...")
    pairs = ["EUR_USD"]
    timeframes = ["M5","M15","M30","H1","H4","D","W"]

    # 1-indicator tests (limit per timeframe for speed)
    res1 = run_tests_per_tf(pairs, timeframes, [(i,) for i in SINGLE_INDICATORS], max_per_timeframe=100)
    res1.to_pickle(os.path.join(OUTPUT_DIR, "results_single.pkl"))

    # 2-indicator combos (limit per timeframe)
    res2 = run_tests_per_tf(pairs, timeframes, COMBOS_2, max_per_timeframe=200)
    res2.to_pickle(os.path.join(OUTPUT_DIR, "results_combo2.pkl"))

    # 3-indicator combos (limit per timeframe)
    res3 = run_tests_per_tf(pairs, timeframes, COMBOS_3, max_per_timeframe=400)
    res3.to_pickle(os.path.join(OUTPUT_DIR, "results_combo3.pkl"))

    logger.info("Backtesting complete. Results saved to:")
    logger.info(OUTPUT_DIR)

if __name__ == "__main__":
    main() 