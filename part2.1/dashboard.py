import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Forex AI Trader - Data Visualization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Directory settings
DATA_DIR = "../trading_ready_data"
HISTORICAL_DATA_DIR = "../oanda_historical_data"


def load_available_data():
    """Load available currency pairs and timeframes"""
    available_data = {}
    
    if os.path.exists(DATA_DIR):
        for pair_dir in os.listdir(DATA_DIR):
            pair_path = os.path.join(DATA_DIR, pair_dir)
            if os.path.isdir(pair_path):
                tfs = set()
                prefix = f"{pair_dir}_"
                suffix = "_with_indicators.pkl"
                for file in os.listdir(pair_path):
                    if file.endswith(suffix) and file.startswith(prefix):
                        # Extract timeframe between prefix and suffix
                        tf = file[len(prefix):-len(suffix)]
                        if tf:
                            tfs.add(tf)
                if tfs:
                    available_data[pair_dir] = sorted(list(tfs))
    
    return available_data


def load_data(pair, timeframe, data_type="indicators"):
    """Load data for selected pair and timeframe"""
    try:
        if data_type == "indicators":
            file_path = os.path.join(DATA_DIR, pair, f"{pair}_{timeframe}_with_indicators.pkl")
        else:
            file_path = os.path.join(HISTORICAL_DATA_DIR, pair, f"{pair}_{timeframe}_normalized.pkl")
        
        if os.path.exists(file_path):
            df = pd.read_pickle(file_path)
            df.index = pd.to_datetime(df.index)
            return df
        else:
            st.error(f"Data file not found: {file_path}")
            return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def create_candlestick_chart(df, title="Price Chart"):
    """Create candlestick chart with volume using normalized prices (pips)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(title, 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['norm_open'],
            high=df['norm_high'],
            low=df['norm_low'],
            close=df['norm_close'],
            name="OHLC",
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume bars
    colors = ['#26a69a' if close >= open else '#ef5350' 
              for close, open in zip(df['norm_close'], df['norm_open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volume",
            marker_color=colors,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    fig.update_yaxes(title_text="Price (pips)", row=1, col=1)
    
    return fig


def add_overlay_indicators(fig, df, selected, ma_periods):
    """Add selected overlay indicators to the price chart."""
    # SMA
    if 'SMA' in selected:
        for w in ma_periods:
            col = f'sma_{w}'
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'SMA {w}', line=dict(width=1.8)), row=1, col=1)
    # EMA
    if 'EMA' in selected:
        for w in ma_periods:
            col = f'ema_{w}'
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=f'EMA {w}', line=dict(width=1.8, dash='dot')), row=1, col=1)
    # Bollinger Bands
    if 'Bollinger Bands' in selected:
        for name, col, dash in [('BB Upper','bb_upper','solid'), ('BB Middle','bb_middle','dot'), ('BB Lower','bb_lower','solid')]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name, line=dict(width=1, color='#888', dash=dash)), row=1, col=1)
    # Ichimoku Cloud
    if 'Ichimoku Cloud' in selected:
        for name, col, clr in [('Ichimoku A','ichimoku_a','#8e24aa'), ('Ichimoku B','ichimoku_b','#5e35b1'), ('Conversion','ichimoku_conv','#1e88e5'), ('Base','ichimoku_base','#43a047')]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name, line=dict(width=1.2, dash='dot')), row=1, col=1)
    # Support/Resistance
    if 'Support/Resistance' in selected:
        for name, col, clr in [('S 20','sr_support_20','#26a69a'), ('R 20','sr_resistance_20','#ef5350'), ('S 50','sr_support_50','#2e7d32'), ('R 50','sr_resistance_50','#c62828')]:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], mode='lines', name=name, line=dict(width=1, color=clr, dash='dash')), row=1, col=1)
    # Fibonacci Levels
    if 'Fibonacci' in selected:
        for name in ['fib_0','fib_0236','fib_0382','fib_0500','fib_0618','fib_0786','fib_1']:
            if name in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[name], mode='lines', name=name.upper(), line=dict(width=0.8, color='#9e9e9e', dash='dot')), row=1, col=1)
    # Pivot Points
    if 'Pivot Points' in selected:
        for name in ['pivot_p','pivot_r1','pivot_s1','pivot_r2','pivot_s2']:
            if name in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df[name], mode='lines', name=name.upper(), line=dict(width=0.8, color='#795548', dash='dash')), row=1, col=1)
    # Pattern Markers
    if 'Patterns' in selected:
        markers = [
            ('bullish_engulfing', 'triangle-up', '#2e7d32'),
            ('bearish_engulfing', 'triangle-down', '#c62828'),
            ('hammer', 'circle', '#00695c'),
            ('shooting_star', 'circle-open', '#ad1457')
        ]
        for col, sym, clr in markers:
            if col in df.columns:
                pts = df[df[col] == 1]
                if not pts.empty:
                    fig.add_trace(go.Scatter(x=pts.index, y=pts['norm_close'], mode='markers', name=col, marker=dict(symbol=sym, size=8, color=clr, line=dict(width=1))), row=1, col=1)

    return fig


def create_indicator_subplot(df, indicator_type, ma_periods):
    """Create subplot for a specific indicator type."""
    if indicator_type == 'RSI' and 'rsi' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], mode='lines', name='RSI', line=dict(color='#9c27b0', width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title="RSI", height=300, template="plotly_white")
        return fig

    if indicator_type == 'MACD' and {'macd','macd_signal','macd_diff'}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df.index, y=df['macd_diff'], name='MACD Hist', marker_color='#90caf9'))
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD', line=dict(color='#1976d2', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], mode='lines', name='Signal', line=dict(color='#ff9800', width=2)))
        fig.update_layout(title="MACD", height=300, template="plotly_white")
        return fig

    if indicator_type == 'Stochastic' and {'stoch_k','stoch_d'}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], mode='lines', name='%K', line=dict(color='#2196f3', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], mode='lines', name='%D', line=dict(color='#ff9800', width=2)))
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.add_hline(y=20, line_dash="dash", line_color="green")
        fig.update_layout(title="Stochastic Oscillator", height=300, template="plotly_white")
        return fig

    if indicator_type == 'ATR' and 'atr' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['atr'], mode='lines', name='ATR', line=dict(color='#4caf50', width=2)))
        fig.update_layout(title="Average True Range", height=300, template="plotly_white")
        return fig

    if indicator_type == 'ADX' and {'adx','adx_pos','adx_neg'}.issubset(df.columns):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['adx'], mode='lines', name='ADX', line=dict(color='#607d8b', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['adx_pos'], mode='lines', name='+DI', line=dict(color='#4caf50', width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df['adx_neg'], mode='lines', name='-DI', line=dict(color='#f44336', width=2)))
        fig.update_layout(title="ADX", height=300, template="plotly_white")
        return fig

    if indicator_type == 'OBV' and 'obv' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['obv'], mode='lines', name='OBV', line=dict(color='#6d4c41', width=2)))
        fig.update_layout(title="On-Balance Volume", height=300, template="plotly_white")
        return fig

    if indicator_type == 'VWAP' and 'vwap' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], mode='lines', name='VWAP', line=dict(color='#009688', width=2)))
        fig.update_layout(title="VWAP", height=300, template="plotly_white")
        return fig

    return None


def display_data_summary(df, pair, timeframe):
    """Display data summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Records", value=f"{len(df):,}")
    with col2:
        date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
        st.metric(label="Date Range", value=date_range)
    with col3:
        avg_volume = f"{df['volume'].mean():,.0f}"
        st.metric(label="Avg Volume", value=avg_volume)
    with col4:
        price_range = f"{df['norm_close'].max() - df['norm_close'].min():.2f}"
        st.metric(label="Price Range (pips)", value=price_range)


def main():
    """Main dashboard function"""
    st.markdown('<h1 class="main-header">📈 Forex AI Trader - Data Visualization</h1>', unsafe_allow_html=True)
    
    available_data = load_available_data()
    if not available_data:
        st.error("No data found! Please run the technical indicators script first.")
        st.info("Run: `cd part2.1 && uv run add_technical_indicators.py`")
        return

    # Sidebar controls
    st.sidebar.header("📊 Data Selection")
    pair = st.sidebar.selectbox("Select Currency Pair", list(available_data.keys()), index=0)
    timeframes = available_data[pair]
    timeframe = st.sidebar.selectbox("Select Timeframe", timeframes, index=0)

    # Date range selection
    st.sidebar.header("📅 Date Range")
    df = load_data(pair, timeframe)
    if df is None:
        return
    min_date = df.index.min()
    max_date = df.index.max()
    yesterday = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)).date()
    max_selectable = min(max_date.date(), yesterday)
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date.date(), max_selectable),
        min_value=min_date.date(),
        max_value=max_selectable,
        key="date_range"
    )
    # sanitize
    if isinstance(date_range, tuple) and len(date_range) == 2:
        raw_start, raw_end = date_range
        sanitized_start = max(min_date.date(), min(raw_start, max_selectable))
        sanitized_end = min(raw_end, max_selectable)
        if (raw_start, raw_end) != (sanitized_start, sanitized_end):
            st.session_state["date_range"] = (sanitized_start, sanitized_end)
            st.rerun()
        start_date = pd.Timestamp(sanitized_start).tz_localize('UTC')
        end_date = pd.Timestamp(sanitized_end).tz_localize('UTC')
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    else:
        df_filtered = df

    # Indicator selections
    st.sidebar.header("📈 Overlay Indicators")
    ma_periods = st.sidebar.multiselect("MA Periods", [50, 80, 100, 200], default=[50, 80, 100, 200])
    overlay_options = ['SMA', 'EMA', 'Bollinger Bands', 'Ichimoku Cloud', 'Support/Resistance', 'Fibonacci', 'Pivot Points', 'Patterns']
    selected_overlays = st.sidebar.multiselect("Choose overlays", overlay_options, default=['SMA', 'EMA'])

    st.sidebar.header("📊 Separate Indicator Charts")
    separate_options = ['RSI', 'MACD', 'Stochastic', 'ATR', 'ADX', 'OBV', 'VWAP']
    selected_separate = st.sidebar.multiselect("Choose separate charts", separate_options, default=['RSI', 'MACD'])

    # Main content
    if not df_filtered.empty:
        # Summary
        st.subheader(f"📊 Data Summary - {pair} {timeframe}")
        display_data_summary(df_filtered, pair, timeframe)

        # Price chart + overlays
        st.subheader(f"📈 Price Chart - {pair} {timeframe}")
        fig = create_candlestick_chart(df_filtered, f"{pair} {timeframe}")
        fig = add_overlay_indicators(fig, df_filtered, selected_overlays, ma_periods)
        fig.update_layout(title=f"{pair} {timeframe} - Price with Selected Indicators", xaxis_title="Date", yaxis_title="Price (pips)", height=650, legend=dict(orientation='h'))
        st.plotly_chart(fig, use_container_width=True)

        # Separate indicator charts
        if selected_separate:
            st.subheader("📊 Technical Indicator Charts")
            cols = st.columns(2)
            for i, name in enumerate(selected_separate):
                fig_sep = create_indicator_subplot(df_filtered, name, ma_periods)
                if fig_sep is not None:
                    with cols[i % 2]:
                        st.plotly_chart(fig_sep, use_container_width=True)

        # Data table
        st.subheader("📋 Data Table")
        price_cols = ['norm_open', 'norm_high', 'norm_low', 'norm_close', 'volume']
        selected_columns = st.multiselect("Select columns to display", df_filtered.columns.tolist(), default=price_cols)
        if selected_columns:
            st.dataframe(df_filtered[selected_columns].tail(300), use_container_width=True)

        # Downloads
        st.subheader("💾 Download Data")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(label="Download CSV", data=df_filtered.to_csv(), file_name=f"{pair}_{timeframe}.csv", mime="text/csv")
        with col2:
            st.download_button(label="Download Sample (1000 rows)", data=df_filtered.head(1000).to_csv(), file_name=f"{pair}_{timeframe}_sample.csv", mime="text/csv")

    else:
        st.warning("No data available for the selected date range.")


if __name__ == "__main__":
    main() 