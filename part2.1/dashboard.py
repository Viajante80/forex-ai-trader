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
                available_data[pair_dir] = []
                for file in os.listdir(pair_path):
                    if file.endswith('_with_indicators.pkl'):
                        timeframe = file.split('_')[2]  # Extract timeframe
                        available_data[pair_dir].append(timeframe)
    
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
    """Create candlestick chart with volume"""
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
    
    return fig

def add_technical_indicators(fig, df, selected_indicators):
    """Add selected technical indicators to the chart"""
    
    indicator_configs = {
        'SMA': {
            'columns': ['sma_10', 'sma_20', 'sma_50', 'sma_200'],
            'colors': ['#ff9800', '#2196f3', '#4caf50', '#9c27b0'],
            'names': ['SMA 10', 'SMA 20', 'SMA 50', 'SMA 200']
        },
        'EMA': {
            'columns': ['ema_10', 'ema_20', 'ema_50'],
            'colors': ['#ff5722', '#3f51b5', '#009688'],
            'names': ['EMA 10', 'EMA 20', 'EMA 50']
        },
        'Bollinger Bands': {
            'columns': ['bb_upper', 'bb_middle', 'bb_lower'],
            'colors': ['#e91e63', '#607d8b', '#e91e63'],
            'names': ['BB Upper', 'BB Middle', 'BB Lower']
        },
        'MACD': {
            'columns': ['macd', 'macd_signal'],
            'colors': ['#2196f3', '#ff9800'],
            'names': ['MACD', 'MACD Signal']
        }
    }
    
    for indicator in selected_indicators:
        if indicator in indicator_configs:
            config = indicator_configs[indicator]
            for col, color, name in zip(config['columns'], config['colors'], config['names']):
                if col in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df[col],
                            mode='lines',
                            name=name,
                            line=dict(color=color, width=2),
                            opacity=0.8
                        ),
                        row=1, col=1
                    )
    
    return fig

def create_indicator_subplot(df, indicator_type):
    """Create subplot for specific indicator types"""
    
    if indicator_type == 'RSI':
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['rsi'],
                mode='lines',
                name='RSI',
                line=dict(color='#9c27b0', width=2)
            )
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        fig.update_layout(title="RSI", height=300, template="plotly_white")
        
    elif indicator_type == 'Stochastic':
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['stoch_k'],
                mode='lines',
                name='%K',
                line=dict(color='#2196f3', width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['stoch_d'],
                mode='lines',
                name='%D',
                line=dict(color='#ff9800', width=2)
            )
        )
        fig.add_hline(y=80, line_dash="dash", line_color="red")
        fig.add_hline(y=20, line_dash="dash", line_color="green")
        fig.update_layout(title="Stochastic Oscillator", height=300, template="plotly_white")
        
    elif indicator_type == 'ATR':
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['atr'],
                mode='lines',
                name='ATR',
                line=dict(color='#4caf50', width=2)
            )
        )
        fig.update_layout(title="Average True Range", height=300, template="plotly_white")
        
    elif indicator_type == 'ADX':
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx'],
                mode='lines',
                name='ADX',
                line=dict(color='#607d8b', width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx_pos'],
                mode='lines',
                name='+DI',
                line=dict(color='#4caf50', width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['adx_neg'],
                mode='lines',
                name='-DI',
                line=dict(color='#f44336', width=2)
            )
        )
        fig.update_layout(title="ADX", height=300, template="plotly_white")
    
    return fig

def display_data_summary(df, pair, timeframe):
    """Display data summary metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}",
            delta=None
        )
    
    with col2:
        date_range = f"{df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}"
        st.metric(
            label="Date Range",
            value=date_range,
            delta=None
        )
    
    with col3:
        avg_volume = f"{df['volume'].mean():,.0f}"
        st.metric(
            label="Avg Volume",
            value=avg_volume,
            delta=None
        )
    
    with col4:
        price_range = f"{df['norm_close'].max() - df['norm_close'].min():.2f}"
        st.metric(
            label="Price Range (pips)",
            value=price_range,
            delta=None
        )

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">📈 Forex AI Trader - Data Visualization</h1>', unsafe_allow_html=True)
    
    # Load available data
    available_data = load_available_data()
    
    if not available_data:
        st.error("No data found! Please run the technical indicators script first.")
        st.info("Run: `cd part2.1 && uv run add_technical_indicators.py`")
        return
    
    # Sidebar controls
    st.sidebar.header("📊 Data Selection")
    
    # Currency pair selection
    pair = st.sidebar.selectbox(
        "Select Currency Pair",
        list(available_data.keys()),
        index=0
    )
    
    # Timeframe selection
    timeframes = available_data[pair]
    timeframe = st.sidebar.selectbox(
        "Select Timeframe",
        timeframes,
        index=0
    )
    
    # Date range selection
    st.sidebar.header("📅 Date Range")
    
    # Load data to get date range
    df = load_data(pair, timeframe)
    if df is None:
        return
    
    min_date = df.index.min()
    max_date = df.index.max()
    
    # Cap the maximum selectable date at yesterday (today - 1 day)
    yesterday = (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)).date()
    max_selectable = min(max_date.date(), yesterday)
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date.date(), max_selectable),
        min_value=min_date.date(),
        max_value=max_selectable,
        key="date_range"
    )
    
    # Sanitize selection to ensure end date never exceeds yesterday
    if isinstance(date_range, tuple) and len(date_range) == 2:
        raw_start, raw_end = date_range
        sanitized_start = max(min_date.date(), min(raw_start, max_selectable))
        sanitized_end = min(raw_end, max_selectable)
        
        # If Streamlit preset chose today, clamp to yesterday and rerun to fix UI warning
        if (raw_start, raw_end) != (sanitized_start, sanitized_end):
            st.session_state["date_range"] = (sanitized_start, sanitized_end)
            st.rerun()
        
        # Convert to timezone-aware timestamps to match the data index
        start_date = pd.Timestamp(sanitized_start).tz_localize('UTC')
        end_date = pd.Timestamp(sanitized_end).tz_localize('UTC')
        
        # Filter data by date range
        df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
    else:
        df_filtered = df
    
    # Technical indicators selection
    st.sidebar.header("📈 Technical Indicators")
    
    available_indicators = ['SMA', 'EMA', 'Bollinger Bands', 'MACD']
    selected_overlay_indicators = st.sidebar.multiselect(
        "Overlay Indicators (on price chart)",
        available_indicators,
        default=['SMA', 'EMA']
    )
    
    # Separate indicator subplots
    st.sidebar.header("📊 Separate Indicator Charts")
    available_separate_indicators = ['RSI', 'Stochastic', 'ATR', 'ADX']
    selected_separate_indicators = st.sidebar.multiselect(
        "Separate Indicator Charts",
        available_separate_indicators,
        default=['RSI']
    )
    
    # Main content
    if not df_filtered.empty:
        # Data summary
        st.subheader(f"📊 Data Summary - {pair} {timeframe}")
        display_data_summary(df_filtered, pair, timeframe)
        
        # Main price chart
        st.subheader(f"📈 Price Chart - {pair} {timeframe}")
        
        # Create candlestick chart
        fig = create_candlestick_chart(df_filtered, f"{pair} {timeframe}")
        
        # Add selected technical indicators
        fig = add_technical_indicators(fig, df_filtered, selected_overlay_indicators)
        
        # Update layout
        fig.update_layout(
            title=f"{pair} {timeframe} - Price Chart with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price (pips)",
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Separate indicator charts
        if selected_separate_indicators:
            st.subheader("📊 Technical Indicators")
            
            # Create columns for indicator charts
            cols = st.columns(min(len(selected_separate_indicators), 2))
            
            for i, indicator in enumerate(selected_separate_indicators):
                col_idx = i % 2
                with cols[col_idx]:
                    indicator_fig = create_indicator_subplot(df_filtered, indicator)
                    st.plotly_chart(indicator_fig, use_container_width=True)
        
        # Data table
        st.subheader("📋 Data Table")
        
        # Select columns to display
        price_cols = ['norm_open', 'norm_high', 'norm_low', 'norm_close', 'volume']
        indicator_cols = [col for col in df_filtered.columns if col not in price_cols]
        
        selected_columns = st.multiselect(
            "Select columns to display",
            df_filtered.columns.tolist(),
            default=price_cols
        )
        
        if selected_columns:
            st.dataframe(
                df_filtered[selected_columns].tail(100),
                use_container_width=True
            )
        
        # Download options
        st.subheader("💾 Download Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df_filtered.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"{pair}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create a sample for download (first 1000 rows)
            sample_data = df_filtered.head(1000).to_csv()
            st.download_button(
                label="Download Sample (1000 rows)",
                data=sample_data,
                file_name=f"{pair}_{timeframe}_sample.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("No data available for the selected date range.")

if __name__ == "__main__":
    main() 