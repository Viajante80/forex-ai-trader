import streamlit as st
import pandas as pd
import plotly.express as px
import os

RESULTS_DIR = "../backtest_strategies"

st.set_page_config(page_title="Backtest Results", page_icon="📊", layout="wide")

st.title("📊 Backtest Results Dashboard")

# Refresh control
colr1, colr2 = st.columns([1,4])
with colr1:
    if st.button("🔄 Refresh results"):
        st.cache_data.clear()
        st.rerun()

if not os.path.exists(RESULTS_DIR):
    st.warning("No results found. Run the backtest engine first.")
    st.stop()


def list_result_files() -> list:
    return [f for f in os.listdir(RESULTS_DIR) if f.startswith("results_") and f.endswith(".pkl")]


files = list_result_files()
if not files:
    st.warning("No result PKLs found.")
    st.info("Run: `cd part2.2 && uv run backtest_engine.py` to generate results.")
    st.stop()


def dir_signature(files: list) -> tuple:
    sig = []
    for f in sorted(files):
        p = os.path.join(RESULTS_DIR, f)
        try:
            sig.append((f, os.path.getmtime(p), os.path.getsize(p)))
        except FileNotFoundError:
            continue
    return tuple(sig)


@st.cache_data
def load_results(sig: tuple) -> pd.DataFrame:
    # Recompute file list on each call; cache is invalidated when sig changes
    current_files = list_result_files()
    dfs = []
    for f in current_files:
        path = os.path.join(RESULTS_DIR, f)
        try:
            df = pd.read_pickle(path)
            df['source'] = f
            dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to read {f}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


sig = dir_signature(files)
results = load_results(sig)

if results.empty:
    st.warning("Result files loaded but contain no rows. Check your backtest run and filters.")
    st.stop()

# Select backtest source (use consistent key so Streamlit preserves selection)
source_options = sorted(results['source'].dropna().unique().tolist())
friendly = {
    'results_single.pkl': 'Single Indicators',
    'results_combo2.pkl': '2-Indicator Combos',
    'results_combo3.pkl': '3-Indicator Combos',
}
label_to_source = {friendly.get(s, s): s for s in source_options}
source_label = st.sidebar.selectbox("Backtest file", options=["All"] + list(label_to_source.keys()), index=0, key="source_select")
if source_label != "All":
    selected_source = label_to_source[source_label]
    results = results[results['source'] == selected_source]
    source_state_key = selected_source
else:
    selected_source = None
    source_state_key = "All"

# Timeframe select with per-source memory (via widget key)
tf_options = sorted(results['timeframe'].dropna().unique().tolist())
if not tf_options:
    st.warning("No timeframes available in the selected results.")
    st.stop()

# Dynamic key so each source keeps its own widget value
tf_key = f"timeframe_select_{source_state_key}"
# Use the widget's existing value if available and valid; otherwise default to first option
current_tf_value = st.session_state.get(tf_key)
default_index = tf_options.index(current_tf_value) if current_tf_value in tf_options else 0
timeframe = st.sidebar.selectbox("Timeframe", tf_options, index=default_index, key=tf_key)

# Other filters
col1, col2, col3 = st.columns(3)
with col1:
    min_trades = st.select_slider("Min trades", options=list(range(0, int(results['num_trades'].max() or 0)+1, 5)), value=10)
with col2:
    metric = st.selectbox("Sort by", ["win_rate", "total_pips", "profit_factor", "avg_pips"], index=0)
with col3:
    top_n = st.slider("Show top N", 5, 50, 25)

filtered = results[results['timeframe'] == timeframe]
filtered = filtered[filtered['num_trades'] >= min_trades]

st.subheader("Top Strategies")
if not filtered.empty:
    top = filtered.sort_values(metric, ascending=False).head(top_n)
    st.dataframe(top, use_container_width=True)

    fig = px.bar(top, x='combo', y=metric, color='win_rate', title=f"Top {metric} - {timeframe}")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No strategies match the current filter.")

st.subheader("Summary Metrics")
agg = filtered.agg({
    'num_trades': 'sum',
    'win_rate': 'mean',
    'avg_pips': 'mean',
    'total_pips': 'sum',
    'profit_factor': 'mean'
}).to_frame('value')
st.dataframe(agg) 