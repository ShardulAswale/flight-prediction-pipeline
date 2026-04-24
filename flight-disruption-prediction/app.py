"""
FE1: Multi-page Streamlit Application — Main Entry Point
"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.streamlit_utils import get_parquet_metadata, read_parquet_limited

st.set_page_config(
    page_title="Flight Disruption Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header { font-size: 1.1rem; color: #a0aec0; margin-bottom: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 1.5rem; border: 1px solid #2d3748;
    }
    .stMetric > div { background: #1a1a2e; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">✈️ Flight Disruption Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-end ML pipeline for predicting flight delays and cancellations</div>', unsafe_allow_html=True)

# Dashboard overview
col1, col2, col3, col4 = st.columns(4)

# Try to load stats
try:
    from src.utils import load_config
    config = load_config('configs/config.yaml')
    data_dir = config['paths']['processed_data_dir']
    
    ml_path = os.path.join(data_dir, config['paths']['ml_dataset_file'])
    if os.path.exists(ml_path):
        meta = get_parquet_metadata(ml_path)
        sample_cols = [c for c in ["label", "region", "source_dataset"] if c in meta.get("columns", [])]
        df = read_parquet_limited(ml_path, columns=sample_cols, limit=200_000)
        col1.metric("Total Flights", f"{meta.get('rows', 0):,}")
        if 'label' in df.columns:
            disrupted = int((df['label'] != 'Normal').sum())
            label = f"{disrupted:,}" if meta.get('rows', 0) <= len(df) else f"{disrupted:,} (sample)"
            col2.metric("Disrupted", label)
        if 'region' in df.columns:
            col3.metric("Regions", df['region'].nunique())
        if 'source_dataset' in df.columns:
            col4.metric("Data Sources", df['source_dataset'].nunique())
    else:
        col1.metric("Status", "No Data")
        col2.info("Run pipeline first")
except Exception as e:
    st.info("Run the pipeline to see dashboard metrics.")

st.divider()

st.markdown("## 📋 Quick Navigation")

pages = {
    "📊 Pipeline Overview": "View pipeline stage status, row counts, and merge funnel",
    "🔍 Data Explorer": "Browse and filter the ML dataset interactively",
    "🗺️ Trajectory Map": "Visualise flight trajectories on an interactive map",
    "📈 Feature Analysis": "Correlation matrices, importance rankings, SHAP summaries",
    "🤖 Model Performance": "Compare trained models, confusion matrices, ROC curves",
    "🔮 Predictions Explorer": "Interactive prediction tool with SHAP explanations",
    "🛡️ Data Quality": "Quality gates, missingness, drift alerts",
}

cols = st.columns(3)
for i, (page, desc) in enumerate(pages.items()):
    with cols[i % 3]:
        st.markdown(f"### {page}")
        st.markdown(f"_{desc}_")

st.divider()
st.caption("Built for Flight Disruption Prediction Research • Navigate using the sidebar →")
