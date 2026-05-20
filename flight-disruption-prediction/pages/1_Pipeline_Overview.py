"""Pipeline overview page with lightweight metadata reads."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.streamlit_utils import get_parquet_metadata, get_run_metadata, load_json, render_walkthrough_header

st.set_page_config(page_title="Data and Pipeline", page_icon="Flow", layout="wide")
st.markdown("# Step 2. Data and Pipeline")
render_walkthrough_header("pipeline")

st.subheader("Stage Outputs")

stages = {
    'Ingest (Euro)': 'data/processed/eurocontrol_combined.parquet',
    'Ingest (BTS)': 'data/processed/bts_combined.parquet',
    'Features': 'data/processed/trajectory_features.parquet',
    'Merge': 'data/processed/ml_dataset_merged.parquet',
    'Weather': 'data/processed/ml_dataset_weather.parquet',
    'Labeled': 'data/processed/ml_dataset_labeled.parquet',
    'Final ML Dataset': 'data/processed/ml_dataset.parquet',
}

cols = st.columns(4)
for i, (name, path) in enumerate(stages.items()):
    with cols[i % 4]:
        meta = get_parquet_metadata(path)
        if meta.get('exists'):
            st.metric(name, f"{meta.get('rows', 0):,} rows", delta="Ready")
        else:
            st.metric(name, "Missing", delta="Not run")

st.divider()
st.subheader("Merge Funnel")

funnel_labels = [
    ('ADS-B Features', 'data/processed/trajectory_features.parquet'),
    ('After Merge', 'data/processed/ml_dataset_merged.parquet'),
    ('After Weather', 'data/processed/ml_dataset_weather.parquet'),
    ('After Labeling', 'data/processed/ml_dataset_labeled.parquet'),
    ('Final Dataset', 'data/processed/ml_dataset.parquet'),
]
funnel_data = {label: get_parquet_metadata(path).get('rows', 0) for label, path in funnel_labels if get_parquet_metadata(path).get('exists')}

if funnel_data:
    fig = go.Figure(
        go.Funnel(
            y=list(funnel_data.keys()),
            x=list(funnel_data.values()),
            textposition="inside",
            textinfo="value+percent initial",
            marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']),
        )
    )
    fig.update_layout(title="Data Flow Through Pipeline", height=400)
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No pipeline outputs found. Run the pipeline first.")

st.divider()
st.subheader("Recent Pipeline Runs")

runs = get_run_metadata()
if runs:
    runs_df = pd.DataFrame(runs[:20])
    display_cols = [c for c in ['stage', 'status', 'start_time', 'duration_seconds', 'row_count'] if c in runs_df.columns]
    st.dataframe(runs_df[display_cols], width="stretch")
else:
    st.info("No run logs found yet.")

st.divider()
st.subheader("Quality Gates Summary")
qg = load_json('logs/quality_gates_report.json')
if qg:
    c1, c2, c3 = st.columns(3)
    c1.metric("Missingness", "PASS" if qg.get('missingness_passed') else "FAIL")
    c2.metric("Duplicates", "PASS" if qg.get('duplicates_passed') else "FAIL")
    c3.metric("Label Balance", "PASS" if qg.get('label_balance_passed') else "FAIL")
else:
    st.info("Quality gates not yet run.")
