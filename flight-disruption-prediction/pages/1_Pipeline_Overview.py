"""FE2: Pipeline Overview — Stage status, row counts, merge funnel."""
import streamlit as st
import pandas as pd
import os, sys, json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.streamlit_utils import load_dataset, load_json, get_run_metadata

st.set_page_config(page_title="Pipeline Overview", page_icon="📊", layout="wide")
st.markdown("# 📊 Pipeline Overview")

# ── Stage Status ─────────────────────────────────────────────────────────
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
        exists = os.path.exists(path)
        if exists:
            try:
                df = pd.read_parquet(path)
                st.metric(name, f"{len(df):,} rows", delta="✅ Ready")
            except Exception:
                st.metric(name, "Error", delta="⚠️")
        else:
            st.metric(name, "Missing", delta="❌ Not run")

# ── Merge Funnel ─────────────────────────────────────────────────────────
st.divider()
st.subheader("Merge Funnel")

funnel_data = {}
for label, path in [
    ('ADS-B Features', 'data/processed/trajectory_features.parquet'),
    ('After Merge', 'data/processed/ml_dataset_merged.parquet'),
    ('After Weather', 'data/processed/ml_dataset_weather.parquet'),
    ('After Labeling', 'data/processed/ml_dataset_labeled.parquet'),
    ('Final Dataset', 'data/processed/ml_dataset.parquet'),
]:
    if os.path.exists(path):
        try:
            funnel_data[label] = len(pd.read_parquet(path))
        except Exception:
            funnel_data[label] = 0

if funnel_data:
    import plotly.graph_objects as go
    fig = go.Figure(go.Funnel(
        y=list(funnel_data.keys()),
        x=list(funnel_data.values()),
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
    ))
    fig.update_layout(title="Data Flow Through Pipeline", height=400)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No pipeline outputs found. Run the pipeline first.")

# ── Recent Runs ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent Pipeline Runs")

runs = get_run_metadata()
if runs:
    runs_df = pd.DataFrame(runs[:20])
    display_cols = [c for c in ['stage', 'status', 'start_time', 'duration_seconds', 'row_count'] if c in runs_df.columns]
    st.dataframe(runs_df[display_cols], use_container_width=True)
else:
    st.info("No run logs found yet.")

# ── Quality Gates ────────────────────────────────────────────────────────
st.divider()
st.subheader("Quality Gates Summary")
qg = load_json('logs/quality_gates_report.json')
if qg:
    c1, c2, c3 = st.columns(3)
    c1.metric("Missingness", "✅ Pass" if qg.get('missingness_passed') else "❌ Fail")
    c2.metric("Duplicates", "✅ Pass" if qg.get('duplicates_passed') else "❌ Fail")
    c3.metric("Label Balance", "✅ Pass" if qg.get('label_balance_passed') else "❌ Fail")
else:
    st.info("Quality gates not yet run.")
