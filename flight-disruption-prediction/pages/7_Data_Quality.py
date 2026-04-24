"""FE8: Data Quality dashboard with monthly missingness and drift trends."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Data Quality", page_icon="🛡️", layout="wide")
st.markdown("# 🛡️ Data Quality Dashboard")

ml_path = Path("data/processed/ml_dataset.parquet")
df = pd.read_parquet(ml_path) if ml_path.exists() else pd.DataFrame()

st.subheader("Quality Gates")
qg_path = Path("logs/quality_gates_report.json")
if qg_path.exists():
    with open(qg_path) as f:
        qg = json.load(f)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Missingness", "PASS" if qg.get("missingness_passed") else "FAIL")
    c2.metric("Duplicates", "PASS" if qg.get("duplicates_passed") else "FAIL")
    c3.metric("Label Balance", "PASS" if qg.get("label_balance_passed") else "FAIL")
    c4.metric("Overall", "PASS" if qg.get("overall_passed") else "FAIL")
else:
    st.info("No quality gate report found yet.")

st.divider()
st.subheader("Monthly Missingness Heatmap")
if not df.empty and "scheduled_dep" in df.columns:
    work = df.copy()
    work["month"] = pd.to_datetime(work["scheduled_dep"], errors="coerce", utc=True).dt.to_period("M").astype(str)
    cols = [c for c in work.columns if c != "month"]
    rows = []
    for month, g in work.groupby("month"):
        null_pct = (g[cols].isna().mean() * 100).to_dict()
        for col, pct in null_pct.items():
            rows.append({"month": month, "column": col, "null_pct": float(pct)})
    heat_df = pd.DataFrame(rows)
    if not heat_df.empty:
        fig = px.density_heatmap(heat_df, x="month", y="column", z="null_pct", color_continuous_scale="YlOrRd")
        st.plotly_chart(fig, width="stretch")
else:
    st.info("Need non-empty ML dataset with scheduled_dep to compute monthly missingness.")

st.divider()
st.subheader("Weather Coverage")
weather_cov_path = Path("logs/weather_coverage.json")
if weather_cov_path.exists():
    with open(weather_cov_path) as f:
        cov = json.load(f)
    cov_df = pd.DataFrame(
        [{"feature": k, "coverage_pct": v.get("coverage_pct", 0), "null_pct": v.get("null_pct", 100)} for k, v in cov.items()]
    )
    fig = px.bar(cov_df, x="feature", y="coverage_pct", title="Weather Feature Coverage %")
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No weather coverage report found.")

st.divider()
st.subheader("Drift Alerts & PSI Trends")
drift_path = Path("logs/drift_report.json")
if drift_path.exists():
    with open(drift_path) as f:
        drift = pd.DataFrame(json.load(f))
    if not drift.empty:
        alert_df = drift[drift["drift_alert"] == True] if "drift_alert" in drift.columns else pd.DataFrame()
        st.metric("Total Drift Alerts", int(len(alert_df)))
        feat_options = sorted(drift["feature"].astype(str).unique().tolist())
        feat = st.selectbox("Feature", feat_options)
        trend = drift[drift["feature"].astype(str) == feat].sort_values("month")
        fig = px.line(trend, x="month", y="psi", markers=True, title=f"PSI Trend - {feat}")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="Alert threshold 0.2")
        st.plotly_chart(fig, width="stretch")
    heatmap_path = Path("outputs/drift_heatmap.png")
    if heatmap_path.exists():
        st.image(str(heatmap_path), width="stretch")
else:
    st.info("No drift report found. Run `python main.py --stage drift`.")

st.divider()
st.subheader("Data Freshness")
log_files = sorted(Path("logs").glob("run_*.json"), reverse=True) if Path("logs").exists() else []
if log_files:
    with open(log_files[0]) as f:
        last_run = json.load(f)
    st.write(f"Last pipeline run: `{last_run.get('end_time', 'unknown')}`")
else:
    st.info("No run metadata found.")
