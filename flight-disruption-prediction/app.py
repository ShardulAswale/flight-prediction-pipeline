"""Streamlit navigation entry point for the dissertation walkthrough app."""
from __future__ import annotations

import streamlit as st

PAGES = [
    st.Page("pages/0_Study_Overview.py", title="Study Overview", default=True),
    st.Page("pages/1_Pipeline_Overview.py", title="Data and Pipeline", url_path="pipeline_overview"),
    st.Page("pages/2_Data_Explorer.py", title="Dataset Construction", url_path="data_explorer"),
    st.Page("pages/3_Trajectory_Map.py", title="ADS-B Trajectory Evidence", url_path="trajectory_map"),
    st.Page("pages/4_Feature_Analysis.py", title="Feature Engineering", url_path="feature_analysis"),
    st.Page("pages/5_Model_Performance.py", title="Model Results", url_path="model_performance"),
    st.Page("pages/6_Data_Quality.py", title="Quality and Limitations", url_path="data_quality"),
    st.Page("pages/7_Predictions_Explorer.py", title="Prediction Demo", url_path="predictions_explorer"),
    st.Page("pages/8_Pipeline_Runbook.py", title="Pipeline Runbook", url_path="pipeline_runbook"),
]

selected_page = st.navigation(PAGES, position="sidebar", expanded=True)
selected_page.run()
