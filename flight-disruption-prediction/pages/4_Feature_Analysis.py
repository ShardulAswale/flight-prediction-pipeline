"""FE5: Feature Analysis - correlation, importance, SHAP, mutual information."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Feature Analysis", page_icon="📈", layout="wide")
st.markdown("# 📈 Feature Analysis")

ml_path = Path("data/processed/ml_dataset.parquet")
if not ml_path.exists():
    st.warning("ML dataset not available.")
    st.stop()

df = pd.read_parquet(ml_path)
if df.empty:
    st.warning("ML dataset is empty.")
    st.stop()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
tabs = st.tabs(["Correlation", "Feature Importance", "Category Importance", "SHAP", "Mutual Information", "Route Deviation", "Distributions"])

with tabs[0]:
    st.subheader("Correlation Matrix (Pearson)")
    if numeric_cols:
        corr = df[numeric_cols].corr(method="pearson")
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns available.")

with tabs[1]:
    st.subheader("Feature Importance")
    imp_path = Path("models/feature_importance.json")
    if imp_path.exists():
        with open(imp_path) as f:
            importance = json.load(f)
        imp_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())}).sort_values(
            "importance", ascending=False
        )
        top = imp_df.head(10)
        fig = px.bar(top.sort_values("importance"), x="importance", y="feature", orientation="h", title="Top 10 Features")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance file not found yet.")

with tabs[2]:
    st.subheader("Feature Importance by Category")
    category_img = Path("outputs/feature_importance_by_category.png")
    if category_img.exists():
        st.image(str(category_img), use_container_width=True)
    else:
        st.info("Grouped category importance image not available yet.")

with tabs[3]:
    st.subheader("SHAP Summary")
    shap_img = Path("outputs/shap_summary.png")
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
    else:
        st.info("SHAP summary image not available.")

with tabs[4]:
    st.subheader("Mutual Information vs Label")
    if "label" in df.columns and numeric_cols:
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.preprocessing import LabelEncoder

            X = df[numeric_cols].fillna(0)
            y = LabelEncoder().fit_transform(df["label"].astype(str))
            mi = mutual_info_classif(X, y, random_state=42)
            mi_df = pd.DataFrame({"feature": numeric_cols, "mi_score": mi}).sort_values("mi_score", ascending=False)
            fig = px.bar(mi_df.head(20).sort_values("mi_score"), x="mi_score", y="feature", orientation="h", title="Top MI Features")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"MI computation unavailable: {e}")
    else:
        st.info("Need label + numeric features for MI.")

with tabs[5]:
    st.subheader("Route Deviation Features")
    deviation_cols = [
        c for c in [
            "lateral_deviation_mean_km", "lateral_deviation_max_km", "lateral_deviation_std_km",
            "route_stretch_ratio", "approach_deviation_km",
            "altitude_deviation_mean_m", "altitude_deviation_max_m",
        ]
        if c in df.columns
    ]
    if deviation_cols:
        coverage = pd.DataFrame({
            "feature": deviation_cols,
            "coverage_pct": [round(float(df[c].notna().mean() * 100), 2) for c in deviation_cols],
        })
        fig_cov = px.bar(coverage, x="feature", y="coverage_pct", title="Route Deviation Coverage %")
        st.plotly_chart(fig_cov, use_container_width=True)

        selected_dev = st.selectbox("Deviation feature", deviation_cols, key="route_dev_feature")
        plot_df = df[[selected_dev] + (["label"] if "label" in df.columns else [])].copy()
        if "label" in plot_df.columns:
            fig = px.histogram(plot_df, x=selected_dev, color="label", barmode="overlay", opacity=0.65)
        else:
            fig = px.histogram(plot_df, x=selected_dev)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No route deviation columns available in the ML dataset.")

with tabs[6]:
    st.subheader("Feature Distributions")
    if not numeric_cols:
        st.info("No numeric columns available.")
    else:
        selected = st.multiselect("Features", numeric_cols, default=numeric_cols[: min(3, len(numeric_cols))])
        for feat in selected:
            if "label" in df.columns:
                fig = px.histogram(df, x=feat, color="label", barmode="overlay", opacity=0.6)
            else:
                fig = px.histogram(df, x=feat)
            st.plotly_chart(fig, use_container_width=True)
