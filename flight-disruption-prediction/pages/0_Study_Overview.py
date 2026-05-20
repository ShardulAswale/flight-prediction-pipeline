"""Dissertation walkthrough landing page for the Streamlit application."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.streamlit_utils import (
    WALKTHROUGH_STEPS,
    get_parquet_metadata,
    read_parquet_limited,
    render_walkthrough_header,
)


def page_href(page: str) -> str:
    page_name = Path(page).stem
    if page_name.startswith("0_"):
        return "/"
    return "/" + page_name.split("_", 1)[1].lower()


st.set_page_config(
    page_title="Flight Disruption Prediction",
    page_icon="Flight",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            border: 1px solid #d7e0ea;
            background: linear-gradient(135deg, #f8fbff 0%, #eef5f7 55%, #f6f1e8 100%);
            margin-bottom: 1rem;
        }
        .hero h1 {
            color: #17324d;
            font-size: 2.35rem;
            margin-bottom: 0.35rem;
        }
        .hero p {
            color: #425466;
            font-size: 1.02rem;
            margin-bottom: 0;
        }
        .step-card {
            min-height: 185px;
            padding: 1rem 1.05rem;
            border-radius: 16px;
            border: 1px solid #d7e0ea;
            background: #ffffff;
            box-shadow: 0 1px 2px rgba(23, 50, 77, 0.08);
        }
        .step-kicker {
            color: #0f6b78;
            font-weight: 700;
            font-size: 0.82rem;
            letter-spacing: 0.02em;
            text-transform: uppercase;
        }
        .step-title {
            color: #17324d;
            font-size: 1.12rem;
            font-weight: 700;
            margin: 0.25rem 0 0.45rem 0;
        }
        .step-copy {
            color: #52616f;
            font-size: 0.92rem;
            line-height: 1.35;
        }
        .step-link {
            display: inline-block;
            margin: 0.65rem 0 1.15rem 0;
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            border: 1px solid #b7ccd8;
            color: #0f5f6e;
            background: #f6fbfd;
            font-weight: 700;
            font-size: 0.9rem;
            line-height: 1.25;
            white-space: normal;
            text-decoration: none;
        }
        .step-link:hover {
            background: #eaf6f8;
            border-color: #79aebd;
            text-decoration: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Flight Disruption Prediction System Overview</h1>
        <p>
            End-to-end implementation summary covering data acquisition, schedule-aware ADS-B processing,
            feature engineering, binary disruption modelling, evaluation, limitations, and inference output.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_walkthrough_header("overview")

st.subheader("Research Aim")
st.write(
    "Develop and evaluate a reproducible machine-learning pipeline that combines schedule data, "
    "ADS-B trajectory evidence, and weather context to estimate flight disruption risk."
)

summary_cols = st.columns([1.4, 1])
with summary_cols[0]:
    framework_path = Path("outputs/chapter_visuals/Figure 1.1 - Study Framework Overview.png")
    if framework_path.exists():
        st.image(str(framework_path), caption="Study framework overview", width="stretch")
    else:
        st.info("Study framework figure not found in `outputs/chapter_visuals/`.")

with summary_cols[1]:
    st.subheader("Final System")
    st.markdown(
        """
        - Final target: `Normal` vs `Disrupted`
        - Final labelled source: BTS records
        - ADS-B use: schedule-aware trace matching and compact flight-level features
        - Weather use: airport METAR context
        - Trained models: Logistic Regression, Random Forest, XGBoost
        - Output: disruption probability plus SHAP-supported explanation
        """
    )

st.divider()
st.subheader("Current Evidence Snapshot")

metric_cols = st.columns(4)
try:
    from src.utils import load_config

    config = load_config("configs/config.yaml")
    data_dir = Path(config["paths"]["processed_data_dir"])
    ml_path = data_dir / config["paths"]["ml_dataset_file"]
    feature_list_path = Path("models/feature_list.json")
    comparison_path = Path("models/model_comparison.json")

    ml_meta = get_parquet_metadata(ml_path)
    metric_cols[0].metric("Final ML records", f"{ml_meta.get('rows', 0):,}")

    sample_cols = [c for c in ["label"] if c in ml_meta.get("columns", [])]
    sample = read_parquet_limited(ml_path, columns=sample_cols, limit=200_000)
    disrupted = int((sample["label"] != "Normal").sum()) if "label" in sample.columns else 0
    disrupted_label = f"{disrupted:,}" if len(sample) == ml_meta.get("rows", 0) else f"{disrupted:,} in preview"
    metric_cols[1].metric("Disrupted records", disrupted_label)

    feature_count = 0
    if feature_list_path.exists():
        with open(feature_list_path, encoding="utf-8") as handle:
            feature_count = len(json.load(handle))
    metric_cols[2].metric("Training features", f"{feature_count}")

    best_model = "Not available"
    if comparison_path.exists():
        with open(comparison_path, encoding="utf-8") as handle:
            best_model = json.load(handle).get("best_model", "Not available")
    metric_cols[3].metric("Best saved model", best_model)
except Exception:
    st.info("Run outputs are not fully available, so some summary metrics could not be loaded.")

st.divider()
st.subheader("System Presentation Structure")
st.write(
    "The system is organised as a staged evidence flow. It begins with the research problem and data "
    "pipeline, then moves through dataset construction, ADS-B trajectory evidence, engineered features, "
    "model evaluation, quality limitations, and the final prediction interface."
)

step_cols = st.columns(2)
for idx, step in enumerate(WALKTHROUGH_STEPS):
    with step_cols[idx % 2]:
        st.markdown(
            f"""
            <div class="step-card">
                <div class="step-kicker">Step {step['number']} | {step['chapter']}</div>
                <div class="step-title">{step['title']}</div>
                <div class="step-copy">{step['purpose']}</div>
                <a class="step-link" href="{page_href(step['page'])}" target="_self">Open {step['title']}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()
st.caption(
    "The pipeline runbook is provided as a technical appendix for commands, source artifacts, and reproducibility checks."
)
