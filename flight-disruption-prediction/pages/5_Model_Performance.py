"""FE6: Model Performance - ranking, confusion matrices, ROC and PR curves."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import render_walkthrough_header

st.set_page_config(page_title="Model Results", page_icon="Model", layout="wide")
st.markdown("# Step 6. Model Results")
render_walkthrough_header("models")

report_path = Path("models/model_comparison.json")
if not report_path.exists():
    st.warning("No model comparison report found. Run `python main.py --stage train`.")
    st.stop()

with open(report_path) as f:
    comparison = json.load(f)
ranking = pd.DataFrame(comparison.get("ranking", []))
if ranking.empty:
    st.warning("Model ranking is empty.")
    st.stop()

ranking.index = range(1, len(ranking) + 1)
ranking.index.name = "rank"
st.dataframe(ranking, width="stretch")
st.success(f"Best model: {comparison.get('best_model', 'N/A')}")

st.divider()
st.subheader("Confusion Matrices")
cm_files = sorted(Path("outputs").glob("confusion_matrix_*.png"))
if cm_files:
    cols = st.columns(2)
    for i, file_path in enumerate(cm_files):
        with cols[i % 2]:
            st.markdown(f"**{file_path.stem.replace('confusion_matrix_', '')}**")
            st.image(str(file_path), width="stretch")
else:
    st.info("No confusion matrix images found.")

st.divider()
st.subheader("ROC Curves")
roc_path = Path("outputs/roc_curves.png")
if roc_path.exists():
    st.image(str(roc_path), width="stretch")
else:
    st.info("ROC plot not available yet.")

st.divider()
st.subheader("Precision-Recall Curves")
try:
    import joblib
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize

    data_path = Path("data/processed/ml_dataset.parquet")
    feat_path = Path("models/feature_list.json")
    imputer_path = Path("models/imputer.pkl")
    scaler_path = Path("models/scaler.pkl")
    encoder_path = Path("models/label_encoder.pkl")
    model_name = comparison.get("best_model", "random_forest")
    model_path = Path(f"models/{model_name}.pkl")

    if all(p.exists() for p in [data_path, feat_path, imputer_path, scaler_path, encoder_path, model_path]):
        df = pd.read_parquet(data_path)
        if "training_eligible" in df.columns:
            df = df[df["training_eligible"] == True]
        if not df.empty and "label" in df.columns:
            with open(feat_path) as f:
                feature_cols = json.load(f)
            feature_cols = [c for c in feature_cols if c in df.columns]
            X = df[feature_cols].copy()
            y = df["label"].astype(str).copy()

            label_encoder = joblib.load(encoder_path)
            y_enc = label_encoder.transform(y)
            _, X_test, _, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

            imputer = joblib.load(imputer_path)
            scaler = joblib.load(scaler_path)
            model = joblib.load(model_path)

            X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)
                y_bin = label_binarize(y_test, classes=range(len(label_encoder.classes_)))
                fig = go.Figure()
                for i, class_name in enumerate(label_encoder.classes_):
                    precision, recall, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
                    fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=str(class_name)))
                fig.update_layout(title="Precision-Recall Curves", xaxis_title="Recall", yaxis_title="Precision")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("Selected model does not expose predict_proba.")
        else:
            st.info("ML dataset empty or label column missing.")
    else:
        st.info("Missing model artifacts required for PR curve rendering.")
except Exception as e:
    st.info(f"PR curve generation skipped: {e}")

st.divider()
st.subheader("Class Distribution (Actual)")
ml_path = Path("data/processed/ml_dataset.parquet")
if ml_path.exists():
    df = pd.read_parquet(ml_path)
    if "label" in df.columns and not df.empty:
        dist = df["label"].value_counts()
        fig = px.pie(values=dist.values, names=dist.index, title="Actual Label Distribution")
        st.plotly_chart(fig, width="stretch")
