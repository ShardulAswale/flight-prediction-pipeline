"""FE7: Predictions Explorer - single + batch predictions with SHAP visuals."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Predictions Explorer", page_icon="🔮", layout="wide")
st.markdown("# 🔮 Predictions Explorer")

try:
    import joblib
except Exception as e:
    st.error(f"joblib unavailable: {e}")
    st.stop()

comparison_path = Path("models/model_comparison.json")
if not comparison_path.exists():
    st.warning("No trained models found. Run `python main.py --stage train` first.")
    st.stop()

with open(comparison_path) as f:
    comparison = json.load(f)
best_model_name = comparison.get("best_model", "random_forest")
model_path = Path(f"models/{best_model_name}.pkl")
feature_path = Path("models/feature_list.json")
imputer_path = Path("models/imputer.pkl")
scaler_path = Path("models/scaler.pkl")
encoder_path = Path("models/label_encoder.pkl")

required = [model_path, feature_path, imputer_path, scaler_path, encoder_path]
if not all(p.exists() for p in required):
    st.warning("Model artifacts missing. Re-run training stage to persist artifacts.")
    st.stop()

model = joblib.load(model_path)
imputer = joblib.load(imputer_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)
with open(feature_path) as f:
    feature_cols = json.load(f)

ml_path = Path("data/processed/ml_dataset.parquet")
df_ref = pd.read_parquet(ml_path) if ml_path.exists() else pd.DataFrame()


def preprocess_features(df_input: pd.DataFrame) -> pd.DataFrame:
    work = df_input.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    work = work[feature_cols]
    work = pd.DataFrame(imputer.transform(work), columns=feature_cols, index=work.index)
    work = pd.DataFrame(scaler.transform(work), columns=feature_cols, index=work.index)
    return work


st.subheader("Single Prediction")
cols = st.columns(3)
input_row = {}
for i, feat in enumerate(feature_cols):
    with cols[i % 3]:
        if not df_ref.empty and feat in df_ref.columns and df_ref[feat].notna().any():
            min_v = float(df_ref[feat].quantile(0.01))
            max_v = float(df_ref[feat].quantile(0.99))
            default_v = float(df_ref[feat].median())
        else:
            min_v, max_v, default_v = 0.0, 100.0, 0.0
        if min_v >= max_v:
            max_v = min_v + 1.0
        input_row[feat] = st.slider(feat, min_value=min_v, max_value=max_v, value=default_v)

if st.button("Predict", type="primary"):
    X = preprocess_features(pd.DataFrame([input_row]))
    pred = model.predict(X)
    label = label_encoder.inverse_transform(pred)[0]
    st.success(f"Predicted label: **{label}**")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        proba_df = pd.DataFrame({"class": label_encoder.classes_, "probability": probs})
        fig = px.bar(proba_df, x="class", y="probability", title="Prediction Confidence")
        st.plotly_chart(fig, use_container_width=True)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        sv = shap_vals[pred[0]][0] if isinstance(shap_vals, list) else shap_vals[0]
        shap_df = pd.DataFrame({"feature": feature_cols, "shap_value": sv}).sort_values("shap_value", key=np.abs)
        shap_fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h", title="Feature Contributions (SHAP)")
        st.plotly_chart(shap_fig, use_container_width=True)
    except Exception as e:
        st.info(f"SHAP explanation unavailable: {e}")

st.divider()
st.subheader("Similar Flights")
if not df_ref.empty and all(c in df_ref.columns for c in feature_cols) and "label" in df_ref.columns:
    sample = df_ref.dropna(subset=[c for c in feature_cols if c in df_ref.columns]).copy()
    if not sample.empty:
        try:
            X_sample = preprocess_features(sample.head(5000))
            X_query = preprocess_features(pd.DataFrame([input_row]))
            dists = np.linalg.norm(X_sample.values - X_query.values, axis=1)
            sample = sample.iloc[np.argsort(dists)[:5]].copy()
            show_cols = [c for c in ["flight_key", "label"] if c in sample.columns]
            st.dataframe(sample[show_cols + feature_cols[:5]], use_container_width=True)
        except Exception:
            st.info("Could not compute nearest neighbours with current data.")

st.divider()
st.subheader("Batch Prediction (CSV Upload)")
uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
if uploaded is not None:
    try:
        batch_df = pd.read_csv(uploaded)
        X_batch = preprocess_features(batch_df)
        pred_batch = model.predict(X_batch)
        batch_df["prediction"] = label_encoder.inverse_transform(pred_batch)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_batch)
            batch_df["prediction_confidence"] = probs.max(axis=1)

        st.dataframe(batch_df.head(20), use_container_width=True)
        out_csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", out_csv, file_name="batch_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
