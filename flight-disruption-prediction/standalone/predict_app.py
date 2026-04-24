"""Standalone single-page predictor with minimal Streamlit components."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.streamlit_utils import read_parquet_filtered

try:
    import joblib
except Exception as exc:
    st.error(f"joblib unavailable: {exc}")
    st.stop()

st.set_page_config(page_title="Standalone Flight Predictor", page_icon="P", layout="wide")
st.title("Standalone Flight Predictor")
st.caption("Minimal prediction app for stable repeated inference.")

comparison_path = PROJECT_ROOT / "models" / "model_comparison.json"
if not comparison_path.exists():
    st.error("models/model_comparison.json not found. Run training first.")
    st.stop()

comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
best_model_name = comparison.get("best_model", "xgboost")
model_path = PROJECT_ROOT / "models" / f"{best_model_name}.pkl"
feature_path = PROJECT_ROOT / "models" / "feature_list.json"
imputer_path = PROJECT_ROOT / "models" / "imputer.pkl"
scaler_path = PROJECT_ROOT / "models" / "scaler.pkl"
encoder_path = PROJECT_ROOT / "models" / "label_encoder.pkl"
ml_path = PROJECT_ROOT / "data" / "processed" / "ml_dataset.parquet"

required = [model_path, feature_path, imputer_path, scaler_path, encoder_path, ml_path]
missing = [str(p) for p in required if not p.exists()]
if missing:
    st.error("Missing required files:")
    st.code("\n".join(missing))
    st.stop()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(model_path)
    imputer = joblib.load(imputer_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    return model, imputer, scaler, encoder, features


@st.cache_data(show_spinner=False)
def load_index() -> pd.DataFrame:
    cols = ["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes"]
    return pd.read_parquet(ml_path, columns=cols)


model, imputer, scaler, label_encoder, feature_cols = load_artifacts()
df_index = load_index()

st.write(f"Model: `{best_model_name}`")
st.write(f"Flights available: `{len(df_index):,}`")

col1, col2, col3, col4 = st.columns(4)
label_filter = col1.selectbox("Label", ["All"] + sorted(df_index["label"].dropna().astype(str).unique().tolist()))
origin_filter = col2.selectbox("Origin", ["All"] + sorted(df_index["origin"].dropna().astype(str).unique().tolist()))
dest_filter = col3.selectbox("Destination", ["All"] + sorted(df_index["destination"].dropna().astype(str).unique().tolist()))
text_filter = col4.text_input("Search flight key", "")

mask = pd.Series(True, index=df_index.index)
if label_filter != "All":
    mask &= df_index["label"].astype(str).eq(label_filter)
if origin_filter != "All":
    mask &= df_index["origin"].astype(str).eq(origin_filter)
if dest_filter != "All":
    mask &= df_index["destination"].astype(str).eq(dest_filter)
if text_filter.strip():
    mask &= df_index["flight_key"].astype(str).str.contains(text_filter.strip(), case=False, na=False)

matches = df_index.loc[mask].head(500).copy()
if matches.empty:
    st.warning("No matching flights found.")
    st.stop()

matches["scheduled_dep"] = pd.to_datetime(matches["scheduled_dep"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M")
matches["display"] = matches[["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes"]].apply(
    lambda row: " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()), axis=1
)

selected_display = st.selectbox("Choose a flight", matches["display"].tolist())
selected_idx = matches.index[matches["display"] == selected_display][0]
selected_key = df_index.loc[selected_idx, "flight_key"]
selected_dep = df_index.loc[selected_idx, "scheduled_dep"]

full_columns = list(dict.fromkeys(feature_cols + ["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes"]))
selected_row = read_parquet_filtered(ml_path, columns=full_columns, filters={"flight_key": selected_key}, limit=10)
if not selected_row.empty:
    dep_ts = pd.to_datetime(selected_row["scheduled_dep"], errors="coerce", utc=True)
    target_ts = pd.to_datetime(selected_dep, errors="coerce", utc=True)
    exact = selected_row.loc[dep_ts.eq(target_ts)]
    if not exact.empty:
        selected_row = exact.head(1)
    else:
        selected_row = selected_row.head(1)
else:
    st.error("Could not fetch selected row from parquet.")
    st.stop()

show_cols = [c for c in ["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes"] if c in selected_row.columns]
st.subheader("Selected Flight")
st.table(selected_row[show_cols])

if st.button("Predict", type="primary"):
    work = selected_row.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    X = work[feature_cols]
    X = pd.DataFrame(imputer.transform(X), columns=feature_cols, index=X.index)
    X = pd.DataFrame(scaler.transform(X), columns=feature_cols, index=X.index)

    pred = model.predict(X)
    pred_idx = int(np.ravel(pred)[0])
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    st.success(f"Predicted label: {pred_label}")
    if probs is not None:
        class_probs = pd.DataFrame({
            "class": label_encoder.classes_,
            "probability": probs,
            "probability_pct": (probs * 100).round(2),
        }).sort_values("probability", ascending=False)
        disruption_risk = float(class_probs.loc[class_probs["class"].isin(["Late", "Cancelled"]), "probability"].sum())
        model_confidence = float(class_probs["probability"].max())

        st.write(f"Disruption risk: {disruption_risk * 100:.1f}%")
        st.write(f"Model confidence: {model_confidence * 100:.1f}%")
        st.subheader("Class probabilities")
        st.table(class_probs[["class", "probability_pct"]])
