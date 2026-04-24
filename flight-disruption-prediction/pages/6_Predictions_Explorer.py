"""Predictions Explorer - existing-flight, manual, and batch probability predictions."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import read_parquet_filtered

st.set_page_config(page_title="Predictions Explorer", page_icon="??", layout="wide")
st.markdown("# Predictions Explorer")

try:
    import joblib
except Exception as exc:
    st.error(f"joblib unavailable: {exc}")
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
missing = [str(p) for p in required if not p.exists()]
if missing:
    st.warning("Model artifacts missing. Re-run training stage to persist artifacts.")
    st.code("\n".join(missing))
    st.stop()


@st.cache_resource(show_spinner=False)
def load_prediction_artifacts(model_file: str, imputer_file: str, scaler_file: str, encoder_file: str, feature_file: str):
    loaded_model = joblib.load(model_file)
    loaded_imputer = joblib.load(imputer_file)
    loaded_scaler = joblib.load(scaler_file)
    loaded_encoder = joblib.load(encoder_file)
    with open(feature_file) as f:
        loaded_features = json.load(f)
    return loaded_model, loaded_imputer, loaded_scaler, loaded_encoder, loaded_features


@st.cache_resource(show_spinner=False)
def load_shap_explainer(model_obj):
    import shap

    return shap.TreeExplainer(model_obj)


@st.cache_data(show_spinner=False)
def load_flight_index(path: str) -> pd.DataFrame:
    columns = ["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes"]
    return pd.read_parquet(path, columns=columns)


model, imputer, scaler, label_encoder, feature_cols = load_prediction_artifacts(
    str(model_path),
    str(imputer_path),
    str(scaler_path),
    str(encoder_path),
    str(feature_path),
)

ml_path = Path("data/processed/ml_dataset.parquet")
if not ml_path.exists():
    st.warning("Final dataset not found. Run the pipeline first.")
    st.stop()

index_cols = ["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes"]

df_index = load_flight_index(str(ml_path))

st.caption(
    f"Loaded `{best_model_name}` with {len(feature_cols)} features. "
    "Probabilities are raw model probabilities, not calibrated probabilities."
)
st.info("Minimal mode is enabled for stability: selection, prediction, and probability table only.")


def fetch_selected_row(index_row: pd.Series) -> pd.DataFrame:
    full_columns = list(dict.fromkeys(index_cols + feature_cols))
    filters = {}
    if "flight_key" in index_row.index and pd.notna(index_row["flight_key"]):
        filters["flight_key"] = index_row["flight_key"]
    row = read_parquet_filtered(str(ml_path), columns=full_columns, filters=filters, limit=10)
    if row.empty:
        return pd.DataFrame([index_row.to_dict()])

    if "scheduled_dep" in row.columns and pd.notna(index_row.get("scheduled_dep")):
        target_ts = pd.to_datetime(index_row["scheduled_dep"], errors="coerce", utc=True)
        row_ts = pd.to_datetime(row["scheduled_dep"], errors="coerce", utc=True)
        exact = row.loc[row_ts.eq(target_ts)]
        if not exact.empty:
            row = exact
    return row.head(1)


def preprocess_features(df_input: pd.DataFrame) -> pd.DataFrame:
    work = df_input.copy()
    for col in feature_cols:
        if col not in work.columns:
            work[col] = np.nan
    work = work[feature_cols]
    work = pd.DataFrame(imputer.transform(work), columns=feature_cols, index=work.index)
    work = pd.DataFrame(scaler.transform(work), columns=feature_cols, index=work.index)
    return work


def prediction_table(probs: np.ndarray) -> pd.DataFrame:
    return (
        pd.DataFrame({"class": label_encoder.classes_, "probability": probs, "probability_pct": probs * 100})
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )


def disruption_probability(probs: np.ndarray) -> float:
    class_to_prob = dict(zip(label_encoder.classes_, probs))
    return float(class_to_prob.get("Late", 0.0) + class_to_prob.get("Cancelled", 0.0))


def render_prediction(df_input: pd.DataFrame, title: str = "Predicted label"):
    X = preprocess_features(df_input)
    pred = model.predict(X)
    pred_idx = int(np.ravel(pred)[0])
    label = label_encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None

    st.success(f"{title}: **{label}**")
    if probs is not None:
        risk = disruption_probability(probs)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Class", str(label))
        c2.metric("Disruption Risk", f"{risk * 100:.1f}%")
        c3.metric("Model Confidence", f"{float(np.max(probs)) * 100:.1f}%")

        proba_df = prediction_table(probs)
        st.dataframe(
            proba_df.assign(probability_pct=proba_df["probability_pct"].map(lambda x: f"{x:.2f}%"))[["class", "probability_pct"]],
            width="stretch",
            hide_index=True,
        )
    return label, probs, X, pred_idx


def render_shap(X: pd.DataFrame, pred_index: int | None = None):
    try:
        explainer = load_shap_explainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            idx = 0 if pred_index is None else int(pred_index)
            sv = shap_vals[idx][0]
        elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            idx = 0 if pred_index is None else int(pred_index)
            sv = shap_vals[0, :, idx]
        else:
            sv = shap_vals[0]

        shap_df = (
            pd.DataFrame({"feature": feature_cols, "shap_value": sv})
            .assign(abs_shap=lambda d: d["shap_value"].abs())
            .sort_values("abs_shap", ascending=False)
            .head(20)
            .sort_values("abs_shap")
        )
        shap_fig = px.bar(shap_df, x="shap_value", y="feature", orientation="h", title="Top Feature Contributions (SHAP)")
        st.plotly_chart(shap_fig, width="stretch")
    except Exception as exc:
        st.info(f"SHAP explanation unavailable: {exc}")


st.subheader("Select Existing Flight")
show_shap = st.sidebar.checkbox("Show SHAP explanation", value=False)

flight_cols = [col for col in index_cols if col in df_index.columns]
st.write(f"Available flights: **{len(df_index):,}**")

filter_cols = st.columns(4)
label_filter = "All"
origin_filter = "All"
dest_filter = "All"
text_filter = ""

if "label" in df_index.columns:
    label_values = ["All"] + sorted(df_index["label"].dropna().astype(str).unique().tolist())
    label_filter = filter_cols[0].selectbox("Label", label_values)
if "origin" in df_index.columns:
    origin_values = ["All"] + sorted(df_index["origin"].dropna().astype(str).unique().tolist())
    origin_filter = filter_cols[1].selectbox("Origin", origin_values)
if "destination" in df_index.columns:
    dest_values = ["All"] + sorted(df_index["destination"].dropna().astype(str).unique().tolist())
    dest_filter = filter_cols[2].selectbox("Destination", dest_values)
text_filter = filter_cols[3].text_input("Search flight key", "")

candidate_mask = pd.Series(True, index=df_index.index)
if label_filter != "All" and "label" in df_index.columns:
    candidate_mask &= df_index["label"].astype(str).eq(label_filter)
if origin_filter != "All" and "origin" in df_index.columns:
    candidate_mask &= df_index["origin"].astype(str).eq(origin_filter)
if dest_filter != "All" and "destination" in df_index.columns:
    candidate_mask &= df_index["destination"].astype(str).eq(dest_filter)
if text_filter.strip() and "flight_key" in df_index.columns:
    candidate_mask &= df_index["flight_key"].astype(str).str.contains(text_filter.strip(), case=False, na=False)

candidate_idx = df_index.index[candidate_mask]
if len(candidate_idx) == 0:
    st.warning("No flights match the current filters.")
    st.stop()

max_options = 1000
limited_idx = candidate_idx[:max_options]
if len(candidate_idx) > max_options:
    st.info(f"Showing first {max_options:,} of {len(candidate_idx):,} matching flights. Use filters to narrow further.")
else:
    st.info(f"Showing {len(candidate_idx):,} matching flights.")

flight_options = df_index.loc[limited_idx, flight_cols].copy()
if "scheduled_dep" in flight_options.columns:
    flight_options["scheduled_dep"] = pd.to_datetime(flight_options["scheduled_dep"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d %H:%M")
flight_options["_row_id"] = limited_idx
flight_options["_display"] = (
    flight_options[flight_cols]
    .apply(lambda row: " | ".join("" if pd.isna(v) else str(v) for v in row.tolist()), axis=1)
)

selected_display = st.selectbox("Choose a flight", options=flight_options["_display"].tolist(), index=0)
selected_idx = int(flight_options.loc[flight_options["_display"] == selected_display, "_row_id"].iloc[0])
selected_index_row = df_index.loc[selected_idx]
selected_row = fetch_selected_row(selected_index_row)

st.markdown("#### Selected Flight")
st.dataframe(selected_row[flight_cols], width="stretch", hide_index=True)

if st.button("Predict Selected Flight", type="primary"):
    label, probs, X_selected, pred_idx = render_prediction(selected_row)
    if "label" in selected_row.columns:
        actual_label = str(selected_row["label"].iloc[0])
        if actual_label == str(label):
            st.info(f"Actual label: **{actual_label}**. Prediction matches the saved label.")
        else:
            st.warning(f"Actual label: **{actual_label}**. Prediction differs from the saved label.")
    if show_shap:
        render_shap(X_selected, pred_idx)
