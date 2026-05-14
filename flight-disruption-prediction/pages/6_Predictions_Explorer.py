"""Predictions Explorer - stable inference UI for binary or multiclass models."""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import read_parquet_filtered

st.set_page_config(page_title="Predictions Explorer", page_icon="P", layout="wide")
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

with open(comparison_path, encoding="utf-8") as f:
    comparison = json.load(f)

available_models = [item.get("model") for item in comparison.get("ranking", []) if item.get("model")]
default_model_name = comparison.get("best_model", available_models[0] if available_models else "random_forest")
selected_model_name = st.sidebar.selectbox(
    "Prediction model",
    available_models or [default_model_name],
    index=(available_models.index(default_model_name) if default_model_name in available_models else 0),
)

model_path = Path(f"models/{selected_model_name}.pkl")
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
    with open(feature_file, encoding="utf-8") as f:
        loaded_features = json.load(f)
    return loaded_model, loaded_imputer, loaded_scaler, loaded_encoder, loaded_features


@st.cache_resource(show_spinner=False)
def load_shap_explainer(_model_obj):
    import shap

    return shap.TreeExplainer(_model_obj)


@st.cache_data(show_spinner=False)
def load_flight_index(path: str) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        parquet_cols = pq.ParquetFile(path).schema.names
    except Exception:
        parquet_cols = pd.read_parquet(path).columns.tolist()

    columns = [
        "flight_key",
        "scheduled_dep",
        "origin",
        "destination",
        "label",
        "label_binary",
        "disruption_subtype",
        "delay_minutes",
    ]
    use_cols = [c for c in columns if c in parquet_cols]
    return pd.read_parquet(path, columns=use_cols)


model, imputer, scaler, label_encoder, feature_cols = load_prediction_artifacts(
    str(model_path),
    str(imputer_path),
    str(scaler_path),
    str(encoder_path),
    str(feature_path),
)

model_classes = [str(c) for c in label_encoder.classes_]
is_binary_disruption_model = set(model_classes) == {"Disrupted", "Normal"}

ml_path = Path("data/processed/ml_dataset.parquet")
if not ml_path.exists():
    st.warning("Final dataset not found. Run the pipeline first.")
    st.stop()

df_index = load_flight_index(str(ml_path))
label_filter_column = "label_binary" if is_binary_disruption_model and "label_binary" in df_index.columns else "label"

st.caption(
    f"Loaded `{selected_model_name}` with {len(feature_cols)} features. "
    f"Target classes: {model_classes}. Probabilities are raw model probabilities, not calibrated probabilities."
)
st.info("Minimal mode is enabled for stability: selection, prediction, and probability table only.")


def fetch_selected_row(index_row: pd.Series) -> pd.DataFrame:
    full_columns = list(
        dict.fromkeys(
            [
                "flight_key",
                "scheduled_dep",
                "origin",
                "destination",
                "label",
                "label_binary",
                "disruption_subtype",
                "delay_minutes",
                *feature_cols,
            ]
        )
    )
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
    if "Disrupted" in class_to_prob:
        return float(class_to_prob.get("Disrupted", 0.0))
    return float(class_to_prob.get("Late", 0.0) + class_to_prob.get("Cancelled", 0.0))


def render_prediction(df_input: pd.DataFrame, title: str = "Predicted label"):
    X = preprocess_features(df_input)
    probs = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    if probs is not None:
        pred_idx = int(np.argmax(probs))
    else:
        pred = model.predict(X)
        pred_arr = np.asarray(pred)
        if pred_arr.ndim > 1:
            pred_idx = int(np.argmax(pred_arr[0]))
        else:
            pred_idx = int(np.ravel(pred_arr)[0])
    label = label_encoder.inverse_transform([pred_idx])[0]

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
            .head(15)
            .sort_values("abs_shap")
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = np.where(shap_df["shap_value"] >= 0, "#d97706", "#2563eb")
        ax.barh(shap_df["feature"], shap_df["shap_value"], color=colors)
        ax.set_title("Top Feature Contributions (SHAP)")
        ax.set_xlabel("SHAP value")
        plt.tight_layout()
        st.pyplot(fig, width="stretch")
        st.dataframe(shap_df.sort_values("abs_shap", ascending=False), width="stretch", hide_index=True)
    except Exception as exc:
        st.info(f"SHAP explanation unavailable: {exc}")


st.subheader("Select Existing Flight")
show_shap = st.sidebar.checkbox("Enable SHAP tools", value=False)

st.write(f"Available flights: **{len(df_index):,}**")

if {"origin", "destination"}.issubset(df_index.columns):
    route_df = (
        df_index[["origin", "destination"]]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values(["origin", "destination"])
        .reset_index(drop=True)
    )
    route_df["route"] = route_df["origin"] + " -> " + route_df["destination"]

    selected_route = st.selectbox("Route", route_df["route"].tolist(), key="predictions_route_select")
    selected_origin, selected_destination = selected_route.split(" -> ", 1)

    route_matches = df_index.loc[
        df_index["origin"].astype(str).eq(selected_origin)
        & df_index["destination"].astype(str).eq(selected_destination)
    ].copy()
    selector_key = f"predictions_flight_select_{selected_origin}_{selected_destination}"
else:
    st.info("Route columns are unavailable in the flight index. Showing flights without route filtering.")
    route_matches = df_index.copy()
    selector_key = "predictions_flight_select_all"

sort_cols = [c for c in ["scheduled_dep", "flight_key"] if c in route_matches.columns]
if sort_cols:
    route_matches = route_matches.sort_values(sort_cols)
route_matches = route_matches.head(1000)

if route_matches.empty:
    st.warning("No flights found for the selected route.")
    st.stop()

if len(route_matches) == 1000:
    st.info("Showing first 1,000 flights for the selected route.")
else:
    st.info(f"Showing {len(route_matches):,} flights for the selected route.")

flight_display_cols = [c for c in ["flight_key", "scheduled_dep", label_filter_column, "disruption_subtype", "delay_minutes"] if c in route_matches.columns]
route_matches["scheduled_dep_display"] = pd.to_datetime(
    route_matches["scheduled_dep"], errors="coerce", utc=True
).dt.strftime("%Y-%m-%d %H:%M")
route_matches["_flight_option"] = route_matches.apply(
    lambda row: " | ".join(
        [
            str(row.get("flight_key", "")),
            str(row.get("scheduled_dep_display", "")),
            *(str(row.get(col, "")) for col in flight_display_cols if col not in {"flight_key", "scheduled_dep"}),
        ]
    ),
    axis=1,
)

flight_option_map = dict(zip(route_matches["_flight_option"], route_matches.index))
selected_display = st.selectbox(
    "Flight",
    options=list(flight_option_map.keys()),
    index=0,
    key=selector_key,
)
selected_idx = int(flight_option_map[selected_display])
selected_index_row = df_index.loc[selected_idx]
selected_row = fetch_selected_row(selected_index_row)

selected_show_cols = [c for c in ["flight_key", "scheduled_dep", "origin", "destination", "label", "label_binary", "disruption_subtype", "delay_minutes"] if c in selected_row.columns]
st.markdown("#### Selected Flight")
st.dataframe(selected_row[selected_show_cols], width="stretch", hide_index=True)

if st.button("Predict Selected Flight", type="primary"):
    label, probs, X_selected, pred_idx = render_prediction(selected_row)
    actual_target_col = "label_binary" if is_binary_disruption_model and "label_binary" in selected_row.columns else "label"
    if actual_target_col in selected_row.columns:
        actual_label = str(selected_row[actual_target_col].iloc[0])
        if actual_label == str(label):
            st.info(f"Actual target label: **{actual_label}**. Prediction matches the saved target.")
        else:
            st.warning(f"Actual target label: **{actual_label}**. Prediction differs from the saved target.")
    if "label" in selected_row.columns and "label_binary" in selected_row.columns:
        st.caption(f"Original label: {selected_row['label'].iloc[0]} | Binary target: {selected_row['label_binary'].iloc[0]}")
    if show_shap and st.button("Generate SHAP explanation for selected flight"):
        render_shap(X_selected, pred_idx)
