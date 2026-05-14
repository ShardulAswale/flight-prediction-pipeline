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
available_models = [item.get("model") for item in comparison.get("ranking", []) if item.get("model")]
best_model_name = comparison.get("best_model", available_models[0] if available_models else "xgboost")
selected_model_name = st.sidebar.selectbox(
    "Prediction model",
    available_models or [best_model_name],
    index=(available_models.index(best_model_name) if best_model_name in available_models else 0),
)

model_path = PROJECT_ROOT / "models" / f"{selected_model_name}.pkl"
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
def load_artifacts(model_file: str, imputer_file: str, scaler_file: str, encoder_file: str, feature_file: str):
    model = joblib.load(model_file)
    imputer = joblib.load(imputer_file)
    scaler = joblib.load(scaler_file)
    encoder = joblib.load(encoder_file)
    features = json.loads(Path(feature_file).read_text(encoding="utf-8"))
    return model, imputer, scaler, encoder, features


@st.cache_data(show_spinner=False)
def load_index(path: str) -> pd.DataFrame:
    parquet_cols = pd.read_parquet(path, columns=[]).columns.tolist()
    cols = ["flight_key", "scheduled_dep", "origin", "destination", "label", "label_binary", "disruption_subtype", "delay_minutes"]
    use_cols = [c for c in cols if c in parquet_cols]
    return pd.read_parquet(path, columns=use_cols)


model, imputer, scaler, label_encoder, feature_cols = load_artifacts(
    str(model_path),
    str(imputer_path),
    str(scaler_path),
    str(encoder_path),
    str(feature_path),
)
model_classes = [str(c) for c in label_encoder.classes_]
is_binary_disruption_model = set(model_classes) == {"Disrupted", "Normal"}

df_index = load_index(str(ml_path))
label_filter_column = "label_binary" if is_binary_disruption_model and "label_binary" in df_index.columns else "label"

st.write(f"Model: `{selected_model_name}`")
st.write(f"Target classes: `{model_classes}`")
st.write(f"Flights available: `{len(df_index):,}`")

route_df = (
    df_index[["origin", "destination"]]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .sort_values(["origin", "destination"])
    .reset_index(drop=True)
)
route_df["route"] = route_df["origin"] + " -> " + route_df["destination"]

selected_route = st.selectbox("Route", route_df["route"].tolist(), key="standalone_route_select")
selected_origin, selected_destination = selected_route.split(" -> ", 1)

route_matches = df_index.loc[
    df_index["origin"].astype(str).eq(selected_origin)
    & df_index["destination"].astype(str).eq(selected_destination)
].copy()
route_matches = route_matches.sort_values(["scheduled_dep", "flight_key"]).head(500)

if route_matches.empty:
    st.warning("No flights found for the selected route.")
    st.stop()

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
    key=f"standalone_flight_select_{selected_origin}_{selected_destination}",
)
selected_idx = int(flight_option_map[selected_display])
selected_key = df_index.loc[selected_idx, "flight_key"]
selected_dep = df_index.loc[selected_idx, "scheduled_dep"]

full_columns = list(dict.fromkeys(feature_cols + ["flight_key", "scheduled_dep", "origin", "destination", "label", "label_binary", "disruption_subtype", "delay_minutes"]))
selected_row = read_parquet_filtered(str(ml_path), columns=full_columns, filters={"flight_key": selected_key}, limit=10)
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

selected_show_cols = [c for c in ["flight_key", "scheduled_dep", "origin", "destination", "label", "label_binary", "disruption_subtype", "delay_minutes"] if c in selected_row.columns]
st.subheader("Selected Flight")
st.table(selected_row[selected_show_cols])

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
        class_to_prob = dict(zip(label_encoder.classes_, probs))
        if "Disrupted" in class_to_prob:
            disruption_risk = float(class_to_prob["Disrupted"])
        else:
            disruption_risk = float(class_probs.loc[class_probs["class"].isin(["Late", "Cancelled"]), "probability"].sum())
        model_confidence = float(class_probs["probability"].max())

        st.write(f"Disruption risk: {disruption_risk * 100:.1f}%")
        st.write(f"Model confidence: {model_confidence * 100:.1f}%")
        st.subheader("Class probabilities")
        st.table(class_probs[["class", "probability_pct"]])
