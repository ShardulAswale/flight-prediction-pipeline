"""Predictions Explorer - stable inference UI for binary or multiclass models."""
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import read_parquet_filtered, render_walkthrough_header

st.set_page_config(page_title="Prediction Demo", page_icon="Predict", layout="wide")
st.markdown("# Step 8. Prediction Demo")
render_walkthrough_header("prediction")

PREDICTION_STATE_KEY = "predictions_explorer_last_prediction"
ROUTE_STATE_KEY = "predictions_explorer_selected_route"
_PREDICT_PROBA_SUBPROCESS_CODE = r"""
import json
import sys

import joblib
import pandas as pd

payload = json.load(sys.stdin)
model = joblib.load(payload["model_path"])
X = pd.DataFrame(payload["values"], columns=payload["columns"])
probs = model.predict_proba(X)[0]
print(json.dumps([float(value) for value in probs]))
"""
_SHAP_SUBPROCESS_CODE = r"""
import json
import sys

import joblib
import numpy as np
import pandas as pd
import shap

payload = json.load(sys.stdin)
model = joblib.load(payload["model_path"])
X = pd.DataFrame(payload["values"], columns=payload["columns"])
explainer = shap.TreeExplainer(model)
shap_vals = explainer.shap_values(X)
pred_index = payload.get("pred_index")
if isinstance(shap_vals, list):
    idx = 0 if pred_index is None else int(pred_index)
    sv = shap_vals[idx][0]
elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
    idx = 0 if pred_index is None else int(pred_index)
    sv = shap_vals[0, :, idx]
else:
    sv = shap_vals[0]
print(json.dumps([float(value) for value in sv]))
"""

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


@st.cache_data(show_spinner=False)
def load_airport_city_lookup(raw_bts_dir: str) -> dict[str, str]:
    raw_dir = Path(raw_bts_dir)
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        return {}

    lookup: dict[str, str] = {}
    use_cols = ["Origin", "OriginCityName", "Dest", "DestCityName"]
    for chunk in pd.read_csv(csv_files[0], usecols=use_cols, chunksize=200_000):
        origin_rows = chunk[["Origin", "OriginCityName"]].dropna().drop_duplicates()
        dest_rows = chunk[["Dest", "DestCityName"]].dropna().drop_duplicates()
        lookup.update(dict(zip(origin_rows["Origin"].astype(str), origin_rows["OriginCityName"].astype(str))))
        lookup.update(dict(zip(dest_rows["Dest"].astype(str), dest_rows["DestCityName"].astype(str))))
    return lookup


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
airport_city_lookup = load_airport_city_lookup("data/raw/bts")
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


def airport_lookup_key(icao_code: object) -> str:
    code = str(icao_code).strip().upper()
    special_cases = {
        "PANC": "ANC",
        "PHNL": "HNL",
        "PHOG": "OGG",
        "PHKO": "KOA",
        "PHLI": "LIH",
        "TJSJ": "SJU",
        "TIST": "STT",
    }
    if code in special_cases:
        return special_cases[code]
    if len(code) == 4 and code.startswith("K"):
        return code[1:]
    return code


def airport_display_name(icao_code: object) -> str:
    code = str(icao_code).strip().upper()
    city_name = airport_city_lookup.get(airport_lookup_key(code))
    return city_name if city_name else code


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


def is_xgboost_model() -> bool:
    return selected_model_name == "xgboost" or type(model).__module__.startswith("xgboost")


def run_json_subprocess(code: str, payload: dict) -> list[float]:
    completed = subprocess.run(
        [sys.executable, "-c", code],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("Subprocess returned no JSON output.")
    return json.loads(output_lines[-1])


def predict_probabilities(X: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    if is_xgboost_model():
        # XGBoost native teardown can terminate the Streamlit script thread on Windows.
        # Running inference in a short-lived main-thread subprocess keeps the UI process alive.
        values = run_json_subprocess(
            _PREDICT_PROBA_SUBPROCESS_CODE,
            {
                "model_path": str(model_path),
                "columns": feature_cols,
                "values": X.to_numpy().tolist(),
            },
        )
        return np.asarray(values, dtype=float)
    return model.predict_proba(X)[0]


def compute_shap_values(X: pd.DataFrame, pred_index: int | None = None) -> np.ndarray:
    if is_xgboost_model():
        values = run_json_subprocess(
            _SHAP_SUBPROCESS_CODE,
            {
                "model_path": str(model_path),
                "columns": feature_cols,
                "values": X.to_numpy().tolist(),
                "pred_index": pred_index,
            },
        )
        return np.asarray(values, dtype=float)

    explainer = load_shap_explainer(model)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        idx = 0 if pred_index is None else int(pred_index)
        return shap_vals[idx][0]
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        idx = 0 if pred_index is None else int(pred_index)
        return shap_vals[0, :, idx]
    return shap_vals[0]


def render_prediction(df_input: pd.DataFrame, title: str = "Predicted label"):
    X = preprocess_features(df_input)
    probs = predict_probabilities(X)
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
    sv = compute_shap_values(X, pred_index)

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


def selected_flight_metadata(df_input: pd.DataFrame) -> dict:
    metadata_cols = [
        "flight_key",
        "scheduled_dep",
        "origin",
        "destination",
        "label",
        "label_binary",
        "disruption_subtype",
        "delay_minutes",
    ]
    if df_input.empty:
        return {}
    row = df_input.iloc[0]
    return {col: row[col] for col in metadata_cols if col in df_input.columns}


def actual_target_metadata(df_input: pd.DataFrame) -> dict:
    actual_target_col = "label_binary" if is_binary_disruption_model and "label_binary" in df_input.columns else "label"
    metadata = {"actual_target_col": actual_target_col}
    if actual_target_col in df_input.columns:
        metadata["actual_label"] = str(df_input[actual_target_col].iloc[0])
    if "label" in df_input.columns:
        metadata["original_label"] = str(df_input["label"].iloc[0])
    if "label_binary" in df_input.columns:
        metadata["binary_label"] = str(df_input["label_binary"].iloc[0])
    return metadata


def render_saved_prediction(prediction: dict, title: str = "Predicted label"):
    label = prediction["label"]
    probs = prediction["probs"]
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


def render_saved_prediction_context(prediction: dict):
    actual_label = prediction.get("actual_label")
    if actual_label is not None:
        if actual_label == str(prediction["label"]):
            st.info(f"Actual target label: **{actual_label}**. Prediction matches the saved target.")
        else:
            st.warning(f"Actual target label: **{actual_label}**. Prediction differs from the saved target.")

    original_label = prediction.get("original_label")
    binary_label = prediction.get("binary_label")
    if original_label is not None and binary_label is not None:
        st.caption(f"Original label: {original_label} | Binary target: {binary_label}")


st.subheader("Select Route")
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
    route_df["route_label"] = route_df.apply(
        lambda row: (
            f"{airport_display_name(row['origin'])} ({row['origin']})"
            f" -> {airport_display_name(row['destination'])} ({row['destination']})"
        ),
        axis=1,
    )
    route_df["route_value"] = route_df["origin"] + " -> " + route_df["destination"]
    route_option_map = dict(zip(route_df["route_label"], route_df["route_value"]))

    selected_route_label = st.selectbox("Route", route_df["route_label"].tolist(), key="predictions_route_select")
    selected_route = route_option_map[selected_route_label]
    selected_origin, selected_destination = selected_route.split(" -> ", 1)

    route_matches = df_index.loc[
        df_index["origin"].astype(str).eq(selected_origin)
        & df_index["destination"].astype(str).eq(selected_destination)
    ].copy()
else:
    st.info("Route columns are unavailable in the flight index. Showing flights without route filtering.")
    route_matches = df_index.copy()

sort_cols = [c for c in ["scheduled_dep", "flight_key"] if c in route_matches.columns]
if sort_cols:
    route_matches = route_matches.sort_values(sort_cols)

if route_matches.empty:
    st.warning("No flights found for the selected route.")
    st.stop()

if st.session_state.get(ROUTE_STATE_KEY) != selected_route:
    st.session_state[ROUTE_STATE_KEY] = selected_route
    st.session_state.pop(PREDICTION_STATE_KEY, None)

st.info(f"Available saved flights on this route: {len(route_matches):,}. Using the latest saved record for prediction.")

selected_idx = int(route_matches.index[-1])
selected_index_row = df_index.loc[selected_idx]
selected_row = fetch_selected_row(selected_index_row)

st.caption("Latest saved flight record selected automatically for this route.")

action_cols = st.columns([1, 1, 4])
predict_clicked = action_cols[0].button("Predict Selected Flight", type="primary")
clear_clicked = action_cols[1].button("Clear Prediction")

if clear_clicked:
    st.session_state.pop(PREDICTION_STATE_KEY, None)

if predict_clicked:
    try:
        label, probs, X_selected, pred_idx = render_prediction(selected_row)
        prediction_state = {
            "label": label,
            "probs": probs,
            "X_selected": X_selected,
            "pred_idx": pred_idx,
            "selected_model_name": selected_model_name,
            "selected_flight": selected_flight_metadata(selected_row),
            **actual_target_metadata(selected_row),
        }
        st.session_state[PREDICTION_STATE_KEY] = prediction_state
        render_saved_prediction_context(prediction_state)
    except Exception as exc:
        st.error("Prediction failed, but the app did not crash.")
        st.exception(exc)

saved_prediction = st.session_state.get(PREDICTION_STATE_KEY)
if saved_prediction is not None and not predict_clicked:
    saved_flight = saved_prediction.get("selected_flight", {})
    saved_flight_key = saved_flight.get("flight_key")
    saved_model = saved_prediction.get("selected_model_name")
    if saved_flight_key is not None or saved_model is not None:
        st.caption(
            "Latest saved prediction"
            f"{f' | model: {saved_model}' if saved_model is not None else ''}"
            f"{f' | flight_key: {saved_flight_key}' if saved_flight_key is not None else ''}"
        )
    render_saved_prediction(saved_prediction)
    render_saved_prediction_context(saved_prediction)

if show_shap and saved_prediction is not None:
    saved_model_name = saved_prediction.get("selected_model_name")
    if saved_model_name != selected_model_name:
        st.info("Run a new prediction with the currently selected model before generating SHAP.")
    elif st.button("Generate SHAP explanation for selected flight"):
        try:
            render_shap(saved_prediction["X_selected"], saved_prediction["pred_idx"])
        except Exception as exc:
            st.warning("SHAP explanation failed, but prediction remains available.")
            st.exception(exc)
