"""Data Explorer - lightweight interactive browser for core datasets."""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import get_parquet_metadata, read_parquet_limited, render_walkthrough_header

st.set_page_config(page_title="Dataset Construction", page_icon="Data", layout="wide")
st.markdown("# Step 3. Dataset Construction")
render_walkthrough_header("dataset")

PATH_MAP = {
    "ADS-B Combined": Path("data/processed/adsb_combined.parquet"),
    "BTS Schedules": Path("data/processed/bts_combined.parquet"),
    "Eurocontrol Schedules": Path("data/processed/eurocontrol_combined.parquet"),
    "Merged": Path("data/processed/ml_dataset_merged.parquet"),
    "ML Dataset": Path("data/processed/ml_dataset.parquet"),
}
DEFAULT_COLUMNS = {
    "ADS-B Combined": ["icao24", "callsign", "timestamp", "latitude", "longitude", "baro_altitude", "velocity", "on_ground"],
    "BTS Schedules": ["flight_key", "scheduled_dep", "origin", "destination", "cancelled", "dep_delay_minutes"],
    "Eurocontrol Schedules": ["flight_key", "scheduled_dep", "origin", "destination", "callsign"],
    "Merged": ["flight_key", "scheduled_dep", "origin", "destination", "label", "trajectory_quality_status"],
    "ML Dataset": ["flight_key", "scheduled_dep", "origin", "destination", "label", "delay_minutes", "trajectory_quality_score"],
}

def load_preview(choice: str, limit: int) -> tuple[pd.DataFrame, dict]:
    data_path = PATH_MAP[choice]
    meta = get_parquet_metadata(data_path)
    if not meta.get("exists"):
        return pd.DataFrame(), meta
    cols = [c for c in DEFAULT_COLUMNS.get(choice, []) if c in meta.get("columns", [])]
    if not cols:
        cols = meta.get("columns", [])[:12]
    df = read_parquet_limited(data_path, columns=cols, limit=limit)
    return df, meta


dataset_choice = st.sidebar.selectbox("Dataset", list(PATH_MAP.keys()))
preview_limit = st.sidebar.slider("Preview rows", min_value=1000, max_value=100000, value=20000, step=1000)
df, meta = load_preview(dataset_choice, preview_limit)

if df.empty:
    st.warning(f"{dataset_choice} is not available yet.")
    st.stop()

st.sidebar.markdown(f"**Total rows:** {meta.get('rows', 0):,}")
st.sidebar.markdown(f"**Preview rows:** {len(df):,}")
st.sidebar.markdown(f"**Columns in file:** {len(meta.get('columns', []))}")

st.info("This page uses a limited preview, not a full parquet load, to keep the app responsive.")

time_col = next((c for c in ["scheduled_dep", "timestamp", "date"] if c in df.columns), None)
airport_col = next((c for c in ["origin", "origin_airport", "airport_code", "ORIGIN"] if c in df.columns), None)
callsign_col = next((c for c in ["callsign", "Callsign"] if c in df.columns), None)

filtered = df.copy()

st.sidebar.subheader("Filters")
if time_col is not None:
    ts = pd.to_datetime(filtered[time_col], errors="coerce", utc=True)
    if ts.notna().any():
        min_dt = ts.min().date()
        max_dt = ts.max().date()
        dr = st.sidebar.date_input("Date Range", value=(min_dt, max_dt), min_value=min_dt, max_value=max_dt)
        if isinstance(dr, tuple) and len(dr) == 2:
            start_dt = pd.Timestamp(dr[0]).tz_localize("UTC")
            end_dt = pd.Timestamp(dr[1]).tz_localize("UTC") + pd.Timedelta(days=1)
            filtered = filtered[(ts >= start_dt) & (ts < end_dt)]

if airport_col is not None:
    airports = sorted(filtered[airport_col].dropna().astype(str).unique().tolist())
    selected_airports = st.sidebar.multiselect("Airports", airports[:200])
    if selected_airports:
        filtered = filtered[filtered[airport_col].astype(str).isin(selected_airports)]

if callsign_col is not None:
    callsign_query = st.sidebar.text_input("Callsign Search")
    if callsign_query.strip():
        filtered = filtered[filtered[callsign_col].astype(str).str.contains(callsign_query.strip(), case=False, na=False)]

st.markdown(f"**Showing {len(filtered):,} preview rows** after filters")

all_cols = filtered.columns.tolist()
default_cols = all_cols[: min(15, len(all_cols))]
selected_cols = st.multiselect("Columns to display", all_cols, default=default_cols)
show_df = filtered[selected_cols] if selected_cols else filtered
st.dataframe(show_df.head(100), width="stretch", height=520, hide_index=True)

st.divider()
stats_col1, stats_col2, stats_col3 = st.columns(3)
stats_col1.metric("Preview Row Count", f"{len(filtered):,}")
stats_col2.metric("Null Cells", f"{int(filtered.isna().sum().sum()):,}")
stats_col3.metric("Column Count", f"{len(filtered.columns)}")

with st.expander("Column Types & Null Percent", expanded=False):
    info_df = pd.DataFrame(
        {
            "column": filtered.columns,
            "dtype": filtered.dtypes.astype(str).values,
            "null_pct": (filtered.isna().mean() * 100).round(2).values,
        }
    ).sort_values("null_pct", ascending=False)
    st.dataframe(info_df, width="stretch", hide_index=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Preview CSV",
    csv_bytes,
    file_name=f"{dataset_choice.lower().replace(' ', '_')}_preview.csv",
    mime="text/csv",
)
