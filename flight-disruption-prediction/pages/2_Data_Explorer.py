"""FE3: Data Explorer - interactive browser for all core datasets."""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Data Explorer", page_icon="🔍", layout="wide")
st.markdown("# 🔍 Data Explorer")


def _read_dataset(choice: str) -> pd.DataFrame:
    path_map = {
        "ADS-B Combined": Path("data/processed/adsb_combined.parquet"),
        "BTS Schedules": Path("data/processed/bts_combined.parquet"),
        "Eurocontrol Schedules": Path("data/processed/eurocontrol_combined.parquet"),
        "Merged": Path("data/processed/ml_dataset_merged.parquet"),
        "ML Dataset": Path("data/processed/ml_dataset.parquet"),
    }
    data_path = path_map[choice]
    if not data_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(data_path)


dataset_choice = st.sidebar.selectbox(
    "Dataset",
    ["ADS-B Combined", "BTS Schedules", "Eurocontrol Schedules", "Merged", "ML Dataset"],
)
df = _read_dataset(dataset_choice)

if df.empty:
    st.warning(f"{dataset_choice} is not available yet.")
    st.stop()

if len(df) > 1_000_000:
    st.info("Large dataset detected (>1M rows). Showing a 1,000,000-row sample for responsiveness.")
    df = df.sample(1_000_000, random_state=42)

st.sidebar.markdown(f"**Rows:** {len(df):,}")
st.sidebar.markdown(f"**Columns:** {len(df.columns)}")

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
    selected_airports = st.sidebar.multiselect("Airports", airports[:500])
    if selected_airports:
        filtered = filtered[filtered[airport_col].astype(str).isin(selected_airports)]

if callsign_col is not None:
    callsign_query = st.sidebar.text_input("Callsign Search")
    if callsign_query.strip():
        filtered = filtered[filtered[callsign_col].astype(str).str.contains(callsign_query.strip(), case=False, na=False)]

st.markdown(f"**Showing {len(filtered):,} rows** after filters")

all_cols = filtered.columns.tolist()
default_cols = all_cols[: min(15, len(all_cols))]
selected_cols = st.multiselect("Columns to display", all_cols, default=default_cols)
show_df = filtered[selected_cols] if selected_cols else filtered
st.dataframe(show_df.head(100), use_container_width=True, height=520)

st.divider()
stats_col1, stats_col2, stats_col3 = st.columns(3)
stats_col1.metric("Row Count", f"{len(filtered):,}")
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
    st.dataframe(info_df, use_container_width=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download Filtered CSV", csv_bytes, file_name=f"{dataset_choice.lower().replace(' ', '_')}_filtered.csv", mime="text/csv")
