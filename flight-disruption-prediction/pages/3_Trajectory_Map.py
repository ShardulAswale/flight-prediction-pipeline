"""Trajectory map page backed by compact trajectory sketches."""
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import get_parquet_metadata, read_parquet_limited, render_walkthrough_header

st.set_page_config(page_title="ADS-B Trajectory Evidence", page_icon="Map", layout="wide")
st.markdown("# Step 4. ADS-B Trajectory Evidence")
render_walkthrough_header("trajectory")

sketch_path = Path("data/processed/trajectory_sketches.parquet")
full_traj_path = Path("data/processed/trajectories.parquet")
source_path = sketch_path if sketch_path.exists() else full_traj_path

if not source_path.exists():
    st.warning("No trajectory data available. Run `python main.py --stage features` first.")
    st.stop()

meta = get_parquet_metadata(source_path)
required_cols = [
    c for c in [
        "trajectory_id", "flight_key", "icao24", "callsign", "timestamp",
        "latitude", "longitude", "baro_altitude", "geo_altitude", "velocity",
    ] if c in meta.get("columns", [])
]
preview = read_parquet_limited(source_path, columns=required_cols, limit=250000)

if preview.empty or "trajectory_id" not in preview.columns:
    st.warning("Trajectory file is missing required columns.")
    st.stop()

st.info(f"Using {'trajectory_sketches.parquet' if source_path == sketch_path else source_path.name} for responsive visualisation.")

id_options = preview[[c for c in ["trajectory_id", "flight_key", "callsign", "icao24"] if c in preview.columns]].drop_duplicates().copy()
display_cols = [c for c in ["trajectory_id", "flight_key", "callsign", "icao24"] if c in id_options.columns]
id_options["_display"] = id_options[display_cols].astype(str).agg(" | ".join, axis=1)
selected_display = st.sidebar.selectbox("Trajectory", id_options["_display"].tolist())
selected_id = id_options.loc[id_options["_display"] == selected_display, "trajectory_id"].iloc[0]
show_profiles = st.sidebar.checkbox("Show Altitude and Speed Profiles", value=True)

traj_data = preview[preview["trajectory_id"].astype(str) == str(selected_id)].copy().sort_values("timestamp")

if traj_data.empty:
    st.info("No points available for selected trajectory in the current preview.")
    st.stop()

lat_col = "latitude" if "latitude" in traj_data.columns else None
lon_col = "longitude" if "longitude" in traj_data.columns else None
alt_col = "geo_altitude" if "geo_altitude" in traj_data.columns else ("baro_altitude" if "baro_altitude" in traj_data.columns else None)
speed_col = "velocity" if "velocity" in traj_data.columns else None

if not lat_col or not lon_col:
    st.warning("Trajectory points are missing latitude/longitude.")
    st.stop()

coords_df = traj_data[[lat_col, lon_col]].dropna()
if coords_df.empty:
    st.warning("Selected trajectory has no valid coordinates.")
    st.stop()

try:
    from streamlit_folium import st_folium
    import folium

    center_lat = float(coords_df[lat_col].mean())
    center_lon = float(coords_df[lon_col].mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="cartodbpositron")

    coords = coords_df.values.tolist()
    folium.PolyLine(coords, color="#2a9d8f", weight=3, opacity=0.9).add_to(fmap)
    if coords:
        folium.Marker(coords[0], tooltip="Start").add_to(fmap)
        folium.Marker(coords[-1], tooltip="End").add_to(fmap)

    st_folium(
        fmap,
        width=1400,
        height=520,
        key=f"trajectory_map_{selected_id}",
        returned_objects=[],
    )
except Exception as exc:
    st.info(f"Map rendering unavailable ({exc}). Showing a geographic scatter plot instead.")
    fig_map = px.scatter_geo(traj_data, lat=lat_col, lon=lon_col, title=f"Trajectory {selected_id}")
    st.plotly_chart(fig_map, width="stretch")

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Points", f"{len(traj_data):,}")
c2.metric("ICAO24", str(traj_data.get("icao24", pd.Series(["N/A"])).iloc[0]))
c3.metric("Callsign", str(traj_data.get("callsign", pd.Series(["N/A"])).iloc[0]))
if "timestamp" in traj_data.columns:
    duration = float(traj_data["timestamp"].max() - traj_data["timestamp"].min())
    c4.metric("Duration (s)", f"{duration:,.0f}")
if alt_col and traj_data[alt_col].notna().any():
    c5.metric("Max Altitude", f"{float(traj_data[alt_col].max()):,.0f}")

traj_data["point_idx"] = range(len(traj_data))

if show_profiles:
    st.subheader("Altitude and Speed Profiles")
    profile_cols = st.columns(2)

    with profile_cols[0]:
        if alt_col and traj_data[alt_col].notna().any():
            alt_fig = px.line(
                traj_data,
                x="point_idx",
                y=alt_col,
                title=f"Altitude Profile - {selected_id}",
                labels={"point_idx": "Point Index", alt_col: "Altitude"},
            )
            st.plotly_chart(alt_fig, width="stretch")
        else:
            st.info("Altitude data is unavailable for this trajectory.")

    with profile_cols[1]:
        if speed_col and traj_data[speed_col].notna().any():
            speed_fig = px.line(
                traj_data,
                x="point_idx",
                y=speed_col,
                title=f"Speed Profile - {selected_id}",
                labels={"point_idx": "Point Index", speed_col: "Speed"},
            )
            st.plotly_chart(speed_fig, width="stretch")
        else:
            st.info("Speed data is unavailable for this trajectory.")
