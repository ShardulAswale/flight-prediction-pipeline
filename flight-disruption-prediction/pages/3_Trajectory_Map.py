"""FE4: Trajectory map with altitude/speed profiles and markers."""
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Trajectory Map", page_icon="🗺️", layout="wide")
st.markdown("# 🗺️ Trajectory Map")

traj_path = "data/processed/trajectories.parquet"
if not os.path.exists(traj_path):
    st.warning("No trajectory data available. Run `python main.py --stage features` first.")
    st.stop()

df_traj = pd.read_parquet(traj_path)
if "trajectory_id" not in df_traj.columns or df_traj.empty:
    st.warning("Trajectory file is missing required columns.")
    st.stop()

trajectory_ids = sorted(df_traj["trajectory_id"].astype(str).unique().tolist())
selected_id = st.sidebar.selectbox("Trajectory ID", trajectory_ids)
show_speed = st.sidebar.checkbox("Show Speed Profile", value=True)

traj_data = df_traj[df_traj["trajectory_id"].astype(str) == selected_id].copy()
traj_data = traj_data.sort_values("timestamp")

if traj_data.empty:
    st.info("No points available for selected trajectory.")
    st.stop()

lat_col = "latitude" if "latitude" in traj_data.columns else None
lon_col = "longitude" if "longitude" in traj_data.columns else None
alt_col = "altitude" if "altitude" in traj_data.columns else ("baro_altitude" if "baro_altitude" in traj_data.columns else None)
speed_col = "velocity" if "velocity" in traj_data.columns else None

if not lat_col or not lon_col:
    st.warning("Trajectory points are missing latitude/longitude.")
    st.stop()

try:
    from streamlit_folium import st_folium
    import folium

    center_lat = float(traj_data[lat_col].dropna().mean())
    center_lon = float(traj_data[lon_col].dropna().mean())
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="cartodbpositron")

    coords = traj_data[[lat_col, lon_col]].dropna().values.tolist()
    folium.PolyLine(coords, color="#2a9d8f", weight=3, opacity=0.9).add_to(fmap)
    if coords:
        folium.Marker(coords[0], tooltip="Departure").add_to(fmap)
        folium.Marker(coords[-1], tooltip="Arrival").add_to(fmap)

    st_folium(fmap, width=1400, height=520)
except Exception as e:
    st.info(f"Map rendering unavailable ({e}). Showing lat/lon scatter instead.")
    fig_map = px.scatter_geo(traj_data, lat=lat_col, lon=lon_col, title=f"Trajectory {selected_id}")
    st.plotly_chart(fig_map, use_container_width=True)

st.divider()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Points", f"{len(traj_data):,}")
c2.metric("ICAO24", str(traj_data.get("icao24", pd.Series(["N/A"])).iloc[0]))
c3.metric("Callsign", str(traj_data.get("callsign", pd.Series(["N/A"])).iloc[0]))
if "timestamp" in traj_data.columns:
    duration = float(traj_data["timestamp"].max() - traj_data["timestamp"].min())
    c4.metric("Duration (s)", f"{duration:,.0f}")
if alt_col:
    c5.metric("Max Altitude", f"{float(traj_data[alt_col].max()):,.0f}")

traj_data["point_idx"] = range(len(traj_data))

if alt_col:
    alt_fig = px.line(
        traj_data,
        x="point_idx",
        y=alt_col,
        title=f"Altitude Profile - {selected_id}",
        labels={"point_idx": "Point Index", alt_col: "Altitude"},
    )
    st.plotly_chart(alt_fig, use_container_width=True)

if show_speed and speed_col:
    speed_fig = px.line(
        traj_data,
        x="point_idx",
        y=speed_col,
        title=f"Speed Profile - {selected_id}",
        labels={"point_idx": "Point Index", speed_col: "Speed"},
    )
    st.plotly_chart(speed_fig, use_container_width=True)
