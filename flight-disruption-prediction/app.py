import streamlit as st
import pandas as pd
import yaml
import folium
from streamlit_folium import st_folium
import os
import sys

# Change working directory so we can load src modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.visualisation import TrajectoryVisualiser
from src.utils import load_config

# Set page wide
st.set_page_config(layout="wide", page_title="Flight Trajectory Explorer")
st.title("Interactive Flight Trajectory Explorer")

@st.cache_data
def load_data():
    config = load_config('configs/config.yaml')
    traj_path = f"{config['paths']['processed_data_dir']}/{config['paths']['trajectories_file']}"
    if os.path.exists(traj_path):
        df_traj = pd.read_parquet(traj_path)
        return df_traj, config
    return pd.DataFrame(), config

df_traj, config = load_data()

if df_traj.empty:
    st.warning("No trajectory data found. Please ensure the pipeline has generated processed data.")
else:
    trajectory_ids = sorted(df_traj['trajectory_id'].unique())
    st.sidebar.header("Trajectory Selection")
    selected_traj = st.sidebar.selectbox("Select a Trajectory:", trajectory_ids)
    
    st.subheader(f"Trajectory: {selected_traj}")
    
    # Generate Map
    m = TrajectoryVisualiser.generate_interactive_map(df_traj, trajectory_ids=[selected_traj])
    
    # Display Stats
    traj_data = df_traj[df_traj['trajectory_id'] == selected_traj]
    col1, col2, col3 = st.columns(3)
    col1.metric("Data Points", len(traj_data))
    
    start_time = pd.to_datetime(traj_data['timestamp'].min(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
    end_time = pd.to_datetime(traj_data['timestamp'].max(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
    
    col2.metric("Start Time", start_time)
    col3.metric("End Time", end_time)
    
    # Display map in Streamlit
    st_folium(m, width=1200, height=600)
