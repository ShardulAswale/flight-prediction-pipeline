import folium
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

class TrajectoryVisualiser:
    """Visualiser for ADS-B Flight Trajectories."""
    
    @staticmethod
    def plot_single_trajectory(df: pd.DataFrame, trajectory_id: str, ax=None):
        """Plot a single trajectory on a 2D matplotlib axis (lat/lon)."""
        traj = df[df['trajectory_id'] == trajectory_id]
        if traj.empty:
            print(f"Trajectory {trajectory_id} not found.")
            return
            
        traj = traj.sort_values('timestamp')
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
        ax.plot(traj['longitude'], traj['latitude'], marker='o', markersize=2, label=trajectory_id)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'Flight Trajectory: {trajectory_id}')
        ax.grid(True)
        ax.legend()
        
        if ax is None:
            plt.show()
            
    @staticmethod
    def plot_multiple_trajectories(df: pd.DataFrame, trajectory_ids: List[str]):
        """Plot multiple trajectories on a single matplotlib canvas."""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for traj_id in trajectory_ids:
            TrajectoryVisualiser.plot_single_trajectory(df, traj_id, ax=ax)
            
        ax.set_title('Multiple Flight Trajectories')
        plt.show()
        
    @staticmethod
    def generate_interactive_map(df: pd.DataFrame, trajectory_ids: Optional[List[str]] = None) -> folium.Map:
        """
        Generate an interactive Folium map for given trajectories.
        If trajectory_ids is None, plots a sample of trajectories up to 10.
        """
        if trajectory_ids is None:
            trajectory_ids = df['trajectory_id'].unique()[:10]
            
        # Determine center of the map
        sample_df = df[df['trajectory_id'].isin(trajectory_ids)]
        if sample_df.empty:
            return folium.Map()
            
        center_lat = sample_df['latitude'].mean()
        center_lon = sample_df['longitude'].mean()
        
        # Set map dimensions directly to prevent tile rendering issues in VS Code
        m = folium.Map(location=[center_lat, center_lon], zoom_start=7, width='100%', height=600)
        
        # Color palette for different trajectories
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
        
        for i, traj_id in enumerate(trajectory_ids):
            traj = df[df['trajectory_id'] == traj_id].sort_values('timestamp')
            if traj.empty:
                continue
                
            coords = list(zip(traj['latitude'], traj['longitude']))
            color = colors[i % len(colors)]
            
            folium.PolyLine(
                coords,
                weight=3,
                color=color,
                opacity=0.8,
                tooltip=f"Trajectory: {traj_id} (Points: {len(coords)})"
            ).add_to(m)
            
            # Start and end markers
            if len(coords) > 0:
                folium.Marker(
                    coords[0], 
                    popup=f"Start: {traj_id}", 
                    icon=folium.Icon(color='green', icon='play')
                ).add_to(m)
                
                folium.Marker(
                    coords[-1], 
                    popup=f"End: {traj_id}", 
                    icon=folium.Icon(color='red', icon='stop')
                ).add_to(m)
                
        return m
        
    @staticmethod
    def display_interactive_dashboard(df: pd.DataFrame):
        """
        Produce a Jupyter Notebook interactive dashboard with a dropdown 
        to view specific flight trajectories on a Folium map.
        Requires `ipywidgets` and `IPython.display`.
        """
        import ipywidgets as widgets
        from IPython.display import display

        if df.empty:
            print("No data available.")
            return

        trajectory_ids = sorted(df['trajectory_id'].unique())
        
        def view_map(Trajectory):
            # Generate the new map
            m = TrajectoryVisualiser.generate_interactive_map(df, trajectory_ids=[Trajectory])
            
            # Display trajectory summary stats
            traj_df = df[df['trajectory_id'] == Trajectory]
            points = len(traj_df)
            start_time = pd.to_datetime(traj_df['timestamp'].min(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
            end_time = pd.to_datetime(traj_df['timestamp'].max(), unit='s').strftime('%Y-%m-%d %H:%M:%S')
            print(f"Points: {points} | Start: {start_time} | End: {end_time}")
            
            # Display map
            display(m)
            
        widgets.interact(view_map, Trajectory=trajectory_ids)
