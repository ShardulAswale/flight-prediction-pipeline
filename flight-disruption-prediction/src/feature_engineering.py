import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from pathlib import Path
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extracts features from flight trajectories."""
    
    def __init__(self):
        pass

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on the earth."""
        # radius of earth in km
        R = 6371.0
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        return distance

    def _extract_single_trajectory(self, df_traj: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for a single trajectory group."""
        if len(df_traj) < 2:
            return {}
            
        df_traj = df_traj.sort_values('timestamp')
        
        duration = df_traj['timestamp'].max() - df_traj['timestamp'].min()
        num_points = len(df_traj)
        
        # Safe extraction for altitude and velocity if data might be missing
        altitudes = df_traj['altitude'].dropna()
        velocities = df_traj['velocity'].dropna()
        headings = df_traj['heading'].dropna()
        
        mean_altitude = altitudes.mean() if not altitudes.empty else np.nan
        alt_variance = altitudes.var() if len(altitudes) > 1 else 0.0
        
        mean_velocity = velocities.mean() if not velocities.empty else np.nan
        vel_variance = velocities.var() if len(velocities) > 1 else 0.0
        
        heading_variability = headings.var() if len(headings) > 1 else 0.0
        
        # Distance calculation
        lats = df_traj['latitude'].values
        lons = df_traj['longitude'].values
        dist = 0.0
        for i in range(1, len(lats)):
            dist += self.haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])
            
        # Climb / descent rate based on altitude
        if not altitudes.empty and len(altitudes) >= 2:
            alt_diff = altitudes.iloc[-1] - altitudes.iloc[0]
            climb_rate = alt_diff / duration if duration > 0 else 0
        else:
            climb_rate = np.nan
            
        trajectory_id = df_traj['trajectory_id'].iloc[0]
        
        return {
            'trajectory_id': trajectory_id,
            'duration_seconds': duration,
            'num_points': num_points,
            'mean_altitude': mean_altitude,
            'altitude_variance': alt_variance,
            'mean_velocity': mean_velocity,
            'velocity_variance': vel_variance,
            'heading_variability': heading_variability,
            'distance_km': dist,
            'climb_rate': climb_rate
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for all trajectories.
        
        Args:
            df: DataFrame containing preprocessed trajectory points.
            
        Returns:
            DataFrame with extracted features per trajectory.
        """
        logger.info(f"Extracting features for {df['trajectory_id'].nunique()} trajectories...")
        
        features_list = []
        for traj_id, group in df.groupby('trajectory_id'):
            feats = self._extract_single_trajectory(group)
            if feats:
                features_list.append(feats)
                
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df)} valid feature vectors.")
        return features_df

    def save_features(self, df: pd.DataFrame, output_path: str | Path):
        """Save trajectory features to Parquet."""
        if df.empty:
            logger.warning("No features to save.")
            return
            
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        logger.info(f"Saving features to {output_path}")
        df.to_parquet(output_path, index=False)
