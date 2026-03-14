import logging
import uuid
import pandas as pd
from typing import Optional
from pathlib import Path
from src.utils import ensure_dir

logger = logging.getLogger(__name__)

class TrajectoryBuilder:
    """Build trajectories from raw OpenSky state vectors."""
    
    def __init__(self, max_gap_minutes: int = 15):
        """
        Initialize builder.
        
        Args:
            max_gap_minutes: Maximum time gap (in minutes) to split a single 
                             aircraft's states into multiple trajectories.
        """
        self.max_gap_seconds = max_gap_minutes * 60
        # Required columns for output
        self.output_cols = [
            'trajectory_id', 'icao24', 'timestamp', 'latitude', 
            'longitude', 'altitude', 'velocity', 'heading'
        ]

    def filter_invalid_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out invalid state vectors (missing lat/lon).
        
        Args:
            df: Raw states DataFrame.
            
        Returns:
            Filtered DataFrame.
        """
        initial_len = len(df)
        df_filtered = df.dropna(subset=['latitude', 'longitude'])
        
        # OpenSky uses baro_altitude for aircraft altitude, geo_altitude for physical
        # We will map 'baro_altitude' to 'altitude' and 'true_track' to 'heading'
        if 'baro_altitude' in df_filtered.columns:
            df_filtered['altitude'] = df_filtered['baro_altitude']
        if 'true_track' in df_filtered.columns:
            df_filtered['heading'] = df_filtered['true_track']
            
        # Keep only required columns that exist
        keep_cols = ['icao24', 'timestamp', 'latitude', 'longitude', 'altitude', 'velocity', 'heading']
        existing_cols = [c for c in keep_cols if c in df_filtered.columns]
        df_filtered = df_filtered[existing_cols].copy()
        
        logger.info(f"Filtered {initial_len - len(df_filtered)} invalid points.")
        return df_filtered

    def sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort data frame by time."""
        if 'timestamp' in df.columns:
            return df.sort_values(['icao24', 'timestamp']).reset_index(drop=True)
        return df

    def group_by_aircraft(self, df: pd.DataFrame) -> pd.api.typing.DataFrameGroupBy:
        """Group data frame by aircraft ICAO24."""
        return df.groupby('icao24')

    def build_flight_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment the states into continuous trajectories.
        Creates a unqiue trajectory_id for each continuous flight segment.
        """
        df = self.filter_invalid_points(df)
        df = self.sort_by_time(df)
        
        logger.info("Building flight segments...")
        
        # Calculate time difference between consecutive points for the same aircraft
        df['time_diff'] = df.groupby('icao24')['timestamp'].diff()
        
        # Identify new trajectories based on large time gaps or first observation
        df['new_trajectory'] = (df['time_diff'] > self.max_gap_seconds) | (df['time_diff'].isna())
        
        # Create a cumulative sum to act as a flight index per aircraft
        df['flight_idx'] = df.groupby('icao24')['new_trajectory'].cumsum()
        
        # Combine icao24 and flight_idx to create a unique trajectory_id
        df['trajectory_id'] = df['icao24'] + "_" + df['flight_idx'].astype(str)
        
        # Drop temporary columns
        df.drop(columns=['time_diff', 'new_trajectory', 'flight_idx'], inplace=True)
        
        # Ensure correct output columns
        for col in self.output_cols:
            if col not in df.columns:
                df[col] = None
        
        return df[self.output_cols]

    def save_trajectories(self, df: pd.DataFrame, output_path: str | Path):
        """Save the processed trajectories as Parquet."""
        if df.empty:
            logger.warning("Empty DataFrame, nothing to save.")
            return
            
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        logger.info(f"Saving {len(df)} trajectory points to {output_path}")
        df.to_parquet(output_path, index=False)
