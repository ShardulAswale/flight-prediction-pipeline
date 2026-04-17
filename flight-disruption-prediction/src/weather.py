import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class WeatherIntegrator:
    """Advanced METAR weather integration with severity and confidence scoring."""
    
    def __init__(self, tolerance_hours: int = 2, log_dir: str = 'logs'):
        self.tolerance = pd.Timedelta(hours=tolerance_hours)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_severity(self, df: pd.DataFrame) -> pd.Series:
        """
        Computes weather severity score (0 to 3).
        3 = Severe (High wind, extremely low visibility)
        2 = Moderate (Moderate wind, heavy precip, low visibility)
        1 = Mild
        0 = Clear
        """
        conditions = [
            (df['wind_speed'] > 30) | (df['visibility'] < 1.0),
            (df['wind_speed'] > 20) | (df['precipitation'] > 0.5) | (df['visibility'] < 3.0),
            (df['wind_speed'] > 15) | (df['precipitation'] > 0.1) | (df['visibility'] < 5.0)
        ]
        choices = [3, 2, 1]
        
        severity = np.select(conditions, choices, default=0)
        is_na = df[['wind_speed', 'visibility', 'precipitation']].isna().all(axis=1)
        severity = np.where(is_na, np.nan, severity)
        
        return pd.Series(severity, index=df.index)

    def _compute_confidence(self, time_diff: pd.Series) -> pd.Series:
        """
        Assigns a confidence score based on the absolute time gap between flight and METAR.
        """
        diff_mins = time_diff.dt.total_seconds().abs() / 60.0
        
        conditions = [
            diff_mins <= 30.0,
            diff_mins <= 60.0,
            diff_mins > 60.0
        ]
        choices = ['High', 'Medium', 'Low']
        
        return pd.Series(np.select(conditions, choices, default='None'), index=time_diff.index).replace('None', np.nan)

    def add_weather_features(self, df_flights: pd.DataFrame, df_weather: pd.DataFrame, 
                             flight_time_col: str = 'scheduled_dep', 
                             airport_col: str = 'origin') -> pd.DataFrame:
        """
        Merge weather features onto flights using nearest-timestamp + nearest-airport logic.
        T14: Emits weather coverage report to logs/weather_coverage.json
        """
        if df_flights.empty or df_weather.empty:
            logger.warning("Flight or weather dataframe is empty.")
            return df_flights
            
        if flight_time_col not in df_flights.columns or airport_col not in df_flights.columns:
            logger.warning(f"Columns {flight_time_col} or {airport_col} missing in flights.")
            return df_flights
            
        f_df = df_flights.copy()
        w_df = df_weather.copy()
        
        f_df['_merge_time'] = pd.to_datetime(f_df[flight_time_col], utc=True).astype('datetime64[ns, UTC]')
        if 'timestamp' in w_df.columns:
            w_df['_weather_time'] = pd.to_datetime(w_df['timestamp'], utc=True).astype('datetime64[ns, UTC]')
        else:
            logger.warning("Weather dataframe missing 'timestamp' column.")
            return df_flights

        if airport_col not in f_df.columns:
            logger.warning(f"Flight dataframe missing airport column '{airport_col}'.")
            return df_flights
        if 'airport_code' not in w_df.columns:
            logger.warning("Weather dataframe missing 'airport_code' column.")
            return df_flights

        # ── Harmonize airport codes to ICAO ──
        # METAR has a mix: US airports in IATA (ATL), EU airports in ICAO (EDDF).
        # Flight origins are already ICAO (KATL, EDDF). Normalize weather to ICAO.
        from src.normalization import ScheduleNormalizer
        iata_to_icao = ScheduleNormalizer.IATA_TO_ICAO  # e.g. ATL -> KATL

        # Flights are already ICAO-coded
        raw_flight_airports = f_df[airport_col].astype(str).str.upper().str.strip()
        f_df['_merge_airport'] = raw_flight_airports

        # Weather: map IATA->ICAO for codes that are IATA, keep ICAO codes as-is
        raw_weather_airports = w_df['airport_code'].astype(str).str.upper().str.strip()
        w_df['_merge_airport'] = raw_weather_airports.map(iata_to_icao).fillna(raw_weather_airports)
            
        # Standardize empty numerics
        num_cols = ['wind_speed', 'visibility', 'temperature', 'precipitation']
        for col in num_cols:
            if col in w_df.columns:
                w_df[col] = pd.to_numeric(w_df[col], errors='coerce')
            
        f_df = f_df.sort_values(['_merge_time', '_merge_airport'])
        w_df = w_df.sort_values(['_weather_time', '_merge_airport'])
        
        merged = pd.merge_asof(
            f_df, 
            w_df,
            left_on='_merge_time',
            right_on='_weather_time',
            by='_merge_airport',
            direction='nearest',
            tolerance=self.tolerance
        )
        
        # Calculate derived metrics
        if '_weather_time' in merged.columns and '_merge_time' in merged.columns:
            time_diff = merged['_weather_time'] - merged['_merge_time']
            merged['weather_confidence'] = self._compute_confidence(time_diff)
            
            for col in num_cols:
                if col not in merged.columns:
                    merged[col] = np.nan
            merged['weather_severity'] = self._compute_severity(merged)
        
        # Drop temporary cols
        drop_cols = ['_merge_time', '_weather_time', '_merge_airport', 'airport_code', 'timestamp']
        merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True)
        
        missing_weather = merged['weather_severity'].isna().sum() if 'weather_severity' in merged.columns else len(merged)
        if missing_weather > 0:
            logger.info(f"Could not find matching weather data for {missing_weather} flights.")
        
        # T14: Save weather coverage report
        try:
            weather_cols = ['wind_speed', 'visibility', 'temperature', 'precipitation', 'weather_severity']
            coverage = {}
            total = len(merged)
            for col in weather_cols:
                if col in merged.columns:
                    null_count = int(merged[col].isna().sum())
                    coverage[col] = {
                        'null_count': null_count,
                        'null_pct': round(null_count / total * 100, 2) if total > 0 else 0,
                        'coverage_pct': round((total - null_count) / total * 100, 2) if total > 0 else 0
                    }
                else:
                    coverage[col] = {'null_count': total, 'null_pct': 100.0, 'coverage_pct': 0.0}
            
            report_path = self.log_dir / 'weather_coverage.json'
            with open(report_path, 'w') as f:
                json.dump(coverage, f, indent=4)
            logger.info(f"Weather coverage report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save weather coverage report: {e}")
            
        return merged
