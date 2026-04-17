import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import shutil
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


class RunningStats:
    """Numerically stable running mean/variance tracker."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.max_value = np.nan

    def add(self, value: float):
        if pd.isna(value):
            return
        value = float(value)
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        self.max_value = value if pd.isna(self.max_value) else max(self.max_value, value)

    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    def std(self) -> float:
        return float(np.sqrt(self.variance()))


class TrajectoryAccumulator:
    """Streaming state for one trajectory."""

    def __init__(
        self,
        trajectory_id: str,
        icao24: str | None = None,
        callsign: str | None = None,
        max_gap_seconds: int = 15 * 60,
    ):
        self.trajectory_id = trajectory_id
        self.icao24 = None if pd.isna(icao24) else icao24
        self.callsign = "UNKNOWN" if pd.isna(callsign) or callsign in ("", None) else str(callsign)
        self.max_gap_seconds = max_gap_seconds
        self.start_timestamp = None
        self.end_timestamp = None
        self.num_points = 0

        self.altitude_stats = RunningStats()
        self.speed_stats = RunningStats()
        self.heading_stats = RunningStats()
        self.vertical_rate_stats = RunningStats()

        self.first_altitude = None
        self.last_altitude = None
        self.max_altitude = None

        self.prev_timestamp = None
        self.prev_altitude = None
        self.prev_latitude = None
        self.prev_longitude = None

        self.distance_km = 0.0
        self.altitude_change_count = 0
        self.smoothed_rate_window = deque(maxlen=3)
        self.last_smoothed_sign = None
        self.final_vertical_rates = deque(maxlen=10)
        self.inter_ping_stats = RunningStats()
        self.middle_gap_count = 0
        self.total_gap_seconds = 0.0
        self.trajectory_quality_status = None
        self.trajectory_quality_score = np.nan
        self.is_full_flight = None
        self.starts_groundish = None
        self.ends_groundish = None
        self.has_altitude_spike = None
        self.is_mappable = None
        self.route_coverage_fraction = np.nan
        self.gap_fraction_of_flight = np.nan
        self.max_inter_ping_seconds = np.nan
        self.median_inter_ping_seconds = np.nan
        self.ping_interval_cv = np.nan

    def update(self, row: Dict[str, Any], distance_fn):
        timestamp = row.get("timestamp")
        altitude = row.get("altitude")
        velocity = row.get("velocity")
        heading = row.get("heading")
        latitude = row.get("latitude")
        longitude = row.get("longitude")

        if pd.notna(timestamp):
            timestamp = float(timestamp)
            self.start_timestamp = timestamp if self.start_timestamp is None else min(self.start_timestamp, timestamp)
            self.end_timestamp = timestamp if self.end_timestamp is None else max(self.end_timestamp, timestamp)

        if pd.notna(row.get("icao24")) and self.icao24 is None:
            self.icao24 = str(row.get("icao24"))
        if pd.notna(row.get("callsign")) and (self.callsign == "UNKNOWN" or not self.callsign):
            self.callsign = str(row.get("callsign"))

        if self.trajectory_quality_status is None and row.get("trajectory_quality_status") is not None and not pd.isna(row.get("trajectory_quality_status")):
            self.trajectory_quality_status = str(row.get("trajectory_quality_status"))
            self.trajectory_quality_score = float(row.get("trajectory_quality_score", np.nan))
            self.is_full_flight = bool(row.get("is_full_flight", False))
            self.starts_groundish = bool(row.get("starts_groundish", False))
            self.ends_groundish = bool(row.get("ends_groundish", False))
            self.has_altitude_spike = bool(row.get("has_altitude_spike", False))
            self.is_mappable = bool(row.get("is_mappable", False))
            self.route_coverage_fraction = float(row.get("route_coverage_fraction", np.nan))
            self.gap_fraction_of_flight = float(row.get("gap_fraction_of_flight", np.nan))
            self.max_inter_ping_seconds = float(row.get("max_inter_ping_seconds", np.nan))
            self.median_inter_ping_seconds = float(row.get("median_inter_ping_seconds", np.nan))
            self.ping_interval_cv = float(row.get("ping_interval_cv", np.nan))

        self.num_points += 1
        self.altitude_stats.add(altitude)
        self.speed_stats.add(velocity)
        self.heading_stats.add(heading)

        if pd.notna(altitude):
            altitude = float(altitude)
            if self.first_altitude is None:
                self.first_altitude = altitude
            self.last_altitude = altitude
            self.max_altitude = altitude if self.max_altitude is None else max(self.max_altitude, altitude)

        if all(pd.notna(v) for v in [self.prev_latitude, self.prev_longitude, latitude, longitude]):
            self.distance_km += distance_fn(self.prev_latitude, self.prev_longitude, latitude, longitude)

        if (
            self.prev_timestamp is not None
            and pd.notna(timestamp)
            and self.prev_altitude is not None
            and pd.notna(altitude)
        ):
            dt = float(timestamp) - float(self.prev_timestamp)
            if dt > 0:
                self.inter_ping_stats.add(dt)
                if dt > self.max_gap_seconds:
                    self.middle_gap_count += 1
                    self.total_gap_seconds += dt
                vr = (float(altitude) - float(self.prev_altitude)) / dt
                self.vertical_rate_stats.add(vr)
                self.final_vertical_rates.append(vr)
                self.smoothed_rate_window.append(vr)
                if len(self.smoothed_rate_window) == 3:
                    smoothed = sum(self.smoothed_rate_window) / 3.0
                    sign = np.sign(smoothed)
                    if (
                        self.last_smoothed_sign is not None
                        and sign != 0
                        and self.last_smoothed_sign != 0
                        and sign != self.last_smoothed_sign
                    ):
                        self.altitude_change_count += 1
                    if sign != 0:
                        self.last_smoothed_sign = sign

        if pd.notna(timestamp):
            self.prev_timestamp = float(timestamp)
        if pd.notna(altitude):
            self.prev_altitude = float(altitude)
        if pd.notna(latitude):
            self.prev_latitude = float(latitude)
        if pd.notna(longitude):
            self.prev_longitude = float(longitude)

    def to_feature_row(self) -> Dict[str, Any]:
        if self.num_points < 2 or self.start_timestamp is None or self.end_timestamp is None:
            return {}

        duration = float(self.end_timestamp - self.start_timestamp)
        mean_speed = self.speed_stats.mean if self.speed_stats.count else np.nan
        heading_variability = self.heading_stats.variance() if self.heading_stats.count > 1 else 0.0
        takeoff_detected = bool(
            self.first_altitude is not None and self.max_altitude is not None and self.first_altitude < 2000 and self.max_altitude > 5000
        )
        landing_detected = bool(
            self.last_altitude is not None and self.max_altitude is not None and self.last_altitude < 2000 and self.max_altitude > 5000
        )

        holding_pattern_count = 0
        if (
            heading_variability > 5000
            and duration > 600
            and pd.notna(mean_speed)
            and self.distance_km < (mean_speed * duration / 1000.0) * 0.3
        ):
            holding_pattern_count = 1

        unstable_descent_flag = False
        if landing_detected and len(self.final_vertical_rates) > 2:
            unstable_descent_flag = float(np.std(self.final_vertical_rates, ddof=1)) > 5.0

        start_time_utc = pd.to_datetime(self.start_timestamp, unit='s', utc=True)
        end_time_utc = pd.to_datetime(self.end_timestamp, unit='s', utc=True)
        dep_anchor_confidence = 1.0 if takeoff_detected else 0.35
        arr_anchor_confidence = 1.0 if landing_detected else 0.35
        gap_fraction_of_flight = float(self.total_gap_seconds / duration) if duration > 0 else 0.0
        route_coverage_fraction = 1.0 - gap_fraction_of_flight
        if pd.notna(self.gap_fraction_of_flight):
            gap_fraction_of_flight = float(self.gap_fraction_of_flight)
            route_coverage_fraction = float(self.route_coverage_fraction)
        max_inter_ping_seconds = (
            float(self.inter_ping_stats.max_value)
            if self.inter_ping_stats.count
            else 0.0
        )
        if pd.notna(self.max_inter_ping_seconds):
            max_inter_ping_seconds = float(self.max_inter_ping_seconds)
        ping_interval_cv = 0.0
        if self.inter_ping_stats.count > 1 and self.inter_ping_stats.mean > 0:
            ping_interval_cv = float(self.inter_ping_stats.std() / self.inter_ping_stats.mean)
        if pd.notna(self.ping_interval_cv):
            ping_interval_cv = float(self.ping_interval_cv)
        trajectory_quality_status = self.trajectory_quality_status or (
            'full_flight' if (takeoff_detected and landing_detected) else (
                'partial_end_missing' if takeoff_detected else (
                    'partial_start_missing' if landing_detected else 'airborne_only'
                )
            )
        )
        trajectory_quality_score = (
            float(self.trajectory_quality_score)
            if pd.notna(self.trajectory_quality_score)
            else float(min(1.0, max(0.0, route_coverage_fraction * 0.5 + float(takeoff_detected) * 0.25 + float(landing_detected) * 0.25)))
        )
        is_full_flight = bool(self.is_full_flight) if self.is_full_flight is not None else bool(takeoff_detected and landing_detected)

        return {
            'trajectory_id': self.trajectory_id,
            'icao24': self.icao24,
            'callsign': self.callsign or 'UNKNOWN',
            'timestamp': self.start_timestamp,
            'start_time_utc': start_time_utc,
            'end_time_utc': end_time_utc,
            'dep_anchor_confidence': dep_anchor_confidence,
            'arr_anchor_confidence': arr_anchor_confidence,
            'dep_anchor_is_partial': not takeoff_detected,
            'arr_anchor_is_partial': not landing_detected,
            'trajectory_quality_status': trajectory_quality_status,
            'trajectory_quality_score': trajectory_quality_score,
            'is_full_flight': is_full_flight,
            'starts_groundish': bool(self.starts_groundish) if self.starts_groundish is not None else bool(takeoff_detected),
            'ends_groundish': bool(self.ends_groundish) if self.ends_groundish is not None else bool(landing_detected),
            'has_altitude_spike': bool(self.has_altitude_spike) if self.has_altitude_spike is not None else False,
            'is_mappable': bool(self.is_mappable) if self.is_mappable is not None else self.distance_km > 0,
            'middle_gap_count': int(self.middle_gap_count),
            'gap_fraction_of_flight': gap_fraction_of_flight,
            'route_coverage_fraction': route_coverage_fraction,
            'max_inter_ping_seconds': max_inter_ping_seconds,
            'median_inter_ping_seconds': float(self.median_inter_ping_seconds) if pd.notna(self.median_inter_ping_seconds) else np.nan,
            'ping_interval_cv': ping_interval_cv,
            'flight_duration': duration,
            'trajectory_length': self.distance_km,
            'num_points': self.num_points,
            'mean_altitude': self.altitude_stats.mean if self.altitude_stats.count else np.nan,
            'altitude_variance': self.altitude_stats.variance(),
            'mean_speed': mean_speed,
            'max_speed': self.speed_stats.max_value if self.speed_stats.count else np.nan,
            'speed_std': self.speed_stats.std(),
            'vertical_rate_std': self.vertical_rate_stats.std(),
            'heading_variability': heading_variability,
            'takeoff_detected': takeoff_detected,
            'landing_detected': landing_detected,
            'holding_pattern_count': holding_pattern_count,
            'altitude_change_count': self.altitude_change_count,
            'unstable_descent_flag': unstable_descent_flag
        }

class FeatureExtractor:
    """Extracts features from flight trajectories."""
    
    def __init__(self, max_gap_minutes: int = 15):
        self.max_gap_seconds = max_gap_minutes * 60

    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points on the earth."""
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def _prepare_trajectory_frame(self, df_traj: pd.DataFrame) -> pd.DataFrame:
        """Align input trajectory columns to the feature extractor contract."""
        df_traj = df_traj.copy()

        if 'altitude' not in df_traj.columns:
            if 'geo_altitude' in df_traj.columns:
                df_traj['altitude'] = df_traj['geo_altitude']
                if 'baro_altitude' in df_traj.columns:
                    df_traj['altitude'] = df_traj['altitude'].fillna(df_traj['baro_altitude'])
            elif 'baro_altitude' in df_traj.columns:
                df_traj['altitude'] = df_traj['baro_altitude']

        if 'heading' not in df_traj.columns and 'true_track' in df_traj.columns:
            df_traj['heading'] = df_traj['true_track']

        numeric_cols = [
            'timestamp',
            'latitude',
            'longitude',
            'altitude',
            'velocity',
            'heading',
            'vertical_rate',
        ]
        for col in numeric_cols:
            if col in df_traj.columns:
                df_traj[col] = pd.to_numeric(df_traj[col], errors='coerce')

        return df_traj

    def _extract_single_trajectory(self, df_traj: pd.DataFrame) -> Dict[str, Any]:
        """Extract features for a single trajectory group."""
        if len(df_traj) < 2:
            return {}

        df_traj = self._prepare_trajectory_frame(df_traj)
        if 'timestamp' not in df_traj.columns:
            return {}

        df_traj = df_traj.sort_values('timestamp')

        duration = df_traj['timestamp'].max() - df_traj['timestamp'].min()
        num_points = len(df_traj)

        # Series extractions
        altitudes = df_traj['altitude'].dropna() if 'altitude' in df_traj.columns else pd.Series(dtype='float64')
        velocities = df_traj['velocity'].dropna() if 'velocity' in df_traj.columns else pd.Series(dtype='float64')
        headings = df_traj['heading'].dropna() if 'heading' in df_traj.columns else pd.Series(dtype='float64')

        # Altitude features
        mean_altitude = altitudes.mean() if not altitudes.empty else np.nan
        alt_variance = altitudes.var() if len(altitudes) > 1 else 0.0

        # Speed features
        mean_speed = velocities.mean() if not velocities.empty else np.nan
        max_speed = velocities.max() if not velocities.empty else np.nan
        speed_std = velocities.std() if len(velocities) > 1 else 0.0

        heading_variability = headings.var() if len(headings) > 1 else 0.0

        # Distance calculation
        lats = df_traj['latitude'].values if 'latitude' in df_traj.columns else np.array([])
        lons = df_traj['longitude'].values if 'longitude' in df_traj.columns else np.array([])
        dist = 0.0
        for i in range(1, len(lats)):
            if any(pd.isna(v) for v in [lats[i - 1], lons[i - 1], lats[i], lons[i]]):
                continue
            dist += self.haversine_distance(lats[i-1], lons[i-1], lats[i], lons[i])

        # Vertical profile logic
        time_diffs = df_traj['timestamp'].diff()
        vertical_rates = pd.Series(dtype='float64')
        if 'altitude' in df_traj.columns:
            vertical_rates = df_traj['altitude'].diff() / time_diffs.replace(0, np.nan)
        vertical_rate_std = vertical_rates.std() if len(vertical_rates.dropna()) > 1 else 0.0

        alt_min = altitudes.min() if not altitudes.empty else 0
        alt_max = altitudes.max() if not altitudes.empty else 0

        takeoff_detected = False
        landing_detected = False

        if not altitudes.empty and len(altitudes) >= 2:
            first_alt = altitudes.iloc[0]
            last_alt = altitudes.iloc[-1]
            # Simple threshold logic for takeoff/landing
            if first_alt < 2000 and alt_max > 5000:
                takeoff_detected = True
            if last_alt < 2000 and alt_max > 5000:
                landing_detected = True

        # Phase changes
        altitude_change_count = 0
        if len(vertical_rates.dropna()) > 2:
            # Count zero-crossings of smoothed vertical rates to denote phase changes
            smoothed_vr = vertical_rates.rolling(3).mean().dropna()
            signs = np.sign(smoothed_vr)
            sign_changes = ((signs.shift(1) * signs) < 0).sum()
            altitude_change_count = int(sign_changes)

        # Holding Patterns
        # Approx: highly variable heading over a long duration without much distance covered.
        holding_pattern_count = 0
        if (
            heading_variability > 5000
            and duration > 600
            and pd.notna(mean_speed)
            and dist < (mean_speed * duration / 1000) * 0.3
        ):
            holding_pattern_count = 1

        unstable_descent_flag = False
        if landing_detected:
            # Check last 10 points for high vertical rate variability
            final_vr = vertical_rates.tail(10).dropna()
            if len(final_vr) > 2 and final_vr.std() > 5.0:  # arbitrary threshold
                unstable_descent_flag = True

        start_time_utc = pd.to_datetime(df_traj['timestamp'].min(), unit='s', utc=True)
        end_time_utc = pd.to_datetime(df_traj['timestamp'].max(), unit='s', utc=True)
        dep_anchor_confidence = 1.0 if takeoff_detected else 0.35
        arr_anchor_confidence = 1.0 if landing_detected else 0.35
        inter_ping = time_diffs.dropna()
        positive_inter_ping = inter_ping[inter_ping > 0]
        large_gaps = positive_inter_ping[positive_inter_ping > self.max_gap_seconds]
        total_gap_seconds = float(large_gaps.sum()) if not large_gaps.empty else 0.0
        gap_fraction_of_flight = float(total_gap_seconds / duration) if duration > 0 else 0.0
        route_coverage_fraction = 1.0 - gap_fraction_of_flight

        trajectory_quality_status = (
            str(df_traj['trajectory_quality_status'].dropna().iloc[0])
            if 'trajectory_quality_status' in df_traj.columns and df_traj['trajectory_quality_status'].notna().any()
            else (
                'full_flight' if (takeoff_detected and landing_detected) else (
                    'partial_end_missing' if takeoff_detected else (
                        'partial_start_missing' if landing_detected else 'airborne_only'
                    )
                )
            )
        )
        trajectory_quality_score = (
            float(pd.to_numeric(df_traj['trajectory_quality_score'], errors='coerce').dropna().iloc[0])
            if 'trajectory_quality_score' in df_traj.columns and pd.to_numeric(df_traj['trajectory_quality_score'], errors='coerce').notna().any()
            else float(min(1.0, max(0.0, route_coverage_fraction * 0.5 + float(takeoff_detected) * 0.25 + float(landing_detected) * 0.25)))
        )
        is_full_flight = (
            bool(df_traj['is_full_flight'].dropna().iloc[0])
            if 'is_full_flight' in df_traj.columns and df_traj['is_full_flight'].notna().any()
            else bool(takeoff_detected and landing_detected)
        )

        return {
            'trajectory_id': df_traj['trajectory_id'].iloc[0],
            'icao24': df_traj['icao24'].iloc[0] if 'icao24' in df_traj.columns else None,
            'callsign': df_traj['callsign'].iloc[0] if 'callsign' in df_traj.columns else 'UNKNOWN',
            'timestamp': df_traj['timestamp'].min(),  # Start of flight
            'start_time_utc': start_time_utc,
            'end_time_utc': end_time_utc,
            'dep_anchor_confidence': dep_anchor_confidence,
            'arr_anchor_confidence': arr_anchor_confidence,
            'dep_anchor_is_partial': not takeoff_detected,
            'arr_anchor_is_partial': not landing_detected,
            'trajectory_quality_status': trajectory_quality_status,
            'trajectory_quality_score': trajectory_quality_score,
            'is_full_flight': is_full_flight,
            'starts_groundish': bool(df_traj['starts_groundish'].dropna().iloc[0]) if 'starts_groundish' in df_traj.columns and df_traj['starts_groundish'].notna().any() else bool(takeoff_detected),
            'ends_groundish': bool(df_traj['ends_groundish'].dropna().iloc[0]) if 'ends_groundish' in df_traj.columns and df_traj['ends_groundish'].notna().any() else bool(landing_detected),
            'has_altitude_spike': bool(df_traj['has_altitude_spike'].dropna().iloc[0]) if 'has_altitude_spike' in df_traj.columns and df_traj['has_altitude_spike'].notna().any() else False,
            'is_mappable': bool(df_traj['is_mappable'].dropna().iloc[0]) if 'is_mappable' in df_traj.columns and df_traj['is_mappable'].notna().any() else dist > 0,
            'middle_gap_count': int(len(large_gaps)),
            'gap_fraction_of_flight': gap_fraction_of_flight,
            'route_coverage_fraction': route_coverage_fraction,
            'max_inter_ping_seconds': float(positive_inter_ping.max()) if not positive_inter_ping.empty else 0.0,
            'median_inter_ping_seconds': float(positive_inter_ping.median()) if not positive_inter_ping.empty else np.nan,
            'ping_interval_cv': float(positive_inter_ping.std(ddof=0) / positive_inter_ping.mean()) if len(positive_inter_ping) >= 2 and float(positive_inter_ping.mean()) > 0 else 0.0,
            'flight_duration': duration,
            'trajectory_length': dist,
            'num_points': num_points,
            'mean_altitude': mean_altitude,
            'altitude_variance': alt_variance,
            'mean_speed': mean_speed,
            'max_speed': max_speed,
            'speed_std': speed_std,
            'vertical_rate_std': vertical_rate_std,
            'heading_variability': heading_variability,
            'takeoff_detected': takeoff_detected,
            'landing_detected': landing_detected,
            'holding_pattern_count': holding_pattern_count,
            'altitude_change_count': altitude_change_count,
            'unstable_descent_flag': unstable_descent_flag
        }

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for all trajectories."""
        logger.info(f"Extracting features for {df['trajectory_id'].nunique()} flights...")
        
        features_list = []
        for traj_id, group in df.groupby('trajectory_id'):
            feats = self._extract_single_trajectory(group)
            if feats:
                features_list.append(feats)
                
        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df)} valid feature vectors.")
        return features_df

    def extract_features_streaming(self, input_path: str | Path, batch_size: int = 200_000) -> pd.DataFrame:
        """Extract features from trajectory parquet in batches to avoid OOM."""
        input_path = Path(input_path)
        parquet_file = pq.ParquetFile(input_path)
        columns = [
            col for col in self._streaming_feature_columns()
            if col in parquet_file.schema.names
        ]

        accumulators: dict[str, TrajectoryAccumulator] = {}
        logger.info("Streaming feature extraction from %s using %s columns.", input_path, len(columns))

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            df_batch = self._prepare_trajectory_frame(batch.to_pandas())
            if 'trajectory_id' not in df_batch.columns:
                raise ValueError("Trajectory parquet is missing 'trajectory_id'. Run notebook 02 first.")

            df_batch = df_batch.sort_values(['timestamp', 'trajectory_id'], kind='stable')
            for row in df_batch.itertuples(index=False):
                row_dict = row._asdict()
                traj_id = str(row_dict.get('trajectory_id'))
                if traj_id not in accumulators:
                    accumulators[traj_id] = TrajectoryAccumulator(
                        trajectory_id=traj_id,
                        icao24=row_dict.get('icao24'),
                        callsign=row_dict.get('callsign'),
                        max_gap_seconds=self.max_gap_seconds,
                    )
                accumulators[traj_id].update(row_dict, self.haversine_distance)

        features_list = []
        for accumulator in accumulators.values():
            feats = accumulator.to_feature_row()
            if feats:
                features_list.append(feats)

        features_df = pd.DataFrame(features_list)
        logger.info(f"Extracted {len(features_df)} valid feature vectors via streaming.")
        return features_df

    def _streaming_feature_columns(self) -> list[str]:
        return [
            'trajectory_id',
            'icao24',
            'callsign',
            'timestamp',
            'latitude',
            'longitude',
            'geo_altitude',
            'baro_altitude',
            'velocity',
            'true_track',
            'vertical_rate',
            'altitude',
            'heading',
            'trajectory_quality_status',
            'trajectory_quality_score',
            'is_full_flight',
            'starts_groundish',
            'ends_groundish',
            'has_altitude_spike',
            'is_mappable',
            'middle_gap_count',
            'largest_gap_minutes',
            'total_gap_minutes',
            'gap_fraction_of_flight',
            'route_coverage_fraction',
            'max_inter_ping_seconds',
            'median_inter_ping_seconds',
            'ping_interval_cv',
        ]

    def extract_features_parallel(
        self,
        input_path: str | Path,
        num_workers: int | None = None,
        partition_batch_size: int = 200_000,
        worker_batch_size: int = 200_000,
        cleanup: bool = True,
    ) -> pd.DataFrame:
        """Parallel feature extraction via trajectory-id hash partitioning."""
        input_path = Path(input_path)
        if num_workers is None:
            num_workers = max(1, min(6, (os.cpu_count() or 2) - 1))
        if num_workers <= 1:
            logger.info("Parallel extraction requested with <=1 worker; falling back to streaming.")
            return self.extract_features_streaming(input_path, batch_size=worker_batch_size)

        parquet_file = pq.ParquetFile(input_path)
        columns = [col for col in self._streaming_feature_columns() if col in parquet_file.schema.names]
        if 'trajectory_id' not in columns:
            raise ValueError("Trajectory parquet is missing 'trajectory_id'. Run notebook 02 first.")

        partition_dir = input_path.parent / "_feature_partitions"
        if partition_dir.exists():
            shutil.rmtree(partition_dir)
        partition_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Partitioning %s into %s trajectory-id shards for parallel feature extraction.",
            input_path,
            num_workers,
        )

        writers: dict[int, pq.ParquetWriter] = {}
        partition_paths = {idx: partition_dir / f"partition_{idx:02d}.parquet" for idx in range(num_workers)}

        try:
            for batch in parquet_file.iter_batches(batch_size=partition_batch_size, columns=columns):
                df_batch = self._prepare_trajectory_frame(batch.to_pandas())
                partition_ids = (
                    pd.util.hash_pandas_object(df_batch['trajectory_id'].astype('string'), index=False).to_numpy()
                    % num_workers
                )

                for partition_id in range(num_workers):
                    subset = df_batch.loc[partition_ids == partition_id]
                    if subset.empty:
                        continue

                    table = pa.Table.from_pandas(subset, preserve_index=False)
                    if partition_id not in writers:
                        writers[partition_id] = pq.ParquetWriter(
                            partition_paths[partition_id],
                            table.schema,
                            compression='snappy',
                        )
                    writers[partition_id].write_table(table)
        finally:
            for writer in writers.values():
                writer.close()

        available_partitions = [path for path in partition_paths.values() if path.exists()]
        logger.info("Created %s non-empty feature partitions.", len(available_partitions))

        feature_frames: list[pd.DataFrame] = []
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_map = {
                    executor.submit(_extract_partition_features_worker, str(path), worker_batch_size): path
                    for path in available_partitions
                }
                for future in as_completed(future_map):
                    partition_path = future_map[future]
                    df_part = future.result()
                    logger.info("Completed feature extraction for %s (%s rows).", partition_path.name, len(df_part))
                    if not df_part.empty:
                        feature_frames.append(df_part)
        except (PermissionError, OSError) as exc:
            logger.warning(
                "Parallel feature extraction is unavailable on this machine (%s). "
                "Falling back to serial partition processing.",
                exc,
            )
            for partition_path in available_partitions:
                df_part = _extract_partition_features_worker(str(partition_path), worker_batch_size)
                logger.info("Completed serial feature extraction for %s (%s rows).", partition_path.name, len(df_part))
                if not df_part.empty:
                    feature_frames.append(df_part)

        combined = pd.concat(feature_frames, ignore_index=True) if feature_frames else pd.DataFrame()

        if cleanup and partition_dir.exists():
            shutil.rmtree(partition_dir)

        logger.info("Parallel feature extraction produced %s feature rows.", len(combined))
        return combined

    def save_features(self, df: pd.DataFrame, output_path: str | Path):
        """Save trajectory features to Parquet."""
        if df.empty:
            logger.warning("No features to save.")
            return
            
        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        
        logger.info(f"Saving features to {output_path}")
        df.to_parquet(output_path, index=False)


def _extract_partition_features_worker(partition_path: str, batch_size: int) -> pd.DataFrame:
    """Top-level worker for Windows-safe multiprocessing."""
    extractor = FeatureExtractor()
    return extractor.extract_features_streaming(partition_path, batch_size=batch_size)
