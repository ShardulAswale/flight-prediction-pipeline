"""
Task 4.1 / 4.2 — Route deviation features (thesis core contribution).

Computes baseline route profiles per (origin, destination) pair from full-flight
trajectories, then measures per-flight deviation from the baseline.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def _haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _normalize_trajectory(lats: np.ndarray, lons: np.ndarray,
                           alts: Optional[np.ndarray],
                           n_waypoints: int = 20) -> dict:
    """
    Resample a trajectory to n_waypoints equally spaced by cumulative distance.
    Returns dict with 'lats', 'lons', 'alts' arrays of length n_waypoints.
    """
    if len(lats) < 2:
        return None

    # Cumulative distance
    seg_dist = _haversine_vec(lats[:-1], lons[:-1], lats[1:], lons[1:])
    cum_dist = np.concatenate([[0.0], np.cumsum(seg_dist)])
    total_dist = cum_dist[-1]

    if total_dist < 1.0:  # less than 1km route
        return None

    # Resample at equal distance intervals
    target_dists = np.linspace(0, total_dist, n_waypoints)
    resampled_lats = np.interp(target_dists, cum_dist, lats)
    resampled_lons = np.interp(target_dists, cum_dist, lons)
    resampled_alts = None
    if alts is not None and len(alts) == len(lats):
        resampled_alts = np.interp(target_dists, cum_dist, alts)

    return {
        'lats': resampled_lats,
        'lons': resampled_lons,
        'alts': resampled_alts,
        'total_dist_km': float(total_dist),
    }


class RouteProfiler:
    """Builds baseline route profiles from historical trajectories."""

    def __init__(self, n_waypoints: int = 20, min_flights: int = 5):
        self.n_waypoints = n_waypoints
        self.min_flights = min_flights
        self.profiles: dict[tuple[str, str], dict] = {}

    def build_profiles(self, df_traj: pd.DataFrame,
                        df_features: pd.DataFrame) -> dict:
        """
        Task 4.1 — Build baseline route profiles.
        
        df_traj: trajectory points with trajectory_id, latitude, longitude, altitude/baro_altitude
        df_features: flight-level features with trajectory_id, origin, destination (from merge)
        
        We only use flights that have both origin and destination mapped.
        """
        # Need origin/dest mapping
        if 'origin' not in df_features.columns or 'destination' not in df_features.columns:
            logger.warning("Cannot build route profiles: missing origin/destination in features.")
            return {}

        route_mapping = df_features[['trajectory_id', 'origin', 'destination']].dropna()
        route_mapping = route_mapping.drop_duplicates(subset=['trajectory_id'])

        # Determine altitude column
        alt_col = 'altitude'
        if alt_col not in df_traj.columns:
            if 'baro_altitude' in df_traj.columns:
                alt_col = 'baro_altitude'
            elif 'geo_altitude' in df_traj.columns:
                alt_col = 'geo_altitude'
            else:
                alt_col = None

        # Group trajectory points by trajectory_id
        logger.info("Building route profiles from %d trajectory IDs...", len(route_mapping))
        route_trajectories: dict[tuple[str, str], list[dict]] = {}

        for _, row in route_mapping.iterrows():
            traj_id = row['trajectory_id']
            origin = row['origin']
            dest = row['destination']
            route_key = (str(origin), str(dest))

            traj_points = df_traj[df_traj['trajectory_id'] == traj_id].sort_values('timestamp')
            if len(traj_points) < 5:
                continue

            lats = traj_points['latitude'].dropna().values
            lons = traj_points['longitude'].dropna().values
            alts = traj_points[alt_col].dropna().values if alt_col else None

            if len(lats) < 5 or len(lons) < 5:
                continue

            normalized = _normalize_trajectory(lats, lons, alts, self.n_waypoints)
            if normalized is None:
                continue

            if route_key not in route_trajectories:
                route_trajectories[route_key] = []
            route_trajectories[route_key].append(normalized)

        # Compute median profile for each route
        for route_key, trajectories in route_trajectories.items():
            if len(trajectories) < self.min_flights:
                continue

            all_lats = np.array([t['lats'] for t in trajectories])
            all_lons = np.array([t['lons'] for t in trajectories])
            all_dists = [t['total_dist_km'] for t in trajectories]

            profile = {
                'median_lats': np.median(all_lats, axis=0),
                'median_lons': np.median(all_lons, axis=0),
                'median_dist_km': float(np.median(all_dists)),
                'n_flights': len(trajectories),
            }

            # Altitude profile if available
            has_alts = all(t['alts'] is not None for t in trajectories)
            if has_alts:
                all_alts = np.array([t['alts'] for t in trajectories])
                profile['median_alts'] = np.median(all_alts, axis=0)

            self.profiles[route_key] = profile

        logger.info("Built %d route profiles (min %d flights each).",
                     len(self.profiles), self.min_flights)
        return self.profiles

    def compute_deviation_features(self, df_traj: pd.DataFrame,
                                    df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Task 4.2 — Compute per-flight route deviation features.
        
        Returns a DataFrame with trajectory_id and deviation feature columns.
        """
        if not self.profiles:
            logger.warning("No route profiles available. Call build_profiles() first.")
            return pd.DataFrame()

        alt_col = 'altitude'
        if alt_col not in df_traj.columns:
            if 'baro_altitude' in df_traj.columns:
                alt_col = 'baro_altitude'
            elif 'geo_altitude' in df_traj.columns:
                alt_col = 'geo_altitude'
            else:
                alt_col = None

        route_mapping = df_features[['trajectory_id', 'origin', 'destination']].dropna()
        route_mapping = route_mapping.drop_duplicates(subset=['trajectory_id'])

        results = []
        matched = 0

        for _, row in route_mapping.iterrows():
            traj_id = row['trajectory_id']
            route_key = (str(row['origin']), str(row['destination']))

            if route_key not in self.profiles:
                results.append({'trajectory_id': traj_id})
                continue

            profile = self.profiles[route_key]
            traj_points = df_traj[df_traj['trajectory_id'] == traj_id].sort_values('timestamp')

            if len(traj_points) < 3:
                results.append({'trajectory_id': traj_id})
                continue

            lats = traj_points['latitude'].dropna().values
            lons = traj_points['longitude'].dropna().values
            alts = traj_points[alt_col].dropna().values if alt_col else None

            normalized = _normalize_trajectory(lats, lons, alts, self.n_waypoints)
            if normalized is None:
                results.append({'trajectory_id': traj_id})
                continue

            # Lateral deviation: distance from each waypoint to baseline
            lat_devs = _haversine_vec(
                normalized['lats'], normalized['lons'],
                profile['median_lats'], profile['median_lons'],
            )

            # Route stretch ratio
            gc_dist = profile['median_dist_km']
            stretch_ratio = normalized['total_dist_km'] / gc_dist if gc_dist > 0 else np.nan

            # Approach deviation (last 25% of waypoints)
            approach_idx = int(self.n_waypoints * 0.75)
            approach_devs = lat_devs[approach_idx:]

            feat = {
                'trajectory_id': traj_id,
                'lateral_deviation_mean_km': float(np.mean(lat_devs)),
                'lateral_deviation_max_km': float(np.max(lat_devs)),
                'lateral_deviation_std_km': float(np.std(lat_devs)),
                'route_stretch_ratio': float(stretch_ratio),
                'approach_deviation_km': float(np.mean(approach_devs)) if len(approach_devs) > 0 else np.nan,
            }

            # Altitude deviation if profiles have altitude
            if normalized['alts'] is not None and 'median_alts' in profile:
                alt_devs = np.abs(normalized['alts'] - profile['median_alts'])
                feat['altitude_deviation_mean_m'] = float(np.mean(alt_devs))
                feat['altitude_deviation_max_m'] = float(np.max(alt_devs))

            matched += 1
            results.append(feat)

        logger.info("Computed route deviation features for %d/%d flights with profiles.",
                     matched, len(route_mapping))
        return pd.DataFrame(results)

    @staticmethod
    def _iter_trajectory_groups_from_parquet(
        traj_path: str | Path,
        *,
        batch_size: int = 200_000,
    ):
        traj_path = Path(traj_path)
        parquet_file = pq.ParquetFile(traj_path)
        columns = [c for c in ['trajectory_id', 'timestamp', 'latitude', 'longitude', 'altitude', 'baro_altitude', 'geo_altitude'] if c in parquet_file.schema.names]
        carry = pd.DataFrame()

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            chunk = batch.to_pandas()
            if not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            if chunk.empty:
                continue

            chunk = chunk.sort_values(['trajectory_id', 'timestamp'], kind='stable').reset_index(drop=True)
            last_traj = chunk['trajectory_id'].iloc[-1]
            complete = chunk[chunk['trajectory_id'] != last_traj].copy()
            carry = chunk[chunk['trajectory_id'] == last_traj].copy()

            if not complete.empty:
                for trajectory_id, group in complete.groupby('trajectory_id', sort=False):
                    yield trajectory_id, group.reset_index(drop=True)

        if not carry.empty:
            for trajectory_id, group in carry.groupby('trajectory_id', sort=False):
                yield trajectory_id, group.reset_index(drop=True)

    def build_profiles_from_parquet(
        self,
        traj_path: str | Path,
        df_features: pd.DataFrame,
        *,
        batch_size: int = 200_000,
    ) -> dict:
        """Streaming profile builder for large trajectory parquet files."""
        if 'origin' not in df_features.columns or 'destination' not in df_features.columns:
            logger.warning("Cannot build route profiles: missing origin/destination in features.")
            return {}

        mapping = df_features[['trajectory_id', 'origin', 'destination']].dropna().drop_duplicates(subset=['trajectory_id']).copy()
        if 'is_full_flight' in df_features.columns:
            full_ids = set(
                df_features.loc[df_features['is_full_flight'].fillna(False).astype(bool), 'trajectory_id']
                .dropna()
                .astype(str)
            )
            mapping = mapping[mapping['trajectory_id'].astype(str).isin(full_ids)].copy()

        if mapping.empty:
            logger.warning("No full-flight mapped trajectories available for route profile construction.")
            return {}

        route_lookup = {
            str(row['trajectory_id']): (str(row['origin']), str(row['destination']))
            for _, row in mapping.iterrows()
        }
        route_trajectories: dict[tuple[str, str], list[dict]] = {}

        for trajectory_id, group in self._iter_trajectory_groups_from_parquet(traj_path, batch_size=batch_size):
            route_key = route_lookup.get(str(trajectory_id))
            if route_key is None:
                continue

            alt_col = 'altitude'
            if alt_col not in group.columns:
                if 'baro_altitude' in group.columns:
                    alt_col = 'baro_altitude'
                elif 'geo_altitude' in group.columns:
                    alt_col = 'geo_altitude'
                else:
                    alt_col = None

            lats = group['latitude'].dropna().to_numpy()
            lons = group['longitude'].dropna().to_numpy()
            alts = group[alt_col].dropna().to_numpy() if alt_col else None
            if len(lats) < 5 or len(lons) < 5:
                continue

            normalized = _normalize_trajectory(lats, lons, alts, self.n_waypoints)
            if normalized is None:
                continue

            route_trajectories.setdefault(route_key, []).append(normalized)

        self.profiles = {}
        for route_key, trajectories in route_trajectories.items():
            if len(trajectories) < self.min_flights:
                continue

            all_lats = np.array([t['lats'] for t in trajectories])
            all_lons = np.array([t['lons'] for t in trajectories])
            all_dists = [t['total_dist_km'] for t in trajectories]
            profile = {
                'median_lats': np.median(all_lats, axis=0),
                'median_lons': np.median(all_lons, axis=0),
                'median_dist_km': float(np.median(all_dists)),
                'n_flights': len(trajectories),
            }
            has_alts = all(t['alts'] is not None for t in trajectories)
            if has_alts:
                all_alts = np.array([t['alts'] for t in trajectories])
                profile['median_alts'] = np.median(all_alts, axis=0)
            self.profiles[route_key] = profile

        logger.info("Built %d route profiles from %s.", len(self.profiles), traj_path)
        return self.profiles

    def compute_deviation_features_from_parquet(
        self,
        traj_path: str | Path,
        df_features: pd.DataFrame,
        *,
        batch_size: int = 200_000,
    ) -> pd.DataFrame:
        """Streaming deviation calculator for large trajectory parquet files."""
        if not self.profiles:
            logger.warning("No route profiles available. Skipping route deviation computation.")
            return pd.DataFrame()

        mapping = (
            df_features[['trajectory_id', 'origin', 'destination']]
            .dropna()
            .drop_duplicates(subset=['trajectory_id'])
            .copy()
        )
        route_lookup = {
            str(row['trajectory_id']): (str(row['origin']), str(row['destination']))
            for _, row in mapping.iterrows()
        }

        results = []
        matched = 0
        for trajectory_id, group in self._iter_trajectory_groups_from_parquet(traj_path, batch_size=batch_size):
            route_key = route_lookup.get(str(trajectory_id))
            if route_key is None or route_key not in self.profiles:
                continue

            profile = self.profiles[route_key]
            alt_col = 'altitude'
            if alt_col not in group.columns:
                if 'baro_altitude' in group.columns:
                    alt_col = 'baro_altitude'
                elif 'geo_altitude' in group.columns:
                    alt_col = 'geo_altitude'
                else:
                    alt_col = None

            lats = group['latitude'].dropna().to_numpy()
            lons = group['longitude'].dropna().to_numpy()
            alts = group[alt_col].dropna().to_numpy() if alt_col else None
            normalized = _normalize_trajectory(lats, lons, alts, self.n_waypoints)
            if normalized is None:
                continue

            lat_devs = _haversine_vec(
                normalized['lats'], normalized['lons'],
                profile['median_lats'], profile['median_lons'],
            )
            gc_dist = profile['median_dist_km']
            stretch_ratio = normalized['total_dist_km'] / gc_dist if gc_dist > 0 else np.nan
            approach_idx = int(self.n_waypoints * 0.75)
            approach_devs = lat_devs[approach_idx:]

            feat = {
                'trajectory_id': str(trajectory_id),
                'lateral_deviation_mean_km': float(np.mean(lat_devs)),
                'lateral_deviation_max_km': float(np.max(lat_devs)),
                'lateral_deviation_std_km': float(np.std(lat_devs)),
                'route_stretch_ratio': float(stretch_ratio),
                'approach_deviation_km': float(np.mean(approach_devs)) if len(approach_devs) > 0 else np.nan,
            }
            if normalized['alts'] is not None and 'median_alts' in profile:
                alt_devs = np.abs(normalized['alts'] - profile['median_alts'])
                feat['altitude_deviation_mean_m'] = float(np.mean(alt_devs))
                feat['altitude_deviation_max_m'] = float(np.max(alt_devs))

            matched += 1
            results.append(feat)

        logger.info("Computed route deviation features for %d trajectories from %s.", matched, traj_path)
        return pd.DataFrame(results)


def add_route_deviation_features(
    df_features: pd.DataFrame,
    traj_path: str | Path,
    *,
    n_waypoints: int = 20,
    min_flights: int = 5,
    batch_size: int = 200_000,
) -> pd.DataFrame:
    """Attach route-deviation features to a merged/enriched dataset."""
    traj_path = Path(traj_path)
    if not traj_path.exists() or df_features.empty or 'trajectory_id' not in df_features.columns:
        return df_features

    profiler = RouteProfiler(n_waypoints=n_waypoints, min_flights=min_flights)
    profiler.build_profiles_from_parquet(traj_path, df_features, batch_size=batch_size)
    if not profiler.profiles:
        logger.warning("Route deviation profiles could not be built; leaving deviation columns empty.")
        return df_features

    deviation_df = profiler.compute_deviation_features_from_parquet(traj_path, df_features, batch_size=batch_size)
    if deviation_df.empty:
        return df_features

    merged = df_features.merge(deviation_df, on='trajectory_id', how='left')
    logger.info("Merged route deviation features for %d/%d rows.", int(merged['lateral_deviation_mean_km'].notna().sum()), len(merged))
    return merged
