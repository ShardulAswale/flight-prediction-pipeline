"""
Task 3.2 / 3.3 / 3.4 — Temporal, airport-level, and congestion feature enrichment.

Adds features that are known top predictors of flight delay:
- Temporal: hour, day-of-week, weekend, peak-hour
- Airport:  flight volume, historical delay rate, great-circle distance
- Congestion: flights in ±1hr window at same airport
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Airport coordinates (top airports, ICAO keyed) ──────────────────────
# Lat/lon for great-circle distance calculation
AIRPORT_COORDS = {
    'KATL': (33.6407, -84.4277), 'KDFW': (32.8998, -97.0403),
    'KORD': (41.9742, -87.9073), 'KDEN': (39.8561, -104.6737),
    'KJFK': (40.6413, -73.7781), 'KLAX': (33.9425, -118.4081),
    'KSFO': (37.6213, -122.3790), 'KLAS': (36.0840, -115.1537),
    'KSEA': (47.4502, -122.3088), 'KCLT': (35.2144, -80.9473),
    'KMCO': (28.4312, -81.3081), 'KMIA': (25.7959, -80.2870),
    'KPHX': (33.4373, -112.0078), 'KIAH': (29.9902, -95.3368),
    'KBOS': (42.3656, -71.0096), 'KEWR': (40.6895, -74.1745),
    'KMSP': (44.8848, -93.2223), 'KDTW': (42.2162, -83.3554),
    'KPHL': (39.8744, -75.2424), 'KLGA': (40.7769, -73.8740),
    'KBWI': (39.1754, -76.6684), 'KSLC': (40.7884, -111.9778),
    'KSAN': (32.7336, -117.1897), 'KIAD': (38.9531, -77.4565),
    'KDCA': (38.8512, -77.0402), 'KMDW': (41.7868, -87.7522),
    'KTPA': (27.9756, -82.5333), 'KFLL': (26.0742, -80.1506),
    'KPDX': (45.5898, -122.5951), 'PHNL': (21.3245, -157.9251),
    'EGLL': (51.4700, -0.4543), 'LFPG': (49.0097, 2.5479),
    'EDDF': (50.0379, 8.5622), 'EHAM': (52.3105, 4.7683),
    'LEMD': (40.4983, -3.5676), 'LEBL': (41.2971, 2.0785),
    'LIRF': (41.8003, 12.2389), 'LTFM': (41.2753, 28.7519),
    'EDDM': (48.3538, 11.7861), 'LSZH': (47.4647, 8.5492),
    'EGKK': (51.1537, -0.1821), 'EBBR': (50.9014, 4.4844),
    'LOWW': (48.1103, 16.5697), 'EIDW': (53.4264, -6.2499),
    'EKCH': (55.6180, 12.6561), 'ENGM': (60.1939, 11.1004),
    'ESSA': (59.6519, 17.9186), 'LPPT': (38.7756, -9.1354),
    'EFHK': (60.3172, 24.9633), 'LGAV': (37.9364, 23.9445),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def add_temporal_features(df: pd.DataFrame,
                          dep_col: str = 'scheduled_dep') -> pd.DataFrame:
    """Task 3.2 — Extract temporal features from scheduled departure."""
    df = df.copy()

    dep_ts = pd.to_datetime(df[dep_col], utc=True, errors='coerce')

    df['dep_hour'] = dep_ts.dt.hour
    df['dep_day_of_week'] = dep_ts.dt.dayofweek  # 0=Mon
    df['dep_month'] = dep_ts.dt.month
    df['is_weekend'] = dep_ts.dt.dayofweek.ge(5).astype(int)
    df['is_peak_hour'] = (
        dep_ts.dt.hour.between(6, 9) | dep_ts.dt.hour.between(16, 20)
    ).astype(int)
    for col in ['dep_hour', 'dep_day_of_week', 'dep_month', 'is_weekend', 'is_peak_hour']:
        if df[col].notna().all():
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')

    logger.info("Added 5 temporal features (dep_hour, dep_day_of_week, dep_month, is_weekend, is_peak_hour)")
    return df


def add_airport_features(df: pd.DataFrame,
                          origin_col: str = 'origin',
                          dest_col: str = 'destination',
                          label_col: str = 'label') -> pd.DataFrame:
    """
    Task 3.3 — Airport-level features.
    
    - Flight volume at origin and destination
    - Historical delay rate (global, not leaky per-row)
    - Great-circle distance between origin and destination
    """
    df = df.copy()

    # Flight volume
    origin_counts = df[origin_col].value_counts().to_dict()
    dest_counts = df[dest_col].value_counts().to_dict()
    df['origin_flight_count'] = df[origin_col].map(origin_counts).fillna(0).astype(int)
    df['dest_flight_count'] = df[dest_col].map(dest_counts).fillna(0).astype(int)
    total_rows = max(len(df), 1)
    df['origin_encoded'] = df['origin_flight_count'] / total_rows
    df['dest_encoded'] = df['dest_flight_count'] / total_rows

    # Historical delay rate at origin & destination (leave-one-out when labels exist)
    if label_col in df.columns:
        is_late = df[label_col].isin(['Late', 'Cancelled']).astype(int)
        origin_total = df.groupby(origin_col)[origin_col].transform('count')
        origin_late = is_late.groupby(df[origin_col]).transform('sum')
        dest_total = df.groupby(dest_col)[dest_col].transform('count')
        dest_late = is_late.groupby(df[dest_col]).transform('sum')

        df['origin_delay_rate'] = np.where(
            origin_total > 1,
            (origin_late - is_late) / (origin_total - 1),
            np.nan,
        )
        df['dest_delay_rate'] = np.where(
            dest_total > 1,
            (dest_late - is_late) / (dest_total - 1),
            np.nan,
        )
    else:
        logger.info("Label column missing; airport delay-rate features will be added after labeling.")

    # Great-circle distance
    def _gc_distance(row):
        o = AIRPORT_COORDS.get(row[origin_col])
        d = AIRPORT_COORDS.get(row[dest_col])
        if o and d:
            return haversine_km(o[0], o[1], d[0], d[1])
        return np.nan

    df['route_gc_distance_km'] = df.apply(_gc_distance, axis=1)

    if label_col in df.columns:
        feat_count = 7
        logger.info(f"Added {feat_count} airport features (volume, delay rate, GC distance)")
    else:
        feat_count = 5
        logger.info(f"Added {feat_count} airport features (volume and GC distance)")
    return df


def add_congestion_features(df: pd.DataFrame,
                             dep_col: str = 'scheduled_dep',
                             origin_col: str = 'origin',
                             dest_col: str = 'destination',
                             window_hours: float = 1.0) -> pd.DataFrame:
    """
    Task 3.4 — Traffic congestion at origin and destination.
    
    Counts how many other flights depart/arrive at the same airport within ±window_hours.
    """
    df = df.copy()

    dep_ts = pd.to_datetime(df[dep_col], utc=True, errors='coerce')
    window = pd.Timedelta(hours=window_hours)

    # Origin congestion — flights departing same airport within ±1hr
    df['_dep_ts'] = dep_ts
    df = df.sort_values('_dep_ts')

    origin_congestion = []
    dest_congestion = []

    # Group by origin for departure congestion
    origin_groups = df.groupby(origin_col)
    origin_cong_map = {}
    for airport, group in origin_groups:
        dep_times = group['_dep_ts'].values
        counts = np.zeros(len(dep_times), dtype=int)
        for i in range(len(dep_times)):
            if pd.isna(dep_times[i]):
                continue
            t = dep_times[i]
            counts[i] = int(np.sum(
                (dep_times >= t - window) & (dep_times <= t + window)
            )) - 1  # exclude self
        for idx, cnt in zip(group.index, counts):
            origin_cong_map[idx] = cnt

    df['origin_flights_1hr'] = df.index.map(origin_cong_map).fillna(0).astype(int)

    # Destination congestion — count flights arriving at same dest
    if dest_col in df.columns and 'scheduled_arr' in df.columns:
        arr_ts = pd.to_datetime(df['scheduled_arr'], utc=True, errors='coerce')
        df['_arr_ts'] = arr_ts
        dest_groups = df.groupby(dest_col)
        dest_cong_map = {}
        for airport, group in dest_groups:
            arr_times = group['_arr_ts'].values
            counts = np.zeros(len(arr_times), dtype=int)
            for i in range(len(arr_times)):
                if pd.isna(arr_times[i]):
                    continue
                t = arr_times[i]
                counts[i] = int(np.sum(
                    (arr_times >= t - window) & (arr_times <= t + window)
                )) - 1
            for idx, cnt in zip(group.index, counts):
                dest_cong_map[idx] = cnt
        df['dest_flights_1hr'] = df.index.map(dest_cong_map).fillna(0).astype(int)
        df.drop(columns=['_arr_ts'], inplace=True, errors='ignore')
    else:
        df['dest_flights_1hr'] = 0

    for col in ['origin_flight_count', 'dest_flight_count', 'origin_flights_1hr', 'dest_flights_1hr']:
        if df[col].notna().all():
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('int64')

    df.drop(columns=['_dep_ts'], inplace=True, errors='ignore')
    logger.info("Added 2 congestion features (origin_flights_1hr, dest_flights_1hr)")
    return df


def add_destination_weather(df_flights: pd.DataFrame,
                             df_weather: pd.DataFrame,
                             dest_col: str = 'destination') -> pd.DataFrame:
    """
    Task 2.3 — Add destination airport weather, suffixed with _dest.
    """
    from src.normalization import ScheduleNormalizer
    icao_to_iata = {v: k for k, v in ScheduleNormalizer.IATA_TO_ICAO.items()}

    f_df = df_flights.copy()
    w_df = df_weather.copy()

    if 'scheduled_arr' not in f_df.columns or dest_col not in f_df.columns:
        logger.warning("Cannot add dest weather: missing scheduled_arr or destination columns.")
        return df_flights

    f_df['_dest_merge_time'] = pd.to_datetime(f_df['scheduled_arr'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')
    
    if 'timestamp' not in w_df.columns or 'airport_code' not in w_df.columns:
        logger.warning("Weather data missing timestamp or airport_code.")
        return df_flights

    from src.normalization import ScheduleNormalizer
    iata_to_icao = ScheduleNormalizer.IATA_TO_ICAO  # ATL -> KATL

    raw_dest = f_df[dest_col].astype(str).str.upper().str.strip()
    f_df['_dest_airport'] = raw_dest  # already ICAO

    raw_weather = w_df['airport_code'].astype(str).str.upper().str.strip()
    w_df['_dest_airport'] = raw_weather.map(iata_to_icao).fillna(raw_weather)

    w_df['_weather_time'] = pd.to_datetime(w_df['timestamp'], utc=True, errors='coerce').astype('datetime64[ns, UTC]')

    num_cols = ['wind_speed', 'visibility', 'temperature', 'precipitation']
    for col in num_cols:
        if col in w_df.columns:
            w_df[col] = pd.to_numeric(w_df[col], errors='coerce')

    # Rename weather cols to avoid collision with origin weather
    rename_map = {}
    for col in num_cols:
        if col in w_df.columns:
            rename_map[col] = f'{col}_dest'
    w_dest = w_df.rename(columns=rename_map)

    f_df = f_df.sort_values(['_dest_merge_time', '_dest_airport'])
    w_dest = w_dest.sort_values(['_weather_time', '_dest_airport'])

    tolerance = pd.Timedelta(hours=2)
    merged = pd.merge_asof(
        f_df, w_dest,
        left_on='_dest_merge_time', right_on='_weather_time',
        by='_dest_airport', direction='nearest', tolerance=tolerance,
    )

    # Compute destination weather severity
    dest_weather_cols = [f'{c}_dest' for c in num_cols if f'{c}_dest' in merged.columns]
    if dest_weather_cols:
        wind = merged.get('wind_speed_dest', pd.Series(np.nan, index=merged.index))
        vis = merged.get('visibility_dest', pd.Series(np.nan, index=merged.index))
        precip = merged.get('precipitation_dest', pd.Series(np.nan, index=merged.index))

        conditions = [
            (wind > 30) | (vis < 1.0),
            (wind > 20) | (precip > 0.5) | (vis < 3.0),
            (wind > 15) | (precip > 0.1) | (vis < 5.0),
        ]
        merged['weather_severity_dest'] = np.select(conditions, [3, 2, 1], default=0)
        is_na = pd.DataFrame({'w': wind, 'v': vis, 'p': precip}).isna().all(axis=1)
        merged.loc[is_na, 'weather_severity_dest'] = np.nan

    drop_cols = ['_dest_merge_time', '_weather_time', '_dest_airport', 'airport_code', 'timestamp']
    merged.drop(columns=[c for c in drop_cols if c in merged.columns], inplace=True, errors='ignore')

    n_matched = merged[[f'{c}_dest' for c in num_cols if f'{c}_dest' in merged.columns]].notna().any(axis=1).sum()
    logger.info(f"Destination weather matched for {n_matched}/{len(merged)} flights.")
    return merged


def enrich_all(df: pd.DataFrame,
               df_weather: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run all feature enrichment steps in order."""
    logger.info("═" * 60)
    logger.info("Running feature enrichment pipeline...")

    df = add_temporal_features(df)
    df = add_airport_features(df)
    df = add_congestion_features(df)

    if df_weather is not None and not df_weather.empty:
        df = add_destination_weather(df, df_weather)

    logger.info("Feature enrichment complete. Final shape: %s", df.shape)
    return df
