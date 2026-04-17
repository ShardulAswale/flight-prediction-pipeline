from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils import ensure_dir, load_config, set_seed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrajectoryThresholds:
    ground_altitude_threshold_m: float = 500.0
    min_altitude_span_m: float = 1000.0
    min_altitude_step_m: float = 50.0
    min_climb_descent_steps: int = 3
    min_clean_rows: int = 8
    min_route_distance_km: float = 1.0
    min_meaningful_route_km: float = 5.0
    max_point_speed_mps: float = 400.0
    max_altitude_jump_split_m: float = 5000.0
    altitude_spike_threshold_m: float = 1500.0
    stationary_dwell_split_seconds: int = 20 * 60
    stationary_dwell_distance_km: float = 0.25
    default_max_proxy_attempts: int = 25


@dataclass
class TrajectoryNotebookContext:
    project_root: Path
    config: dict[str, Any]
    adsb_input_path: Path
    traj_path: Path
    thresholds: TrajectoryThresholds

    @property
    def max_gap_minutes(self) -> int:
        return int(self.config.get('trajectory', {}).get('max_gap_minutes', 15))

    @property
    def seed(self) -> int:
        return int(self.config.get('seed', 42))


@dataclass(frozen=True)
class PrototypeControls:
    sample_rows: int = 2_000_000
    random_state: int = 42
    manual_icao24: str | None = None
    manual_callsign: str | None = None
    max_gap_minutes: int = 15
    max_proxy_attempts: int = 25
    use_full_source: bool = False
    month_prefix: str | None = None
    min_rows_per_proxy: int = 20
    full_source_window_groups: int = 4
    full_source_step_groups: int = 2
    full_source_top_candidates: int = 25


@dataclass
class PrototypeResult:
    raw_segment: pd.DataFrame
    cleaned_segment: pd.DataFrame | None
    ranking: pd.DataFrame
    summary: pd.DataFrame
    prototype_status: str


_CONTEXT_CACHE: dict[str, TrajectoryNotebookContext] = {}
_INSPECT_CACHE: dict[tuple[Any, ...], dict[str, Any]] = {}
_SAMPLE_CACHE: dict[tuple[Any, ...], pd.DataFrame] = {}
_PROTOTYPE_CACHE: dict[tuple[Any, ...], PrototypeResult] = {}
_NOTEBOOK_CONTROLS_CACHE: dict[str, PrototypeControls] = {}


def _stat_token(path: Path) -> int:
    if not path.exists():
        return -1
    return path.stat().st_mtime_ns


def _copy_result(result: PrototypeResult) -> PrototypeResult:
    return PrototypeResult(
        raw_segment=result.raw_segment.copy(deep=True),
        cleaned_segment=None if result.cleaned_segment is None else result.cleaned_segment.copy(deep=True),
        ranking=result.ranking.copy(deep=True),
        summary=result.summary.copy(deep=True),
        prototype_status=result.prototype_status,
    )


def detect_time_column(columns: Iterable[str]) -> str | None:
    for candidate in ['timestamp', 'time', 'last_contact', 'time_position']:
        if candidate in columns:
            return candidate
    return None


def choose_altitude_column(df: pd.DataFrame) -> str:
    if 'geo_altitude' in df.columns and not df['geo_altitude'].isna().all():
        return 'geo_altitude'
    if 'baro_altitude' in df.columns:
        return 'baro_altitude'
    return 'altitude'


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    radius = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))


def build_notebook_context(project_root: str | Path) -> TrajectoryNotebookContext:
    resolved_root = Path(project_root).resolve()
    cache_key = str(resolved_root)
    if cache_key in _CONTEXT_CACHE:
        return _CONTEXT_CACHE[cache_key]

    config = load_config(resolved_root / 'configs' / 'config.yaml')
    set_seed(config.get('seed', 42))
    paths_cfg = config['paths']
    ingest_cfg = config.get('ingestion', {})

    adsb_input_path = resolved_root / ingest_cfg.get('adsb_combined_file', 'data/processed/adsb_combined.parquet')
    traj_path = resolved_root / paths_cfg['processed_data_dir'] / paths_cfg['trajectories_file']

    ctx = TrajectoryNotebookContext(
        project_root=resolved_root,
        config=config,
        adsb_input_path=adsb_input_path,
        traj_path=traj_path,
        thresholds=TrajectoryThresholds(),
    )
    _CONTEXT_CACHE[cache_key] = ctx
    return ctx


def make_prototype_controls(
    ctx: TrajectoryNotebookContext,
    *,
    sample_rows: int = 2_000_000,
    random_state: int | None = None,
    manual_icao24: str | None = None,
    manual_callsign: str | None = None,
    max_gap_minutes: int | None = None,
    max_proxy_attempts: int | None = None,
    use_full_source: bool = False,
    month_prefix: str | None = None,
    min_rows_per_proxy: int = 20,
    full_source_window_groups: int = 4,
    full_source_step_groups: int = 2,
    full_source_top_candidates: int = 25,
) -> PrototypeControls:
    return PrototypeControls(
        sample_rows=int(sample_rows),
        random_state=ctx.seed if random_state is None else int(random_state),
        manual_icao24=manual_icao24,
        manual_callsign=manual_callsign,
        max_gap_minutes=ctx.max_gap_minutes if max_gap_minutes is None else int(max_gap_minutes),
        max_proxy_attempts=(ctx.thresholds.default_max_proxy_attempts if max_proxy_attempts is None else int(max_proxy_attempts)),
        use_full_source=bool(use_full_source),
        month_prefix=None if month_prefix in {None, ''} else str(month_prefix),
        min_rows_per_proxy=int(min_rows_per_proxy),
        full_source_window_groups=int(full_source_window_groups),
        full_source_step_groups=int(full_source_step_groups),
        full_source_top_candidates=int(full_source_top_candidates),
    )


def cache_notebook_controls(ctx: TrajectoryNotebookContext, controls: PrototypeControls) -> PrototypeControls:
    _NOTEBOOK_CONTROLS_CACHE[str(ctx.project_root)] = controls
    return controls


def get_notebook_controls(ctx: TrajectoryNotebookContext) -> PrototypeControls:
    return _NOTEBOOK_CONTROLS_CACHE.get(str(ctx.project_root), make_prototype_controls(ctx))


def inspect_adsb_source(ctx: TrajectoryNotebookContext, preview_rows: int = 5) -> dict[str, Any]:
    cache_key = (str(ctx.adsb_input_path), _stat_token(ctx.adsb_input_path), int(preview_rows))
    if cache_key in _INSPECT_CACHE:
        cached = _INSPECT_CACHE[cache_key]
        return {**cached, 'preview_df': cached['preview_df'].copy(deep=True)}

    if not ctx.adsb_input_path.exists():
        result = {
            'exists': False,
            'input_path': ctx.adsb_input_path,
            'row_count': 0,
            'column_count': 0,
            'time_column': None,
            'preview_df': pd.DataFrame(),
        }
        _INSPECT_CACHE[cache_key] = result
        return result

    parquet_file = pq.ParquetFile(ctx.adsb_input_path)
    columns = parquet_file.schema.names
    preview_batch = next(parquet_file.iter_batches(batch_size=preview_rows), None)
    preview_df = preview_batch.to_pandas() if preview_batch is not None else pd.DataFrame()
    result = {
        'exists': True,
        'input_path': ctx.adsb_input_path,
        'row_count': parquet_file.metadata.num_rows,
        'column_count': len(columns),
        'time_column': detect_time_column(columns),
        'preview_df': preview_df,
    }
    _INSPECT_CACHE[cache_key] = {**result, 'preview_df': preview_df.copy(deep=True)}
    return result


def normalize_adsb_frame(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    df = df.copy()

    if 'icao24' in df.columns:
        df['icao24'] = df['icao24'].astype('string').str.strip().str.lower()
    else:
        df['icao24'] = pd.Series(dtype='string')

    if 'callsign' in df.columns:
        df['callsign'] = df['callsign'].astype('string').str.strip().str.upper()
        df['callsign'] = df['callsign'].replace({'': pd.NA, 'NAN': pd.NA, 'NONE': pd.NA})
    else:
        df['callsign'] = pd.Series(dtype='string')

    df['callsign_clean'] = df['callsign']

    if 'timestamp' not in df.columns and time_col in df.columns:
        df['timestamp'] = df[time_col]

    numeric_cols = ['timestamp', 'latitude', 'longitude', 'geo_altitude', 'baro_altitude', 'velocity', 'true_track', 'vertical_rate', 'time_position', 'last_contact']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'on_ground' in df.columns:
        df['on_ground'] = df['on_ground'].fillna(False).astype(bool)
    else:
        df['on_ground'] = False

    if 'latitude' not in df.columns:
        df['latitude'] = np.nan
    if 'longitude' not in df.columns:
        df['longitude'] = np.nan

    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s', utc=True, errors='coerce')
    return df


def filter_month_rows(df: pd.DataFrame, month_prefix: str | None) -> pd.DataFrame:
    if month_prefix in {None, ''} or 'sample_date' not in df.columns:
        return df
    sample_date = df['sample_date'].astype('string')
    return df.loc[sample_date.str.startswith(str(month_prefix), na=False)].copy()


def compute_elapsed_minutes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'timestamp_dt' in df.columns and df['timestamp_dt'].notna().any():
        start_ts = df['timestamp_dt'].dropna().min()
        df['elapsed_minutes'] = (df['timestamp_dt'] - start_ts).dt.total_seconds() / 60.0
    else:
        df['elapsed_minutes'] = np.arange(len(df), dtype=float)
    return df


def valid_coordinate_mask(df: pd.DataFrame) -> pd.Series:
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return pd.Series(False, index=df.index)
    lat_ok = df['latitude'].between(-90, 90, inclusive='both')
    lon_ok = df['longitude'].between(-180, 180, inclusive='both')
    return lat_ok & lon_ok

def drop_missing_timestamp_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if 'timestamp' not in df.columns or 'timestamp_dt' not in df.columns:
        return df.copy(), 0
    missing_mask = df['timestamp'].isna() | df['timestamp_dt'].isna()
    return df.loc[~missing_mask].reset_index(drop=True), int(missing_mask.sum())


def drop_invalid_coordinate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return df.copy(), 0

    lat_present = df['latitude'].notna()
    lon_present = df['longitude'].notna()
    invalid_mask = (
        (lat_present & ~df['latitude'].between(-90, 90, inclusive='both'))
        | (lon_present & ~df['longitude'].between(-180, 180, inclusive='both'))
    )
    return df.loc[~invalid_mask].reset_index(drop=True), int(invalid_mask.sum())


def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df_out = df.drop_duplicates().reset_index(drop=True)
    return df_out, before - len(df_out)


def drop_consecutive_stale_points(df: pd.DataFrame, altitude_col: str) -> tuple[pd.DataFrame, int]:
    if df.empty:
        return df.copy(), 0

    compare_cols = [c for c in ['latitude', 'longitude', altitude_col, 'on_ground'] if c in df.columns]
    if not compare_cols:
        return df.copy(), 0

    prev = df[compare_cols].shift(1)
    same_mask = pd.Series(True, index=df.index)
    for col in compare_cols:
        current = df[col]
        previous = prev[col]
        if pd.api.types.is_numeric_dtype(current):
            col_same = np.isclose(current.astype(float), previous.astype(float), equal_nan=True)
            col_same = pd.Series(col_same, index=df.index)
        else:
            col_same = current.fillna('__NA__').astype(str).eq(previous.fillna('__NA__').astype(str))
        same_mask &= col_same

    same_mask &= df['timestamp'].diff().fillna(0) > 0
    if not same_mask.empty:
        same_mask.iloc[0] = False
    return df.loc[~same_mask].reset_index(drop=True), int(same_mask.sum())


def remove_altitude_spikes(df: pd.DataFrame, altitude_col: str, spike_threshold_m: float) -> tuple[pd.DataFrame, int]:
    if altitude_col not in df.columns or len(df) < 2:
        return df.copy(), 0

    altitude = pd.to_numeric(df[altitude_col], errors='coerce')
    spike_mask = pd.Series(False, index=df.index)

    if len(df) >= 3:
        first_three = altitude.iloc[:3]
        if first_three.notna().all():
            if abs(first_three.iloc[0] - first_three.iloc[1]) > spike_threshold_m and abs(first_three.iloc[1] - first_three.iloc[2]) < spike_threshold_m * 0.35:
                spike_mask.iloc[0] = True

        last_three = altitude.iloc[-3:]
        if last_three.notna().all():
            if abs(last_three.iloc[-1] - last_three.iloc[-2]) > spike_threshold_m and abs(last_three.iloc[-2] - last_three.iloc[-3]) < spike_threshold_m * 0.35:
                spike_mask.iloc[-1] = True

        for i in range(1, len(df) - 1):
            prev_alt = altitude.iloc[i - 1]
            curr_alt = altitude.iloc[i]
            next_alt = altitude.iloc[i + 1]
            if pd.isna(prev_alt) or pd.isna(curr_alt) or pd.isna(next_alt):
                continue
            is_needle = (
                abs(curr_alt - prev_alt) > spike_threshold_m
                and abs(curr_alt - next_alt) > spike_threshold_m
                and abs(prev_alt - next_alt) < spike_threshold_m * 0.35
            )
            if is_needle:
                spike_mask.iloc[i] = True

    return df.loc[~spike_mask].reset_index(drop=True), int(spike_mask.sum())


def select_segment_callsign(df_segment: pd.DataFrame, fallback: str = 'UNKNOWN') -> str:
    for col in ['callsign_clean', 'callsign']:
        if col in df_segment.columns and df_segment[col].notna().any():
            return str(df_segment[col].mode(dropna=True).iloc[0])
    return fallback


def is_informative_callsign(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().upper()
    if text in {'', 'UNKNOWN', 'NONE', 'NAN', 'NULL'}:
        return False
    if len(text) < 3:
        return False
    if text.isdigit():
        return False
    if not any(ch.isalpha() for ch in text):
        return False
    return True


def strip_helper_columns(df: pd.DataFrame) -> pd.DataFrame:
    helper_cols = ['callsign_clean', 'timestamp_dt', 'elapsed_minutes', 'is_valid_coord', 'segment_id_local']
    drop_cols = [col for col in helper_cols if col in df.columns]
    return df.drop(columns=drop_cols, errors='ignore')


def split_trace_into_segments(
    df: pd.DataFrame,
    altitude_col: str,
    thresholds: TrajectoryThresholds,
    max_gap_seconds: int,
    *,
    split_on_time_gap: bool = True,
) -> list[pd.DataFrame]:
    if df.empty:
        return []

    df = df.sort_values('timestamp').reset_index(drop=True).copy()
    segment_ids = [0]
    current_segment = 0

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        split_here = False

        dt = curr['timestamp'] - prev['timestamp']
        if pd.isna(dt) or dt <= 0 or (split_on_time_gap and dt > max_gap_seconds):
            split_here = True

        prev_call = prev.get('callsign_clean')
        curr_call = curr.get('callsign_clean')
        if not split_here and pd.notna(prev_call) and pd.notna(curr_call) and str(prev_call) != str(curr_call):
            split_here = True

        prev_has_coords = pd.notna(prev.get('latitude')) and pd.notna(prev.get('longitude'))
        curr_has_coords = pd.notna(curr.get('latitude')) and pd.notna(curr.get('longitude'))
        distance_km = np.nan
        if prev_has_coords and curr_has_coords:
            distance_km = haversine_km(prev['latitude'], prev['longitude'], curr['latitude'], curr['longitude'])

        if not split_here and prev_has_coords and curr_has_coords and 0 < dt <= max_gap_seconds:
            derived_speed_mps = (distance_km * 1000.0) / float(dt)
            if derived_speed_mps > thresholds.max_point_speed_mps:
                split_here = True

        prev_alt = prev.get(altitude_col)
        curr_alt = curr.get(altitude_col)
        if not split_here and pd.notna(prev_alt) and pd.notna(curr_alt) and dt <= 60:
            if abs(float(curr_alt) - float(prev_alt)) > thresholds.max_altitude_jump_split_m:
                split_here = True

        prev_groundish = bool(prev.get('on_ground', False)) or (pd.notna(prev_alt) and float(prev_alt) <= thresholds.ground_altitude_threshold_m)
        curr_groundish = bool(curr.get('on_ground', False)) or (pd.notna(curr_alt) and float(curr_alt) <= thresholds.ground_altitude_threshold_m)
        if not split_here and dt >= thresholds.stationary_dwell_split_seconds and prev_groundish and curr_groundish:
            if pd.isna(distance_km) or distance_km <= thresholds.stationary_dwell_distance_km:
                split_here = True

        if split_here:
            current_segment += 1
        segment_ids.append(current_segment)

    df['segment_id_local'] = segment_ids
    return [segment_df.reset_index(drop=True).copy() for _, segment_df in df.groupby('segment_id_local', sort=False)]


def compute_route_distance_km(df: pd.DataFrame) -> float:
    if df.empty or not {'latitude', 'longitude'}.issubset(df.columns):
        return 0.0

    valid = df.dropna(subset=['latitude', 'longitude']).sort_values('timestamp')
    if len(valid) < 2:
        return 0.0

    distance_km = 0.0
    for i in range(1, len(valid)):
        distance_km += haversine_km(
            valid.iloc[i - 1]['latitude'],
            valid.iloc[i - 1]['longitude'],
            valid.iloc[i]['latitude'],
            valid.iloc[i]['longitude'],
        )
    return float(distance_km)

def evaluate_clean_segment(df_raw_segment: pd.DataFrame, altitude_col: str, thresholds: TrajectoryThresholds) -> tuple[pd.DataFrame, dict[str, Any]]:
    df_segment = df_raw_segment.copy().sort_values('timestamp').reset_index(drop=True)
    raw_rows = len(df_segment)

    df_segment, missing_timestamp_removed = drop_missing_timestamp_rows(df_segment)
    df_segment, invalid_coordinate_rows_removed = drop_invalid_coordinate_rows(df_segment)
    df_segment, exact_removed = drop_exact_duplicates(df_segment)
    df_segment, stale_removed = drop_consecutive_stale_points(df_segment, altitude_col)
    df_segment, spike_removed = remove_altitude_spikes(df_segment, altitude_col, thresholds.altitude_spike_threshold_m)
    df_segment = df_segment.sort_values('timestamp').reset_index(drop=True)
    df_segment = compute_elapsed_minutes(df_segment)

    if 'latitude' in df_segment.columns and 'longitude' in df_segment.columns:
        df_segment['is_valid_coord'] = valid_coordinate_mask(df_segment)
        path_df = df_segment.loc[df_segment['is_valid_coord']].copy().reset_index(drop=True)
    else:
        df_segment['is_valid_coord'] = False
        path_df = df_segment.iloc[0:0].copy()

    path_df = compute_elapsed_minutes(path_df)
    clean_rows = len(df_segment)
    valid_map_points = len(path_df)
    unique_coord_points = 0
    route_distance_km = 0.0
    if valid_map_points > 0:
        unique_coord_points = len(path_df[['latitude', 'longitude']].drop_duplicates())
        route_distance_km = compute_route_distance_km(path_df)

    altitudes = pd.to_numeric(df_segment.get(altitude_col), errors='coerce')
    alt_non_na = altitudes.dropna().reset_index(drop=True)
    altitude_span = float(alt_non_na.max() - alt_non_na.min()) if not alt_non_na.empty else 0.0
    duration_minutes = float(df_segment['elapsed_minutes'].max()) if 'elapsed_minutes' in df_segment.columns and not df_segment.empty else 0.0

    start_altitude = float(altitudes.dropna().iloc[0]) if altitudes.notna().any() else np.nan
    end_altitude = float(altitudes.dropna().iloc[-1]) if altitudes.notna().any() else np.nan

    starts_groundish = False
    ends_groundish = False
    if not df_segment.empty:
        first_row = df_segment.iloc[0]
        last_row = df_segment.iloc[-1]
        starts_groundish = bool(first_row.get('on_ground', False)) or (pd.notna(start_altitude) and start_altitude <= thresholds.ground_altitude_threshold_m)
        ends_groundish = bool(last_row.get('on_ground', False)) or (pd.notna(end_altitude) and end_altitude <= thresholds.ground_altitude_threshold_m)

    has_sustained_climb = False
    has_sustained_descent = False
    if len(alt_non_na) >= thresholds.min_climb_descent_steps + 1:
        peak_pos = int(alt_non_na.idxmax())
        climb_diffs = alt_non_na.iloc[: peak_pos + 1].diff().dropna()
        descent_diffs = alt_non_na.iloc[peak_pos:].diff().dropna()
        has_sustained_climb = int((climb_diffs > thresholds.min_altitude_step_m).sum()) >= thresholds.min_climb_descent_steps
        has_sustained_descent = int((descent_diffs < -thresholds.min_altitude_step_m).sum()) >= thresholds.min_climb_descent_steps

    is_mappable = bool(valid_map_points >= 2 and unique_coord_points >= 2 and route_distance_km > 0.1)
    has_altitude_spike = spike_removed > 0

    invalid_too_short = clean_rows < thresholds.min_clean_rows
    invalid_no_geometry = valid_map_points < 2 or unique_coord_points < 2
    invalid_no_movement = route_distance_km < thresholds.min_route_distance_km
    spike_dominated = has_altitude_spike and altitude_span < thresholds.min_altitude_span_m and route_distance_km < thresholds.min_meaningful_route_km
    ground_only = starts_groundish and ends_groundish and altitude_span < thresholds.min_altitude_span_m and route_distance_km >= thresholds.min_route_distance_km
    airborne_only = (
        (not starts_groundish)
        and (not ends_groundish)
        and route_distance_km >= thresholds.min_meaningful_route_km
        and altitude_span >= thresholds.min_altitude_span_m
        and (has_sustained_climb or has_sustained_descent)
    )
    coherent_departure = (
        starts_groundish
        and (not ends_groundish)
        and route_distance_km >= thresholds.min_meaningful_route_km
        and altitude_span >= thresholds.min_altitude_span_m
        and has_sustained_climb
        and not has_altitude_spike
    )
    coherent_arrival = (
        (not starts_groundish)
        and ends_groundish
        and route_distance_km >= thresholds.min_meaningful_route_km
        and altitude_span >= thresholds.min_altitude_span_m
        and has_sustained_descent
        and not has_altitude_spike
    )
    coherent_midflight = (
        (not starts_groundish)
        and (not ends_groundish)
        and route_distance_km >= thresholds.min_meaningful_route_km
        and altitude_span >= thresholds.min_altitude_span_m
        and (has_sustained_climb or has_sustained_descent)
        and not has_altitude_spike
    )

    is_full_flight = bool(
        clean_rows >= thresholds.min_clean_rows
        and is_mappable
        and route_distance_km >= thresholds.min_meaningful_route_km
        and altitude_span >= thresholds.min_altitude_span_m
        and starts_groundish
        and ends_groundish
        and has_sustained_climb
        and has_sustained_descent
        and not has_altitude_spike
    )

    if spike_dominated or (has_altitude_spike and clean_rows < thresholds.min_clean_rows):
        quality_status = 'invalid_spike'
    elif invalid_too_short:
        quality_status = 'invalid_too_short'
    elif invalid_no_geometry:
        quality_status = 'invalid_no_geometry'
    elif invalid_no_movement:
        quality_status = 'invalid_no_movement'
    elif is_full_flight:
        quality_status = 'full_flight'
    elif ground_only:
        quality_status = 'ground_only'
    elif coherent_departure:
        quality_status = 'partial_end_missing'
    elif coherent_arrival:
        quality_status = 'partial_start_missing'
    elif coherent_midflight:
        quality_status = 'partial_both_ends'
    elif airborne_only:
        quality_status = 'airborne_only'
    elif not starts_groundish and not ends_groundish:
        quality_status = 'airborne_only'
    elif not starts_groundish:
        quality_status = 'partial_start_missing'
    elif not ends_groundish:
        quality_status = 'partial_end_missing'
    else:
        quality_status = 'sparse_route'

    quality_score = (
        min(route_distance_km / max(thresholds.min_meaningful_route_km, 1.0), 1.0) * 0.25
        + min(altitude_span / max(thresholds.min_altitude_span_m, 1.0), 1.0) * 0.20
        + 0.15 * float(starts_groundish)
        + 0.15 * float(ends_groundish)
        + 0.10 * float(is_mappable)
        + 0.075 * float(has_sustained_climb)
        + 0.075 * float(has_sustained_descent)
        + min(clean_rows / 40.0, 1.0) * 0.05
        - 0.15 * float(has_altitude_spike)
    )
    quality_score = float(np.clip(quality_score, 0.0, 1.0))
    if quality_status.startswith('invalid_'):
        quality_score = min(quality_score, 0.24)

    cleaned_output = path_df.copy().reset_index(drop=True)
    cleaned_output = strip_helper_columns(cleaned_output)
    if quality_status.startswith('invalid_'):
        cleaned_output = cleaned_output.iloc[0:0].copy()

    summary = {
        'raw_rows': raw_rows,
        'clean_rows': clean_rows,
        'map_rows': len(path_df),
        'missing_timestamp_rows_removed': missing_timestamp_removed,
        'invalid_coordinate_rows_removed': invalid_coordinate_rows_removed,
        'exact_duplicates_removed': exact_removed,
        'stale_rows_removed': stale_removed,
        'altitude_spikes_removed': spike_removed,
        'duration_minutes': duration_minutes,
        'valid_map_points': valid_map_points,
        'route_distance_km': route_distance_km,
        'altitude_span_m': altitude_span,
        'start_altitude_m': start_altitude,
        'end_altitude_m': end_altitude,
        'starts_groundish': starts_groundish,
        'ends_groundish': ends_groundish,
        'has_sustained_climb': has_sustained_climb,
        'has_sustained_descent': has_sustained_descent,
        'has_altitude_spike': has_altitude_spike,
        'is_mappable': is_mappable,
        'is_full_flight': is_full_flight,
        'trajectory_quality_status': quality_status,
        'trajectory_quality_score': quality_score,
        'kept_in_output': not quality_status.startswith('invalid_'),
    }
    return cleaned_output, summary


def get_flight_proxy_candidates(
    df: pd.DataFrame,
    *,
    manual_icao24: str | None = None,
    manual_callsign: str | None = None,
    random_state: int | None = None,
    min_rows: int = 20,
) -> tuple[pd.DataFrame, str]:
    if df.empty:
        raise ValueError('No ADS-B rows are available for trajectory candidate selection.')

    working = df[df['icao24'].notna()].copy()
    working['callsign_clean'] = working['callsign_clean'].fillna('UNKNOWN')
    if manual_callsign is None:
        working = working[working['callsign_clean'].map(is_informative_callsign)].copy()
    candidate_counts = working.groupby(['icao24', 'callsign_clean'], dropna=False).size().reset_index(name='rows')
    candidate_counts = candidate_counts[candidate_counts['rows'] >= min_rows].copy()

    if manual_icao24 is not None:
        manual_icao24 = str(manual_icao24).strip().lower()
        candidate_counts = candidate_counts[candidate_counts['icao24'] == manual_icao24]
    if manual_callsign is not None:
        manual_callsign = str(manual_callsign).strip().upper()
        candidate_counts = candidate_counts[candidate_counts['callsign_clean'] == manual_callsign]

    if candidate_counts.empty:
        raise ValueError('No sufficiently dense flight proxies were found for the requested filter.')

    if manual_icao24 is not None or manual_callsign is not None:
        candidate_counts = candidate_counts.sort_values(['rows', 'icao24', 'callsign_clean'], ascending=[False, True, True]).reset_index(drop=True)
        selection_reason = 'manual filter'
    else:
        candidate_counts = candidate_counts.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        selection_reason = 'random dense candidates'

    return candidate_counts, selection_reason


def choose_diagnostic_fallback(ranking: pd.DataFrame) -> pd.Series | None:
    if ranking.empty:
        return None

    working = ranking.copy()
    status = working['trajectory_quality_status'].astype(str)
    acceptable = working[
        ~status.str.startswith('invalid_')
        & ~status.isin({'ground_only'})
    ].copy()
    if acceptable.empty:
        return None

    fallback_priority = {
        'partial_end_missing': 0,
        'partial_start_missing': 1,
        'partial_both_ends': 2,
        'airborne_only': 3,
        'sparse_route': 4,
    }
    acceptable['fallback_priority'] = acceptable['trajectory_quality_status'].map(fallback_priority).fillna(9)
    acceptable = acceptable.sort_values(
        ['fallback_priority', 'trajectory_quality_score', 'route_distance_km', 'clean_rows'],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)
    return acceptable.iloc[0]


def summarize_middle_gaps(df: pd.DataFrame, gap_threshold_seconds: int) -> dict[str, float | int | bool]:
    if df.empty or 'timestamp' not in df.columns:
        return {
            'middle_gap_count': 0,
            'largest_gap_minutes': 0.0,
            'total_gap_minutes': 0.0,
            'gap_fraction_of_flight': 0.0,
            'has_middle_gap': False,
            'max_inter_ping_seconds': 0.0,
            'median_inter_ping_seconds': 0.0,
            'ping_interval_cv': 0.0,
        }

    ts = pd.to_numeric(df['timestamp'], errors='coerce').dropna().sort_values()
    if len(ts) < 2:
        return {
            'middle_gap_count': 0,
            'largest_gap_minutes': 0.0,
            'total_gap_minutes': 0.0,
            'gap_fraction_of_flight': 0.0,
            'has_middle_gap': False,
            'max_inter_ping_seconds': 0.0,
            'median_inter_ping_seconds': 0.0,
            'ping_interval_cv': 0.0,
        }

    diffs = ts.diff().dropna()
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.empty:
        return {
            'middle_gap_count': 0,
            'largest_gap_minutes': 0.0,
            'total_gap_minutes': 0.0,
            'gap_fraction_of_flight': 0.0,
            'has_middle_gap': False,
            'max_inter_ping_seconds': 0.0,
            'median_inter_ping_seconds': 0.0,
            'ping_interval_cv': 0.0,
        }

    large_gaps = positive_diffs[positive_diffs > gap_threshold_seconds]
    max_inter_ping_seconds = float(positive_diffs.max()) if not positive_diffs.empty else 0.0
    median_inter_ping_seconds = float(positive_diffs.median()) if not positive_diffs.empty else 0.0
    if len(positive_diffs) >= 2 and float(positive_diffs.mean()) > 0:
        ping_interval_cv = float(positive_diffs.std(ddof=0) / positive_diffs.mean())
    else:
        ping_interval_cv = 0.0
    total_gap_minutes = float(large_gaps.sum() / 60.0) if not large_gaps.empty else 0.0
    total_duration_minutes = float((ts.iloc[-1] - ts.iloc[0]) / 60.0) if len(ts) >= 2 else 0.0
    gap_fraction_of_flight = float(total_gap_minutes / total_duration_minutes) if total_duration_minutes > 0 else 0.0
    return {
        'middle_gap_count': int(len(large_gaps)),
        'largest_gap_minutes': float(positive_diffs.max() / 60.0),
        'total_gap_minutes': total_gap_minutes,
        'gap_fraction_of_flight': gap_fraction_of_flight,
        'has_middle_gap': bool(len(large_gaps) > 0),
        'max_inter_ping_seconds': max_inter_ping_seconds,
        'median_inter_ping_seconds': median_inter_ping_seconds,
        'ping_interval_cv': ping_interval_cv,
    }


def _sample_adsb_source(ctx: TrajectoryNotebookContext, controls: PrototypeControls) -> pd.DataFrame:
    cache_key = (str(ctx.adsb_input_path), _stat_token(ctx.adsb_input_path), controls.sample_rows, controls.month_prefix)
    if cache_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[cache_key].copy(deep=True)

    parquet_file = pq.ParquetFile(ctx.adsb_input_path)
    columns = parquet_file.schema.names
    time_col = detect_time_column(columns)
    if time_col is None:
        raise ValueError('No usable time column found in ADS-B dataset.')

    sampled_parts: list[pd.DataFrame] = []
    rows_collected = 0
    target_rows = max(controls.sample_rows, 100_000)
    for batch in parquet_file.iter_batches(batch_size=min(controls.sample_rows, 500_000), columns=columns):
        df_batch = normalize_adsb_frame(batch.to_pandas(), time_col)
        df_batch = filter_month_rows(df_batch, controls.month_prefix)
        if df_batch.empty:
            continue
        sampled_parts.append(df_batch)
        rows_collected += len(df_batch)
        if rows_collected >= target_rows:
            break

    if not sampled_parts:
        raise ValueError('No ADS-B rows were read from the source parquet.')

    df_sample = pd.concat(sampled_parts, ignore_index=True)
    df_sample = df_sample[df_sample['icao24'].notna() & df_sample['timestamp'].notna()].copy()
    _SAMPLE_CACHE[cache_key] = df_sample.copy(deep=True)
    return df_sample


def _get_proxy_candidates_from_full_source(ctx: TrajectoryNotebookContext, controls: PrototypeControls) -> tuple[pd.DataFrame, str]:
    parquet_file = pq.ParquetFile(ctx.adsb_input_path)
    columns = parquet_file.schema.names
    time_col = detect_time_column(columns)
    if time_col is None:
        raise ValueError('No usable time column found in ADS-B dataset.')

    count_by_proxy: Counter[tuple[str, str]] = Counter()
    needed_columns = [col for col in ['icao24', 'callsign', 'sample_date', time_col, 'timestamp'] if col in columns]

    for batch in parquet_file.iter_batches(batch_size=500_000, columns=needed_columns):
        df_batch = normalize_adsb_frame(batch.to_pandas(), time_col)
        df_batch = filter_month_rows(df_batch, controls.month_prefix)
        if df_batch.empty:
            continue
        df_batch = df_batch[df_batch['icao24'].notna() & df_batch['timestamp'].notna()].copy()
        if controls.manual_icao24 is not None:
            df_batch = df_batch[df_batch['icao24'] == str(controls.manual_icao24).strip().lower()]
        if controls.manual_callsign is not None:
            df_batch = df_batch[df_batch['callsign_clean'] == str(controls.manual_callsign).strip().upper()]
        elif 'callsign_clean' in df_batch.columns:
            df_batch = df_batch[df_batch['callsign_clean'].map(is_informative_callsign)]
        if df_batch.empty:
            continue

        grouped = df_batch.groupby(['icao24', 'callsign_clean'], dropna=False).size()
        count_by_proxy.update({(str(idx[0]), str(idx[1])): int(value) for idx, value in grouped.items()})

    if not count_by_proxy:
        raise ValueError('No sufficiently dense flight proxies were found in the requested full-source scan.')

    rows = [
        {'icao24': icao24, 'callsign_clean': callsign, 'rows': row_count}
        for (icao24, callsign), row_count in count_by_proxy.items()
    ]
    candidate_counts = pd.DataFrame(rows)
    candidate_counts = candidate_counts[candidate_counts['rows'] >= controls.min_rows_per_proxy].copy()

    if controls.manual_icao24 is not None:
        candidate_counts = candidate_counts[candidate_counts['icao24'] == str(controls.manual_icao24).strip().lower()]
    if controls.manual_callsign is not None:
        candidate_counts = candidate_counts[candidate_counts['callsign_clean'] == str(controls.manual_callsign).strip().upper()]

    if candidate_counts.empty:
        raise ValueError('No sufficiently dense flight proxies were found for the requested month-wide scan.')

    candidate_counts = candidate_counts.sort_values(['rows', 'icao24', 'callsign_clean'], ascending=[False, True, True]).reset_index(drop=True)
    month_label = controls.month_prefix or 'all_available_dates'
    selection_reason = f'full-source month scan ({month_label})'
    return candidate_counts, selection_reason


def _load_proxy_traces_from_full_source(
    ctx: TrajectoryNotebookContext,
    controls: PrototypeControls,
    proxy_candidates: pd.DataFrame,
) -> dict[tuple[str, str], pd.DataFrame]:
    if proxy_candidates.empty:
        return {}

    parquet_file = pq.ParquetFile(ctx.adsb_input_path)
    columns = parquet_file.schema.names
    time_col = detect_time_column(columns)
    if time_col is None:
        raise ValueError('No usable time column found in ADS-B dataset.')

    target_proxies = {
        (str(row['icao24']), str(row['callsign_clean']))
        for _, row in proxy_candidates.iterrows()
    }
    collected_parts: dict[tuple[str, str], list[pd.DataFrame]] = {key: [] for key in target_proxies}

    for batch in parquet_file.iter_batches(batch_size=500_000, columns=columns):
        df_batch = normalize_adsb_frame(batch.to_pandas(), time_col)
        df_batch = filter_month_rows(df_batch, controls.month_prefix)
        if df_batch.empty:
            continue

        df_batch = df_batch[df_batch['icao24'].notna() & df_batch['timestamp'].notna()].copy()
        if df_batch.empty:
            continue

        if controls.manual_callsign is None:
            df_batch = df_batch[df_batch['callsign_clean'].map(is_informative_callsign)]
        if df_batch.empty:
            continue

        key_series = list(zip(df_batch['icao24'].astype(str), df_batch['callsign_clean'].astype(str)))
        mask = pd.Series([key in target_proxies for key in key_series], index=df_batch.index)
        matched = df_batch.loc[mask].copy()
        if matched.empty:
            continue

        for proxy_key, proxy_df in matched.groupby(['icao24', 'callsign_clean'], dropna=False):
            key = (str(proxy_key[0]), str(proxy_key[1]))
            if key in collected_parts:
                collected_parts[key].append(proxy_df.copy())

    traces: dict[tuple[str, str], pd.DataFrame] = {}
    for key, parts in collected_parts.items():
        if parts:
            traces[key] = pd.concat(parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    return traces


def _result_from_trace(
    *,
    ctx: TrajectoryNotebookContext,
    controls: PrototypeControls,
    df_candidate: pd.DataFrame,
    selected_icao24: str,
    selected_callsign: str,
    selection_reason: str,
    candidate_attempt: int,
    prototype_status: str,
    full_flight_candidates: int,
    diagnostic_fallback_used: bool,
    extra_summary: dict[str, Any] | None = None,
) -> PrototypeResult:
    max_gap_seconds = max(int(controls.max_gap_minutes * 60), 60)
    altitude_col = choose_altitude_column(df_candidate)
    cleaned_trace, summary = evaluate_clean_segment(df_candidate, altitude_col=altitude_col, thresholds=ctx.thresholds)
    trajectory_id = f'{selected_icao24}_{selected_callsign}_0'
    summary.update(
        {
            'trajectory_id': trajectory_id,
            'segment_index': 0,
            'selection_reason': selection_reason,
            'candidate_attempt': candidate_attempt,
            'selected_icao24': selected_icao24,
            'selected_callsign': selected_callsign,
            'split_count': 1,
            'trace_policy': 'bridge_middle_gaps',
            'prototype_status': prototype_status,
            'full_flight_candidates': full_flight_candidates,
            'diagnostic_fallback_used': diagnostic_fallback_used,
        }
    )
    summary.update(summarize_middle_gaps(df_candidate, max_gap_seconds))
    if extra_summary:
        summary.update(extra_summary)

    raw_trace = compute_elapsed_minutes(df_candidate.copy())
    raw_trace['trajectory_id'] = trajectory_id
    cleaned_trace = compute_elapsed_minutes(cleaned_trace.copy())
    cleaned_trace['trajectory_id'] = trajectory_id

    return PrototypeResult(
        raw_segment=raw_trace.copy(deep=True),
        cleaned_segment=cleaned_trace.copy(deep=True),
        ranking=pd.DataFrame([summary]),
        summary=pd.DataFrame([summary]),
        prototype_status=prototype_status,
    )


def _build_single_flight_prototype_full_source(ctx: TrajectoryNotebookContext, controls: PrototypeControls) -> PrototypeResult:
    parquet_file = pq.ParquetFile(ctx.adsb_input_path)
    columns = parquet_file.schema.names
    time_col = detect_time_column(columns)
    if time_col is None:
        raise ValueError('No usable time column found in ADS-B dataset.')

    month_label = controls.month_prefix or 'all_available_dates'
    selection_reason = f'windowed full-source month scan ({month_label})'

    best_partial: tuple[float, PrototypeResult] | None = None
    best_rejected_summary: dict[str, Any] | None = None
    candidate_attempt = 0

    for start in range(0, parquet_file.num_row_groups, max(controls.full_source_step_groups, 1)):
        end = min(start + max(controls.full_source_window_groups, 1), parquet_file.num_row_groups)
        row_groups = list(range(start, end))
        table = parquet_file.read_row_groups(row_groups, columns=columns)
        df_window = normalize_adsb_frame(table.to_pandas(), time_col)
        df_window = filter_month_rows(df_window, controls.month_prefix)
        if df_window.empty:
            continue

        df_window = df_window[df_window['icao24'].notna() & df_window['timestamp'].notna()].copy()
        if controls.manual_icao24 is not None:
            df_window = df_window[df_window['icao24'] == str(controls.manual_icao24).strip().lower()]
        if controls.manual_callsign is not None:
            df_window = df_window[df_window['callsign_clean'] == str(controls.manual_callsign).strip().upper()]
        else:
            df_window = df_window[df_window['callsign_clean'].map(is_informative_callsign)]
        if df_window.empty:
            continue

        counts = df_window.groupby(['icao24', 'callsign_clean'], dropna=False).size().reset_index(name='rows')
        counts = counts[counts['rows'] >= controls.min_rows_per_proxy].sort_values('rows', ascending=False).head(max(controls.full_source_top_candidates, 1))
        if counts.empty:
            continue

        for _, proxy_row in counts.iterrows():
            selected_icao24 = str(proxy_row['icao24'])
            selected_callsign = str(proxy_row['callsign_clean'])
            df_candidate = df_window[
                (df_window['icao24'] == selected_icao24)
                & (df_window['callsign_clean'] == selected_callsign)
            ].copy().sort_values('timestamp').reset_index(drop=True)
            if df_candidate.empty:
                continue

            candidate_attempt += 1
            extra_summary = {
                'window_group_start': start,
                'window_group_end': end - 1,
                'window_proxy_rows': int(proxy_row['rows']),
            }
            probe = _result_from_trace(
                ctx=ctx,
                controls=controls,
                df_candidate=df_candidate,
                selected_icao24=selected_icao24,
                selected_callsign=selected_callsign,
                selection_reason=selection_reason,
                candidate_attempt=candidate_attempt,
                prototype_status='valid_full_flight',
                full_flight_candidates=1,
                diagnostic_fallback_used=False,
                extra_summary=extra_summary,
            )
            summary_row = probe.summary.iloc[0].to_dict()

            if bool(summary_row['is_full_flight']):
                return probe

            if not str(summary_row['trajectory_quality_status']).startswith('invalid_') and str(summary_row['trajectory_quality_status']) != 'ground_only':
                partial_result = _result_from_trace(
                    ctx=ctx,
                    controls=controls,
                    df_candidate=df_candidate,
                    selected_icao24=selected_icao24,
                    selected_callsign=selected_callsign,
                    selection_reason=selection_reason,
                    candidate_attempt=candidate_attempt,
                    prototype_status='valid_partial_flight',
                    full_flight_candidates=0,
                    diagnostic_fallback_used=True,
                    extra_summary=extra_summary,
                )
                score = float(partial_result.summary['trajectory_quality_score'].iloc[0])
                if best_partial is None or score > best_partial[0]:
                    best_partial = (score, partial_result)

            if best_rejected_summary is None or float(summary_row['trajectory_quality_score']) > float(best_rejected_summary.get('trajectory_quality_score', -1.0)):
                best_rejected_summary = summary_row.copy()

    if best_partial is not None:
        return best_partial[1]

    empty_summary = {
        **({} if best_rejected_summary is None else best_rejected_summary),
        'selected_icao24': None if best_rejected_summary is None else best_rejected_summary.get('selected_icao24'),
        'selected_callsign': None if best_rejected_summary is None else best_rejected_summary.get('selected_callsign'),
        'selected_reason': selection_reason,
        'trajectory_id': None if best_rejected_summary is None else best_rejected_summary.get('trajectory_id'),
        'trajectory_quality_status': 'no_acceptable_fallback',
        'trajectory_quality_score': 0.0,
        'full_flight_candidates': 0,
        'prototype_status': 'no_valid_full_flight',
        'split_count': 1,
        'trace_policy': 'bridge_middle_gaps',
        'diagnostic_fallback_used': False,
    }
    return PrototypeResult(
        raw_segment=pd.DataFrame(),
        cleaned_segment=None,
        ranking=pd.DataFrame([empty_summary]),
        summary=pd.DataFrame([empty_summary]),
        prototype_status='no_valid_full_flight',
    )

def build_single_flight_prototype(ctx: TrajectoryNotebookContext, controls: PrototypeControls) -> PrototypeResult:
    cache_key = (
        str(ctx.adsb_input_path),
        _stat_token(ctx.adsb_input_path),
        controls.sample_rows,
        controls.random_state,
        controls.manual_icao24 or '',
        controls.manual_callsign or '',
        controls.max_gap_minutes,
        controls.max_proxy_attempts,
        controls.use_full_source,
        controls.month_prefix or '',
        controls.min_rows_per_proxy,
    )
    if cache_key in _PROTOTYPE_CACHE:
        return _copy_result(_PROTOTYPE_CACHE[cache_key])

    if not ctx.adsb_input_path.exists():
        raise FileNotFoundError(f'ADS-B input not found: {ctx.adsb_input_path}')

    if controls.use_full_source:
        result = _build_single_flight_prototype_full_source(ctx, controls)
        _PROTOTYPE_CACHE[cache_key] = _copy_result(result)
        return result

    df_sample = _sample_adsb_source(ctx, controls)
    proxy_candidates, selection_reason = get_flight_proxy_candidates(
        df_sample,
        manual_icao24=controls.manual_icao24,
        manual_callsign=controls.manual_callsign,
        random_state=controls.random_state,
        min_rows=controls.min_rows_per_proxy,
    )
    proxy_slice = proxy_candidates.head(controls.max_proxy_attempts).copy().reset_index(drop=True)

    max_gap_seconds = max(int(controls.max_gap_minutes * 60), 60)
    attempted_summaries: list[dict[str, Any]] = []
    best_fallback: dict[str, Any] | None = None
    best_rejected: dict[str, Any] | None = None

    for attempt_index, proxy_row in proxy_slice.iterrows():
        selected_icao24 = str(proxy_row['icao24'])
        selected_callsign = str(proxy_row['callsign_clean'])
        df_candidate = df_sample[
            (df_sample['icao24'] == selected_icao24)
            & (df_sample['callsign_clean'].fillna('UNKNOWN') == selected_callsign)
        ].copy()
        if df_candidate.empty:
            continue

        df_candidate = df_candidate.sort_values('timestamp').reset_index(drop=True)
        altitude_col = choose_altitude_column(df_candidate)
        cleaned_trace, summary = evaluate_clean_segment(df_candidate, altitude_col=altitude_col, thresholds=ctx.thresholds)
        trajectory_id = f'{selected_icao24}_{selected_callsign}_0'
        summary.update(
            {
                'trajectory_id': trajectory_id,
                'segment_index': 0,
                'selection_reason': selection_reason,
                'candidate_attempt': attempt_index + 1,
                'selected_icao24': selected_icao24,
                'selected_callsign': selected_callsign,
                'split_count': 1,
                'trace_policy': 'bridge_middle_gaps',
            }
        )
        summary.update(summarize_middle_gaps(df_candidate, max_gap_seconds))

        raw_trace = compute_elapsed_minutes(df_candidate.copy())
        raw_trace['trajectory_id'] = trajectory_id
        cleaned_trace = compute_elapsed_minutes(cleaned_trace.copy())
        cleaned_trace['trajectory_id'] = trajectory_id

        attempted_summaries.append(summary.copy())
        ranking = pd.DataFrame(attempted_summaries).sort_values(
            ['is_full_flight', 'trajectory_quality_score', 'route_distance_km', 'clean_rows'],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        if summary['is_full_flight']:
            chosen_summary = summary.copy()
            chosen_summary['selected_reason'] = selection_reason
            chosen_summary['prototype_status'] = 'valid_full_flight'
            chosen_summary['full_flight_candidates'] = int(ranking['is_full_flight'].sum())
            chosen_summary['diagnostic_fallback_used'] = False
            result = PrototypeResult(
                raw_segment=raw_trace.copy(deep=True),
                cleaned_segment=cleaned_trace.copy(deep=True),
                ranking=pd.DataFrame([chosen_summary]),
                summary=pd.DataFrame([chosen_summary]),
                prototype_status='valid_full_flight',
            )
            _PROTOTYPE_CACHE[cache_key] = _copy_result(result)
            return result

        rejected_record = {
            'ranking': ranking.copy(deep=True),
            'summary': summary.copy(),
        }
        if best_rejected is None or float(summary['trajectory_quality_score']) > float(best_rejected['summary']['trajectory_quality_score']):
            best_rejected = rejected_record

        single_row_ranking = pd.DataFrame([summary])
        fallback_row = choose_diagnostic_fallback(single_row_ranking)
        if fallback_row is not None:
            fallback_summary = fallback_row.to_dict()
            fallback_record = {
                'raw': raw_trace.copy(deep=True),
                'cleaned': cleaned_trace.copy(deep=True),
                'ranking': ranking.copy(deep=True),
                'summary': fallback_summary.copy(),
            }
            if best_fallback is None or float(fallback_summary['trajectory_quality_score']) > float(best_fallback['summary']['trajectory_quality_score']):
                best_fallback = fallback_record

    if best_fallback is None:
        top_candidate = proxy_candidates.iloc[0]
        base_summary = {} if best_rejected is None else best_rejected['summary'].copy()
        empty_summary = {
            **base_summary,
            'selected_icao24': str(top_candidate['icao24']) if 'icao24' in top_candidate else base_summary.get('selected_icao24'),
            'selected_callsign': str(top_candidate['callsign_clean']) if 'callsign_clean' in top_candidate else base_summary.get('selected_callsign'),
            'selected_reason': selection_reason,
            'trajectory_id': base_summary.get('trajectory_id'),
            'raw_rows': base_summary.get('raw_rows', 0),
            'clean_rows': base_summary.get('clean_rows', 0),
            'map_rows': base_summary.get('map_rows', 0),
            'missing_timestamp_rows_removed': base_summary.get('missing_timestamp_rows_removed', 0),
            'invalid_coordinate_rows_removed': base_summary.get('invalid_coordinate_rows_removed', 0),
            'exact_duplicates_removed': base_summary.get('exact_duplicates_removed', 0),
            'stale_rows_removed': base_summary.get('stale_rows_removed', 0),
            'altitude_spikes_removed': base_summary.get('altitude_spikes_removed', 0),
            'duration_minutes': base_summary.get('duration_minutes', 0.0),
            'valid_map_points': base_summary.get('valid_map_points', 0),
            'route_distance_km': base_summary.get('route_distance_km', 0.0),
            'altitude_span_m': base_summary.get('altitude_span_m', 0.0),
            'start_altitude_m': base_summary.get('start_altitude_m', np.nan),
            'end_altitude_m': base_summary.get('end_altitude_m', np.nan),
            'starts_groundish': base_summary.get('starts_groundish', False),
            'ends_groundish': base_summary.get('ends_groundish', False),
            'has_sustained_climb': base_summary.get('has_sustained_climb', False),
            'has_sustained_descent': base_summary.get('has_sustained_descent', False),
            'has_altitude_spike': base_summary.get('has_altitude_spike', False),
            'is_mappable': base_summary.get('is_mappable', False),
            'is_full_flight': False,
            'trajectory_quality_status': 'no_acceptable_fallback',
            'trajectory_quality_score': 0.0,
            'full_flight_candidates': 0,
            'prototype_status': 'no_valid_full_flight',
            'split_count': 1,
            'trace_policy': 'bridge_middle_gaps',
            'middle_gap_count': base_summary.get('middle_gap_count', 0),
            'largest_gap_minutes': base_summary.get('largest_gap_minutes', 0.0),
            'has_middle_gap': base_summary.get('has_middle_gap', False),
            'diagnostic_fallback_used': False,
        }
        result = PrototypeResult(
            raw_segment=pd.DataFrame(),
            cleaned_segment=None,
            ranking=pd.DataFrame([empty_summary]),
            summary=pd.DataFrame([empty_summary]),
            prototype_status='no_valid_full_flight',
        )
        _PROTOTYPE_CACHE[cache_key] = _copy_result(result)
        return result

    fallback_summary = best_fallback['summary'].copy()
    fallback_summary['selected_reason'] = selection_reason
    fallback_summary['prototype_status'] = 'valid_partial_flight'
    fallback_summary['full_flight_candidates'] = 0
    fallback_summary['diagnostic_fallback_used'] = True
    result = PrototypeResult(
        raw_segment=best_fallback['raw'].copy(deep=True),
        cleaned_segment=best_fallback['cleaned'].copy(deep=True),
        ranking=pd.DataFrame([fallback_summary]),
        summary=pd.DataFrame([fallback_summary]),
        prototype_status='valid_partial_flight',
    )
    _PROTOTYPE_CACHE[cache_key] = _copy_result(result)
    return result


def summarize_prototype_segments(result: PrototypeResult) -> pd.DataFrame:
    if result.ranking.empty:
        return pd.DataFrame(columns=['trajectory_quality_status', 'trajectory_count', 'pct'])
    counts = (
        result.ranking['trajectory_quality_status']
        .value_counts()
        .rename_axis('trajectory_quality_status')
        .reset_index(name='trajectory_count')
    )
    counts['pct'] = counts['trajectory_count'] / counts['trajectory_count'].sum() * 100
    return counts


def summarize_partial_status_from_parquet(input_path: Path, batch_size: int = 100_000) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f'Trajectory parquet not found: {input_path}')

    parquet_file = pq.ParquetFile(input_path)
    needed_cols = [c for c in ['trajectory_id', 'trajectory_quality_status', 'is_full_flight'] if c in parquet_file.schema.names]
    if 'trajectory_id' not in needed_cols or 'trajectory_quality_status' not in needed_cols:
        raise ValueError('Trajectory parquet does not include quality status columns yet. Run the batch reconstruction cell first.')

    frames: list[pd.DataFrame] = []
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=needed_cols):
        frames.append(batch.to_pandas())
    if not frames:
        return pd.DataFrame(columns=['trajectory_quality_status', 'trajectory_count', 'pct'])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['trajectory_id'])
    counts = df['trajectory_quality_status'].value_counts().rename_axis('trajectory_quality_status').reset_index(name='trajectory_count')
    counts['pct'] = counts['trajectory_count'] / counts['trajectory_count'].sum() * 100
    return counts


def reconstruct_full_dataset(
    ctx: TrajectoryNotebookContext,
    *,
    batch_size: int = 200_000,
    max_gap_minutes: int | None = None,
) -> tuple[Path, pd.DataFrame]:
    if not ctx.adsb_input_path.exists():
        raise FileNotFoundError(f'ADS-B input not found: {ctx.adsb_input_path}')

    max_gap_seconds = max(int((ctx.max_gap_minutes if max_gap_minutes is None else max_gap_minutes) * 60), 60)
    parquet_file = pq.ParquetFile(ctx.adsb_input_path)
    columns = parquet_file.schema.names
    time_col = detect_time_column(columns)
    if time_col is None:
        raise ValueError('No usable time column found in ADS-B dataset.')

    output_path = ctx.traj_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f'{output_path.stem}.tmp{output_path.suffix}')
    if temp_output_path.exists():
        temp_output_path.unlink()

    writer: pq.ParquetWriter | None = None
    total_output_rows = 0
    trajectory_quality_rows: list[dict[str, Any]] = []
    active_by_icao24: dict[str, pd.DataFrame] = {}
    segment_counters: Counter[str] = Counter()
    last_batch_max_timestamp: float | None = None

    def materialize_segment(icao24: str, raw_segment: pd.DataFrame) -> pd.DataFrame | None:
        if raw_segment.empty:
            return None

        altitude_col = choose_altitude_column(raw_segment)
        cleaned_segment, summary = evaluate_clean_segment(
            raw_segment,
            altitude_col=altitude_col,
            thresholds=ctx.thresholds,
        )
        summary.update(summarize_middle_gaps(raw_segment, max_gap_seconds))
        summary['route_coverage_fraction'] = 1.0 - float(summary.get('gap_fraction_of_flight', 0.0))

        segment_callsign = select_segment_callsign(raw_segment, fallback='UNKNOWN')
        segment_index = segment_counters[icao24]
        segment_counters[icao24] += 1
        trajectory_id = f'{icao24}_{segment_callsign}_{segment_index}'

        trajectory_quality_rows.append(
            {
                'trajectory_id': trajectory_id,
                'trajectory_quality_status': summary['trajectory_quality_status'],
                'trajectory_quality_score': summary['trajectory_quality_score'],
                'is_full_flight': summary['is_full_flight'],
                'starts_groundish': summary['starts_groundish'],
                'ends_groundish': summary['ends_groundish'],
                'has_altitude_spike': summary['has_altitude_spike'],
                'is_mappable': summary['is_mappable'],
                'middle_gap_count': summary.get('middle_gap_count', 0),
                'largest_gap_minutes': summary.get('largest_gap_minutes', 0.0),
                'total_gap_minutes': summary.get('total_gap_minutes', 0.0),
                'gap_fraction_of_flight': summary.get('gap_fraction_of_flight', 0.0),
                'route_coverage_fraction': summary.get('route_coverage_fraction', 1.0),
                'max_inter_ping_seconds': summary.get('max_inter_ping_seconds', 0.0),
                'median_inter_ping_seconds': summary.get('median_inter_ping_seconds', 0.0),
                'ping_interval_cv': summary.get('ping_interval_cv', 0.0),
                'kept_in_output': summary['kept_in_output'],
            }
        )

        if cleaned_segment.empty:
            return None

        cleaned_segment = cleaned_segment.copy()
        cleaned_segment['trajectory_id'] = trajectory_id
        for key in [
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
        ]:
            cleaned_segment[key] = summary[key]
        return strip_helper_columns(cleaned_segment)

    def write_outputs(outputs: list[pd.DataFrame]) -> None:
        nonlocal writer, total_output_rows
        if not outputs:
            return
        df_out = pd.concat(outputs, ignore_index=True)
        table = pa.Table.from_pandas(df_out, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(temp_output_path, table.schema, compression='snappy')
        writer.write_table(table)
        total_output_rows += len(df_out)

    try:
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            df_batch = normalize_adsb_frame(record_batch.to_pandas(), time_col)
            if df_batch.empty:
                continue

            batch_outputs: list[pd.DataFrame] = []
            batch_max_timestamp = pd.to_numeric(df_batch['timestamp'], errors='coerce').max()
            if pd.notna(batch_max_timestamp):
                batch_max_timestamp = float(batch_max_timestamp)
                if last_batch_max_timestamp is not None and batch_max_timestamp < last_batch_max_timestamp:
                    logger.warning(
                        'ADS-B parquet batches are not strictly chronological; trajectory carry-over will still run, '
                        'but full-flight stitching may be less reliable.'
                    )
                last_batch_max_timestamp = batch_max_timestamp

            for icao24, aircraft_df in df_batch.groupby('icao24', sort=False):
                if pd.isna(icao24):
                    continue
                icao24_key = str(icao24)
                aircraft_parts = []
                if icao24_key in active_by_icao24:
                    aircraft_parts.append(active_by_icao24.pop(icao24_key))
                aircraft_parts.append(aircraft_df)
                aircraft_df = pd.concat(aircraft_parts, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
                altitude_col = choose_altitude_column(aircraft_df)
                raw_segments = split_trace_into_segments(
                    aircraft_df,
                    altitude_col=altitude_col,
                    thresholds=ctx.thresholds,
                    max_gap_seconds=max_gap_seconds,
                    split_on_time_gap=False,
                )

                if not raw_segments:
                    continue

                for raw_segment in raw_segments[:-1]:
                    cleaned_segment = materialize_segment(icao24_key, raw_segment)
                    if cleaned_segment is not None:
                        batch_outputs.append(cleaned_segment)

                active_by_icao24[icao24_key] = raw_segments[-1].copy()

            if last_batch_max_timestamp is not None:
                stale_icao24s: list[str] = []
                for icao24_key, active_segment in active_by_icao24.items():
                    if active_segment.empty or 'timestamp' not in active_segment.columns:
                        stale_icao24s.append(icao24_key)
                        continue
                    active_max_timestamp = pd.to_numeric(active_segment['timestamp'], errors='coerce').max()
                    if pd.isna(active_max_timestamp):
                        stale_icao24s.append(icao24_key)
                        continue
                    last_row = active_segment.sort_values('timestamp').iloc[-1]
                    altitude_col = choose_altitude_column(active_segment)
                    last_altitude = last_row.get(altitude_col)
                    ends_groundish = bool(last_row.get('on_ground', False)) or (
                        pd.notna(last_altitude)
                        and float(last_altitude) <= ctx.thresholds.ground_altitude_threshold_m
                    )
                    inactive_seconds = last_batch_max_timestamp - float(active_max_timestamp)
                    if ends_groundish and inactive_seconds > max(max_gap_seconds * 4, 3600):
                        stale_icao24s.append(icao24_key)

                for icao24_key in stale_icao24s:
                    active_segment = active_by_icao24.pop(icao24_key, pd.DataFrame())
                    cleaned_segment = materialize_segment(icao24_key, active_segment)
                    if cleaned_segment is not None:
                        batch_outputs.append(cleaned_segment)

            write_outputs(batch_outputs)

        final_outputs: list[pd.DataFrame] = []
        for icao24_key, active_segment in list(active_by_icao24.items()):
            cleaned_segment = materialize_segment(icao24_key, active_segment)
            if cleaned_segment is not None:
                final_outputs.append(cleaned_segment)
        active_by_icao24.clear()
        write_outputs(final_outputs)
    finally:
        if writer is not None:
            writer.close()

    if total_output_rows == 0:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise ValueError('No cleaned trajectory rows were produced during batch reconstruction.')

    backup_path = output_path.with_name(f'{output_path.stem}.bak{output_path.suffix}')
    if backup_path.exists():
        backup_path.unlink()
    if output_path.exists():
        output_path.replace(backup_path)
    temp_output_path.replace(output_path)
    if backup_path.exists():
        backup_path.unlink()

    _INSPECT_CACHE.clear()
    _SAMPLE_CACHE.clear()
    _PROTOTYPE_CACHE.clear()

    return output_path, pd.DataFrame(trajectory_quality_rows)

class TrajectoryBuilder:
    """Legacy simple trajectory builder kept for backward compatibility."""

    def __init__(self, max_gap_minutes: int = 15):
        self.max_gap_seconds = max_gap_minutes * 60
        self.output_cols = ['trajectory_id', 'icao24', 'timestamp', 'latitude', 'longitude', 'altitude', 'velocity', 'heading']

    def filter_invalid_points(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_len = len(df)
        df_filtered = df.dropna(subset=['latitude', 'longitude'])
        if 'baro_altitude' in df_filtered.columns:
            df_filtered['altitude'] = df_filtered['baro_altitude']
        if 'true_track' in df_filtered.columns:
            df_filtered['heading'] = df_filtered['true_track']
        keep_cols = ['icao24', 'timestamp', 'latitude', 'longitude', 'altitude', 'velocity', 'heading']
        existing_cols = [col for col in keep_cols if col in df_filtered.columns]
        df_filtered = df_filtered[existing_cols].copy()
        logger.info('Filtered %s invalid points.', initial_len - len(df_filtered))
        return df_filtered

    def sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'timestamp' in df.columns:
            return df.sort_values(['icao24', 'timestamp']).reset_index(drop=True)
        return df

    def load_partitioned_data(self, base_dir: str | Path, pattern: str = 'states_*.parquet', batch_size: int = 20) -> pd.DataFrame:
        base_dir = Path(base_dir)
        parquet_files = sorted(base_dir.rglob(pattern))
        if not parquet_files:
            logger.warning('No partitioned files found under %s with pattern %s', base_dir, pattern)
            return pd.DataFrame()

        logger.info('Loading %s partitioned state files from %s', len(parquet_files), base_dir)
        batch_frames: list[pd.DataFrame] = []
        all_batches: list[pd.DataFrame] = []
        for idx, file_path in enumerate(parquet_files, start=1):
            batch_frames.append(pd.read_parquet(file_path))
            if len(batch_frames) >= batch_size:
                all_batches.append(pd.concat(batch_frames, ignore_index=True))
                batch_frames = []
                logger.info('  Loaded %s/%s files...', idx, len(parquet_files))

        if batch_frames:
            all_batches.append(pd.concat(batch_frames, ignore_index=True))

        return pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame()

    def group_by_aircraft(self, df: pd.DataFrame):
        return df.groupby('icao24')

    def build_flight_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.filter_invalid_points(df)
        df = self.sort_by_time(df)

        logger.info('Building flight segments...')
        df['time_diff'] = df.groupby('icao24')['timestamp'].diff()
        df['new_trajectory'] = (df['time_diff'] > self.max_gap_seconds) | (df['time_diff'].isna())
        df['flight_idx'] = df.groupby('icao24')['new_trajectory'].cumsum()
        df['trajectory_id'] = df['icao24'] + '_' + df['flight_idx'].astype(str)
        df.drop(columns=['time_diff', 'new_trajectory', 'flight_idx'], inplace=True)

        for col in self.output_cols:
            if col not in df.columns:
                df[col] = None
        return df[self.output_cols]

    def save_trajectories(self, df: pd.DataFrame, output_path: str | Path):
        if df.empty:
            logger.warning('Empty DataFrame, nothing to save.')
            return

        output_path = Path(output_path)
        ensure_dir(output_path.parent)
        logger.info('Saving %s trajectory points to %s', len(df), output_path)
        df.to_parquet(output_path, index=False)


__all__ = [
    'cache_notebook_controls',
    'PrototypeControls',
    'PrototypeResult',
    'TrajectoryBuilder',
    'TrajectoryNotebookContext',
    'TrajectoryThresholds',
    'build_notebook_context',
    'build_single_flight_prototype',
    'choose_altitude_column',
    'compute_elapsed_minutes',
    'detect_time_column',
    'get_notebook_controls',
    'inspect_adsb_source',
    'make_prototype_controls',
    'reconstruct_full_dataset',
    'summarize_partial_status_from_parquet',
    'summarize_prototype_segments',
]
