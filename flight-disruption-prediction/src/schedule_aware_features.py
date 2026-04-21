"""Schedule-aware direct ADS-B feature extraction.

This module avoids materializing a full point-level ``trajectories.parquet``.
It filters ADS-B pings to scheduled callsigns/time windows, downsamples each
matched flight window, and writes compact flight-level feature vectors.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.feature_engineering import FeatureExtractor
from src.merge import DataMerger
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScheduleAwareFeatureConfig:
    """Controls for schedule-aware direct feature extraction."""

    batch_size: int = 200_000
    partitions: int = 16
    pre_departure_hours: float = 2.0
    post_arrival_hours: float = 3.0
    fallback_duration_hours: float = 3.0
    downsample_interval_seconds: int = 60
    phase_detail_minutes: int = 15
    phase_interval_seconds: int = 30
    max_points_per_flight: int = 300
    min_points_per_flight: int = 10
    save_trajectory_sketches: bool = True
    sketch_interval_seconds: int = 600
    sketch_phase_interval_seconds: int = 120
    sketch_phase_detail_minutes: int = 15
    sketch_max_points_per_flight: int = 80
    sketch_output_file: str = "trajectory_sketches.parquet"
    max_schedule_rows: int | None = None
    schedule_sources: tuple[str, ...] = ("bts",)
    cleanup_work_dir: bool = False
    force_repartition: bool = False


def extract_schedule_aware_features(
    *,
    adsb_path: str | Path,
    schedule_paths: dict[str, str | Path],
    output_path: str | Path,
    work_dir: str | Path,
    config: ScheduleAwareFeatureConfig | None = None,
    max_gap_minutes: int = 15,
) -> pd.DataFrame:
    """Build compact flight-level vectors directly from ADS-B + schedules."""

    cfg = config or ScheduleAwareFeatureConfig()
    adsb_path = Path(adsb_path)
    output_path = Path(output_path)
    work_dir = Path(work_dir)
    partition_dir = work_dir / "filtered_adsb_by_callsign"
    sketch_path = output_path.parent / cfg.sketch_output_file

    if not adsb_path.exists():
        raise FileNotFoundError(f"ADS-B input not found: {adsb_path}")

    schedules = load_schedule_windows(schedule_paths, cfg)
    if schedules.empty:
        logger.warning("No schedule windows available; writing empty feature table.")
        ensure_dir(output_path.parent)
        pd.DataFrame().to_parquet(output_path, index=False)
        return pd.DataFrame()

    logger.info(
        "Schedule-aware feature extraction using %s schedule rows and %s unique callsigns.",
        len(schedules),
        schedules["callsign_clean"].nunique(),
    )

    if cfg.force_repartition and partition_dir.exists():
        shutil.rmtree(partition_dir)

    partition_paths = partition_adsb_by_scheduled_callsign(
        adsb_path=adsb_path,
        schedules=schedules,
        partition_dir=partition_dir,
        cfg=cfg,
    )

    extractor = FeatureExtractor(max_gap_minutes=max_gap_minutes)
    frames: list[pd.DataFrame] = []
    sketch_writer: pq.ParquetWriter | None = None
    sketch_schema: pa.Schema | None = None
    sketch_rows = 0
    if cfg.save_trajectory_sketches:
        ensure_dir(sketch_path.parent)
        if sketch_path.exists():
            sketch_path.unlink()

    for idx, part_path in enumerate(partition_paths, start=1):
        logger.info("Processing schedule-aware partition %d/%d: %s", idx, len(partition_paths), part_path.name)
        part_features, part_sketches = extract_features_from_partition(
            part_path,
            schedules=schedules,
            cfg=cfg,
            extractor=extractor,
        )
        if not part_features.empty:
            frames.append(part_features)
        if cfg.save_trajectory_sketches and part_sketches is not None and not part_sketches.empty:
            table = pa.Table.from_pandas(part_sketches, preserve_index=False)
            if sketch_writer is None:
                sketch_schema = table.schema
                sketch_writer = pq.ParquetWriter(sketch_path, sketch_schema, compression="snappy")
            else:
                table = table.cast(sketch_schema)
            sketch_writer.write_table(table)
            sketch_rows += len(part_sketches)
        logger.info(
            "Partition %s produced %s feature vectors and %s sketch points.",
            part_path.name,
            len(part_features),
            0 if part_sketches is None else len(part_sketches),
        )

    if sketch_writer is not None:
        sketch_writer.close()

    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    features = coerce_feature_dtypes(features)

    ensure_dir(output_path.parent)
    features.to_parquet(output_path, index=False)
    logger.info("Schedule-aware direct features saved to %s (%s rows).", output_path, len(features))
    if cfg.save_trajectory_sketches:
        logger.info("Trajectory sketches saved to %s (%s point rows).", sketch_path, f"{sketch_rows:,}")

    if cfg.cleanup_work_dir and partition_dir.exists():
        shutil.rmtree(partition_dir)

    return features


def load_schedule_windows(
    schedule_paths: dict[str, str | Path],
    cfg: ScheduleAwareFeatureConfig,
) -> pd.DataFrame:
    """Load and normalize schedule rows into callsign/time windows."""

    merger = DataMerger()
    frames: list[pd.DataFrame] = []
    wanted_sources = {source.lower() for source in cfg.schedule_sources}

    for source_name, raw_path in schedule_paths.items():
        source_key = source_name.lower()
        if source_key not in wanted_sources:
            continue
        path = Path(raw_path)
        if not path.exists():
            logger.warning("Schedule source %s not found at %s; skipping.", source_name, path)
            continue

        df_schedule = pd.read_parquet(path)
        if "region" not in df_schedule.columns:
            df_schedule["region"] = "US" if source_key == "bts" else "EU"

        prepared, detected_source = merger._prepare_schedule(df_schedule)
        if prepared.empty:
            continue

        prepared = prepared.copy()
        prepared["schedule_source_name"] = detected_source
        prepared["schedule_row_id"] = np.arange(len(prepared), dtype=np.int64)
        prepared["scheduled_dep_ref"] = pd.to_datetime(prepared["scheduled_dep_ref"], utc=True, errors="coerce")
        prepared["scheduled_arr_ref"] = pd.to_datetime(prepared["scheduled_arr_ref"], utc=True, errors="coerce")

        fallback_arr = prepared["scheduled_dep_ref"] + pd.to_timedelta(cfg.fallback_duration_hours, unit="h")
        prepared["scheduled_arr_ref"] = prepared["scheduled_arr_ref"].fillna(fallback_arr)
        prepared["window_start_utc"] = prepared["scheduled_dep_ref"] - pd.to_timedelta(cfg.pre_departure_hours, unit="h")
        prepared["window_end_utc"] = prepared["scheduled_arr_ref"] + pd.to_timedelta(cfg.post_arrival_hours, unit="h")

        prepared = prepared.dropna(subset=["callsign_clean", "scheduled_dep_ref", "window_start_utc", "window_end_utc"])
        prepared = prepared[prepared["window_end_utc"] > prepared["window_start_utc"]].copy()
        frames.append(prepared)

    if not frames:
        return pd.DataFrame()

    schedules = pd.concat(frames, ignore_index=True)
    schedules = schedules.sort_values(["scheduled_dep_ref", "callsign_clean"], kind="stable").reset_index(drop=True)
    schedules["schedule_row_id"] = np.arange(len(schedules), dtype=np.int64)

    if cfg.max_schedule_rows is not None and cfg.max_schedule_rows > 0:
        schedules = schedules.head(cfg.max_schedule_rows).copy()

    schedules["window_start_ts"] = schedules["window_start_utc"].astype("int64") // 1_000_000_000
    schedules["window_end_ts"] = schedules["window_end_utc"].astype("int64") // 1_000_000_000
    schedules["scheduled_dep_ts"] = schedules["scheduled_dep_ref"].astype("int64") // 1_000_000_000
    schedules["scheduled_arr_ts"] = schedules["scheduled_arr_ref"].astype("int64") // 1_000_000_000

    keep_cols = [
        "schedule_row_id",
        "flight_key",
        "callsign",
        "callsign_clean",
        "scheduled_dep_ref",
        "scheduled_arr_ref",
        "window_start_utc",
        "window_end_utc",
        "window_start_ts",
        "window_end_ts",
        "scheduled_dep_ts",
        "scheduled_arr_ts",
        "origin",
        "destination",
        "region",
        "source_dataset",
        "schedule_source_name",
        "label_source",
    ]
    return schedules[[col for col in keep_cols if col in schedules.columns]].copy()


def partition_adsb_by_scheduled_callsign(
    *,
    adsb_path: Path,
    schedules: pd.DataFrame,
    partition_dir: Path,
    cfg: ScheduleAwareFeatureConfig,
) -> list[Path]:
    """Filter ADS-B to relevant callsigns/global range and shard by callsign."""

    existing_parts = sorted(partition_dir.glob("adsb_sched_part_*.parquet")) if partition_dir.exists() else []
    if existing_parts and not cfg.force_repartition:
        logger.info("Reusing %s existing schedule-aware ADS-B partitions.", len(existing_parts))
        return existing_parts

    if partition_dir.exists():
        shutil.rmtree(partition_dir)
    partition_dir.mkdir(parents=True, exist_ok=True)

    callsigns = set(schedules["callsign_clean"].dropna().astype(str).unique())
    min_ts = int(schedules["window_start_ts"].min())
    max_ts = int(schedules["window_end_ts"].max())

    parquet_file = pq.ParquetFile(adsb_path)
    available_cols = parquet_file.schema.names
    wanted_cols = [
        "icao24",
        "callsign",
        "timestamp",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "on_ground",
        "velocity",
        "true_track",
        "vertical_rate",
        "geo_altitude",
    ]
    columns = [col for col in wanted_cols if col in available_cols]
    if "callsign" not in columns or "timestamp" not in columns:
        raise ValueError("ADS-B input must contain at least 'callsign' and 'timestamp'.")

    relevant_row_groups = relevant_parquet_row_groups(parquet_file, "timestamp", min_ts, max_ts)
    if relevant_row_groups:
        relevant_rows = sum(parquet_file.metadata.row_group(idx).num_rows for idx in relevant_row_groups)
        logger.info(
            "ADS-B row-group pruning selected %s/%s row groups (%s/%s rows) for schedule window %s to %s.",
            len(relevant_row_groups),
            parquet_file.metadata.num_row_groups,
            f"{relevant_rows:,}",
            f"{parquet_file.metadata.num_rows:,}",
            pd.to_datetime(min_ts, unit="s", utc=True),
            pd.to_datetime(max_ts, unit="s", utc=True),
        )
    else:
        logger.warning("No timestamp row-group stats available; scanning all ADS-B row groups.")
        relevant_row_groups = list(range(parquet_file.metadata.num_row_groups))

    partition_paths = {
        idx: partition_dir / f"adsb_sched_part_{idx:03d}.parquet"
        for idx in range(max(1, cfg.partitions))
    }
    writers: dict[int, pq.ParquetWriter] = {}
    rows_seen = 0
    rows_kept = 0
    batch_idx = 0

    try:
        for batch in parquet_file.iter_batches(
            batch_size=cfg.batch_size,
            columns=columns,
            row_groups=relevant_row_groups,
        ):
            batch_idx += 1
            df = batch.to_pandas()
            rows_seen += len(df)
            df = normalize_adsb_batch(df)
            df = df[
                df["callsign_clean"].isin(callsigns)
                & df["timestamp"].between(min_ts, max_ts, inclusive="both")
            ].copy()
            if df.empty:
                if batch_idx % 50 == 0:
                    logger.info("Schedule-aware ADS-B filter batch %s: %s rows seen, %s kept.", batch_idx, f"{rows_seen:,}", f"{rows_kept:,}")
                continue

            partition_ids = (
                pd.util.hash_pandas_object(df["callsign_clean"].astype("string"), index=False).to_numpy()
                % max(1, cfg.partitions)
            )

            for partition_id in range(max(1, cfg.partitions)):
                subset = df.loc[partition_ids == partition_id]
                if subset.empty:
                    continue
                table = pa.Table.from_pandas(subset, preserve_index=False)
                if partition_id not in writers:
                    writers[partition_id] = pq.ParquetWriter(
                        partition_paths[partition_id],
                        table.schema,
                        compression="snappy",
                    )
                writers[partition_id].write_table(table)
                rows_kept += len(subset)

            if batch_idx % 25 == 0:
                logger.info(
                    "Schedule-aware ADS-B filter batch %s: %s rows seen, %s kept.",
                    batch_idx,
                    f"{rows_seen:,}",
                    f"{rows_kept:,}",
                )
    finally:
        for writer in writers.values():
            writer.close()

    parts = sorted(path for path in partition_paths.values() if path.exists())
    logger.info(
        "Schedule-aware ADS-B filtering kept %s/%s rows into %s partitions.",
        f"{rows_kept:,}",
        f"{rows_seen:,}",
        len(parts),
    )
    return parts


def relevant_parquet_row_groups(
    parquet_file: pq.ParquetFile,
    column_name: str,
    min_value: int,
    max_value: int,
) -> list[int]:
    """Return row groups whose min/max statistics overlap the wanted range."""

    names = parquet_file.schema.names
    if column_name not in names:
        return []

    column_index = names.index(column_name)
    row_groups: list[int] = []
    for idx in range(parquet_file.metadata.num_row_groups):
        row_group = parquet_file.metadata.row_group(idx)
        stats = row_group.column(column_index).statistics
        if stats is None or stats.min is None or stats.max is None:
            return []
        try:
            stat_min = int(stats.min)
            stat_max = int(stats.max)
        except (TypeError, ValueError):
            return []
        if stat_max >= min_value and stat_min <= max_value:
            row_groups.append(idx)
    return row_groups


def normalize_adsb_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw ADS-B columns used by the direct feature path."""

    df = df.copy()
    df["callsign_clean"] = DataMerger._clean_callsign(df.get("callsign", pd.Series(pd.NA, index=df.index)), "adsb")
    for col in ["timestamp", "latitude", "longitude", "baro_altitude", "geo_altitude", "velocity", "true_track", "vertical_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df = df.dropna(subset=["timestamp"])
        df["timestamp"] = df["timestamp"].astype("int64")
    if "altitude" not in df.columns:
        if "geo_altitude" in df.columns:
            df["altitude"] = df["geo_altitude"]
            if "baro_altitude" in df.columns:
                df["altitude"] = df["altitude"].fillna(df["baro_altitude"])
        elif "baro_altitude" in df.columns:
            df["altitude"] = df["baro_altitude"]
        else:
            df["altitude"] = np.nan
    if "heading" not in df.columns:
        df["heading"] = df["true_track"] if "true_track" in df.columns else np.nan
    return df.dropna(subset=["callsign_clean"]).copy()


def extract_features_from_partition(
    partition_path: Path,
    *,
    schedules: pd.DataFrame,
    cfg: ScheduleAwareFeatureConfig,
    extractor: FeatureExtractor,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Extract scheduled-flight features from one callsign partition."""

    if not partition_path.exists() or partition_path.stat().st_size == 0:
        return pd.DataFrame(), pd.DataFrame() if cfg.save_trajectory_sketches else None

    df_adsb = pd.read_parquet(partition_path)
    if df_adsb.empty:
        return pd.DataFrame(), pd.DataFrame() if cfg.save_trajectory_sketches else None

    df_adsb = normalize_adsb_batch(df_adsb)
    callsigns = set(df_adsb["callsign_clean"].dropna().astype(str).unique())
    part_schedules = schedules[schedules["callsign_clean"].astype(str).isin(callsigns)].copy()
    if part_schedules.empty:
        return pd.DataFrame(), pd.DataFrame() if cfg.save_trajectory_sketches else None

    feature_rows: list[dict] = []
    sketch_frames: list[pd.DataFrame] = []
    for callsign_clean, sched_group in part_schedules.groupby("callsign_clean", sort=False):
        adsb_group = df_adsb[df_adsb["callsign_clean"].astype(str) == str(callsign_clean)].copy()
        if adsb_group.empty:
            continue
        adsb_group = adsb_group.sort_values("timestamp", kind="stable").reset_index(drop=True)
        ts_values = adsb_group["timestamp"].to_numpy(dtype=np.int64)

        for schedule_row in sched_group.itertuples(index=False):
            left = int(np.searchsorted(ts_values, int(schedule_row.window_start_ts), side="left"))
            right = int(np.searchsorted(ts_values, int(schedule_row.window_end_ts), side="right"))
            if right - left < cfg.min_points_per_flight:
                continue

            window = adsb_group.iloc[left:right].copy()
            window = choose_dominant_aircraft(window)
            if len(window) < cfg.min_points_per_flight:
                continue

            feature_row = build_scheduled_feature_row(window, schedule_row, cfg, extractor)
            if feature_row:
                feature_rows.append(feature_row)
                if cfg.save_trajectory_sketches:
                    sketch = build_trajectory_sketch(window, schedule_row, feature_row, cfg)
                    if not sketch.empty:
                        sketch_frames.append(sketch)

    features = pd.DataFrame(feature_rows)
    sketches = pd.concat(sketch_frames, ignore_index=True) if sketch_frames else pd.DataFrame()
    return features, sketches if cfg.save_trajectory_sketches else None


def choose_dominant_aircraft(df: pd.DataFrame) -> pd.DataFrame:
    """For duplicate callsign windows, keep the icao24 with the most pings."""

    if "icao24" not in df.columns or df["icao24"].nunique(dropna=True) <= 1:
        return df
    dominant = df["icao24"].astype("string").value_counts(dropna=True).index[0]
    return df[df["icao24"].astype("string") == dominant].copy()


def build_scheduled_feature_row(
    df_window: pd.DataFrame,
    schedule_row,
    cfg: ScheduleAwareFeatureConfig,
    extractor: FeatureExtractor,
) -> dict:
    """Downsample one scheduled window and compute a flight-level feature row."""

    df_window = clean_flight_window(df_window)
    if len(df_window) < cfg.min_points_per_flight:
        return {}

    df_sampled = downsample_flight_window(df_window, schedule_row, cfg)
    if len(df_sampled) < cfg.min_points_per_flight:
        return {}

    trajectory_id = make_trajectory_id(schedule_row)
    df_sampled = df_sampled.copy()
    df_sampled["trajectory_id"] = trajectory_id
    df_sampled["callsign"] = str(schedule_row.callsign_clean)

    starts_groundish, ends_groundish = infer_groundish_flags(df_sampled)
    route_coverage = compute_route_coverage(df_sampled, extractor.max_gap_seconds)
    status = classify_scheduled_window(df_sampled, starts_groundish, ends_groundish)
    quality_score = score_scheduled_window(status, route_coverage)

    df_sampled["trajectory_quality_status"] = status
    df_sampled["trajectory_quality_score"] = quality_score
    df_sampled["is_full_flight"] = status == "full_flight"
    df_sampled["starts_groundish"] = starts_groundish
    df_sampled["ends_groundish"] = ends_groundish
    df_sampled["has_altitude_spike"] = False
    df_sampled["is_mappable"] = has_mappable_geometry(df_sampled)

    row = extractor._extract_single_trajectory(df_sampled)
    if not row:
        return {}

    row.update(
        {
            "trajectory_id": trajectory_id,
            "callsign": str(schedule_row.callsign_clean),
            "schedule_flight_key_hint": getattr(schedule_row, "flight_key", None),
            "schedule_source_hint": getattr(schedule_row, "schedule_source_name", None),
            "candidate_scheduled_dep": getattr(schedule_row, "scheduled_dep_ref", pd.NaT),
            "candidate_scheduled_arr": getattr(schedule_row, "scheduled_arr_ref", pd.NaT),
            "candidate_origin": getattr(schedule_row, "origin", None),
            "candidate_destination": getattr(schedule_row, "destination", None),
            "raw_window_points": int(len(df_window)),
            "downsampled_points": int(len(df_sampled)),
            "feature_extraction_mode": "schedule_aware_direct",
        }
    )
    return row


def clean_flight_window(df: pd.DataFrame) -> pd.DataFrame:
    """Remove obvious duplicates and unusable timestamps from a scheduled window."""

    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values("timestamp", kind="stable")
    df = df.drop_duplicates()
    stale_cols = [col for col in ["latitude", "longitude", "altitude", "on_ground"] if col in df.columns]
    if stale_cols:
        same_as_prev = df[stale_cols].eq(df[stale_cols].shift()).all(axis=1)
        same_time = df["timestamp"].eq(df["timestamp"].shift())
        df = df[~(same_as_prev & ~same_time)].copy()
    return df


def downsample_flight_window(df: pd.DataFrame, schedule_row, cfg: ScheduleAwareFeatureConfig) -> pd.DataFrame:
    """Keep a compact path sketch while preserving endpoints and flight phases."""

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if len(df) <= cfg.max_points_per_flight:
        return df

    dep_ts = int(getattr(schedule_row, "scheduled_dep_ts"))
    arr_ts = int(getattr(schedule_row, "scheduled_arr_ts"))
    phase_seconds = cfg.phase_detail_minutes * 60

    keep = np.zeros(len(df), dtype=bool)
    keep[0] = True
    keep[-1] = True
    last_kept_ts: int | None = None

    for idx, ts in enumerate(df["timestamp"].to_numpy(dtype=np.int64)):
        in_phase = abs(ts - dep_ts) <= phase_seconds or abs(ts - arr_ts) <= phase_seconds
        interval = cfg.phase_interval_seconds if in_phase else cfg.downsample_interval_seconds
        if last_kept_ts is None or ts - last_kept_ts >= interval:
            keep[idx] = True
            last_kept_ts = int(ts)

    sampled = df.loc[keep].copy()
    if len(sampled) > cfg.max_points_per_flight:
        keep_idx = np.linspace(0, len(sampled) - 1, cfg.max_points_per_flight).round().astype(int)
        sampled = sampled.iloc[np.unique(keep_idx)].copy()
    return sampled.reset_index(drop=True)


def downsample_trajectory_sketch(df: pd.DataFrame, schedule_row, cfg: ScheduleAwareFeatureConfig) -> pd.DataFrame:
    """Keep a map/weather-friendly route sketch at a coarser interval.

    The feature extractor may keep denser points around takeoff/landing. The
    sketch intentionally stores fewer rows, usually one point every 10 minutes,
    while retaining endpoints and a little extra detail around departure/arrival.
    """

    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    if df.empty:
        return df

    dep_ts = int(getattr(schedule_row, "scheduled_dep_ts", int(df["timestamp"].min())))
    arr_ts = int(getattr(schedule_row, "scheduled_arr_ts", int(df["timestamp"].max())))
    phase_seconds = max(0, int(cfg.sketch_phase_detail_minutes * 60))
    cruise_interval = max(1, int(cfg.sketch_interval_seconds))
    phase_interval = max(1, int(cfg.sketch_phase_interval_seconds))

    keep = np.zeros(len(df), dtype=bool)
    keep[0] = True
    keep[-1] = True
    last_kept_ts: int | None = None

    for idx, ts in enumerate(df["timestamp"].to_numpy(dtype=np.int64)):
        in_phase = abs(int(ts) - dep_ts) <= phase_seconds or abs(int(ts) - arr_ts) <= phase_seconds
        interval = phase_interval if in_phase else cruise_interval
        if last_kept_ts is None or int(ts) - last_kept_ts >= interval:
            keep[idx] = True
            last_kept_ts = int(ts)

    sampled = df.loc[keep].copy()
    if len(sampled) > cfg.sketch_max_points_per_flight:
        keep_idx = np.linspace(0, len(sampled) - 1, cfg.sketch_max_points_per_flight).round().astype(int)
        sampled = sampled.iloc[np.unique(keep_idx)].copy()
    return sampled.reset_index(drop=True)


def build_trajectory_sketch(
    df_window: pd.DataFrame,
    schedule_row,
    feature_row: dict,
    cfg: ScheduleAwareFeatureConfig,
) -> pd.DataFrame:
    """Build the compact point-level route geometry retained by the fast path."""

    df_clean = clean_flight_window(df_window)
    if df_clean.empty:
        return pd.DataFrame()

    sketch = downsample_trajectory_sketch(df_clean, schedule_row, cfg)
    if sketch.empty:
        return pd.DataFrame()

    if "altitude" not in sketch.columns:
        if "geo_altitude" in sketch.columns:
            sketch["altitude"] = sketch["geo_altitude"]
            if "baro_altitude" in sketch.columns:
                sketch["altitude"] = sketch["altitude"].fillna(sketch["baro_altitude"])
        elif "baro_altitude" in sketch.columns:
            sketch["altitude"] = sketch["baro_altitude"]
        else:
            sketch["altitude"] = np.nan
    if "heading" not in sketch.columns:
        sketch["heading"] = sketch["true_track"] if "true_track" in sketch.columns else np.nan

    sketch = sketch.sort_values("timestamp", kind="stable").reset_index(drop=True)
    start_ts = float(sketch["timestamp"].min())
    end_ts = float(sketch["timestamp"].max())
    duration = max(0.0, end_ts - start_ts)
    elapsed_seconds = pd.to_numeric(sketch["timestamp"], errors="coerce") - start_ts
    progress = np.where(duration > 0, elapsed_seconds / duration, 0.0)

    out = pd.DataFrame(
        {
            "trajectory_id": feature_row.get("trajectory_id"),
            "flight_key": getattr(schedule_row, "flight_key", None),
            "callsign": str(getattr(schedule_row, "callsign_clean", feature_row.get("callsign", ""))),
            "icao24": sketch.get("icao24", pd.Series(pd.NA, index=sketch.index)).astype("object"),
            "point_idx": np.arange(len(sketch), dtype=np.int32),
            "timestamp": pd.to_numeric(sketch["timestamp"], errors="coerce").astype("int64"),
            "timestamp_utc": pd.to_datetime(sketch["timestamp"], unit="s", utc=True, errors="coerce"),
            "elapsed_minutes": elapsed_seconds / 60.0,
            "flight_progress": progress,
            "latitude": pd.to_numeric(sketch.get("latitude"), errors="coerce"),
            "longitude": pd.to_numeric(sketch.get("longitude"), errors="coerce"),
            "altitude": pd.to_numeric(sketch.get("altitude"), errors="coerce"),
            "velocity": pd.to_numeric(sketch.get("velocity"), errors="coerce"),
            "heading": pd.to_numeric(sketch.get("heading"), errors="coerce"),
            "vertical_rate": pd.to_numeric(sketch.get("vertical_rate"), errors="coerce"),
            "on_ground": sketch.get("on_ground", pd.Series(False, index=sketch.index)).fillna(False).astype(bool),
            "origin": getattr(schedule_row, "origin", None),
            "destination": getattr(schedule_row, "destination", None),
            "scheduled_dep": getattr(schedule_row, "scheduled_dep_ref", pd.NaT),
            "scheduled_arr": getattr(schedule_row, "scheduled_arr_ref", pd.NaT),
            "trajectory_quality_status": feature_row.get("trajectory_quality_status"),
            "trajectory_quality_score": feature_row.get("trajectory_quality_score"),
            "is_full_flight": feature_row.get("is_full_flight"),
            "source": "schedule_aware_direct",
            "sketch_interval_seconds": int(cfg.sketch_interval_seconds),
        }
    )

    valid_coord = out["latitude"].between(-90, 90) & out["longitude"].between(-180, 180)
    out = out[valid_coord | (out[["latitude", "longitude"]].isna().all(axis=1))].copy()
    return out


def infer_groundish_flags(df: pd.DataFrame) -> tuple[bool, bool]:
    altitude = pd.to_numeric(df.get("altitude", pd.Series(np.nan, index=df.index)), errors="coerce")
    on_ground = df.get("on_ground", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    first_alt = altitude.dropna().iloc[0] if altitude.notna().any() else np.nan
    last_alt = altitude.dropna().iloc[-1] if altitude.notna().any() else np.nan
    starts = bool(on_ground.head(5).any() or (pd.notna(first_alt) and first_alt < 1000))
    ends = bool(on_ground.tail(5).any() or (pd.notna(last_alt) and last_alt < 1000))
    return starts, ends


def classify_scheduled_window(df: pd.DataFrame, starts_groundish: bool, ends_groundish: bool) -> str:
    altitude = pd.to_numeric(df.get("altitude", pd.Series(np.nan, index=df.index)), errors="coerce")
    max_alt = altitude.max(skipna=True)
    has_airborne = bool(pd.notna(max_alt) and max_alt > 2500)
    if starts_groundish and ends_groundish and has_airborne:
        return "full_flight"
    if starts_groundish and has_airborne:
        return "partial_end_missing"
    if ends_groundish and has_airborne:
        return "partial_start_missing"
    if has_airborne:
        return "airborne_only"
    return "ground_only"


def compute_route_coverage(df: pd.DataFrame, max_gap_seconds: int) -> float:
    diffs = pd.to_numeric(df["timestamp"], errors="coerce").sort_values().diff().dropna()
    duration = float(df["timestamp"].max() - df["timestamp"].min()) if len(df) >= 2 else 0.0
    if duration <= 0:
        return 0.0
    large_gap_seconds = float(diffs[diffs > max_gap_seconds].sum()) if not diffs.empty else 0.0
    return float(max(0.0, min(1.0, 1.0 - large_gap_seconds / duration)))


def score_scheduled_window(status: str, route_coverage: float) -> float:
    base = {
        "full_flight": 1.0,
        "partial_end_missing": 0.78,
        "partial_start_missing": 0.72,
        "airborne_only": 0.55,
        "ground_only": 0.20,
    }.get(status, 0.10)
    return float(max(0.0, min(1.0, base * (0.5 + 0.5 * route_coverage))))


def has_mappable_geometry(df: pd.DataFrame) -> bool:
    if not {"latitude", "longitude"}.issubset(df.columns):
        return False
    coords = df[["latitude", "longitude"]].dropna()
    if len(coords) < 2:
        return False
    return bool(coords["latitude"].nunique() > 1 or coords["longitude"].nunique() > 1)


def make_trajectory_id(schedule_row) -> str:
    callsign = str(getattr(schedule_row, "callsign_clean", "UNKNOWN"))
    dep = pd.Timestamp(getattr(schedule_row, "scheduled_dep_ref", pd.NaT))
    dep_tag = dep.strftime("%Y%m%d%H%M") if not pd.isna(dep) else str(getattr(schedule_row, "schedule_row_id", "0"))
    flight_key = str(getattr(schedule_row, "flight_key", "") or "")
    suffix = re.sub(r"[^A-Za-z0-9]+", "", flight_key)[-16:] or str(getattr(schedule_row, "schedule_row_id", "0"))
    return f"{callsign}_{dep_tag}_{suffix}"


def coerce_feature_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce core feature columns to the pipeline schema-friendly dtypes."""

    if df.empty:
        return df
    df = df.copy()

    datetime_cols = ["start_time_utc", "end_time_utc", "candidate_scheduled_dep", "candidate_scheduled_arr"]
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    float_cols = [
        "timestamp",
        "dep_anchor_confidence",
        "arr_anchor_confidence",
        "trajectory_quality_score",
        "gap_fraction_of_flight",
        "route_coverage_fraction",
        "max_inter_ping_seconds",
        "median_inter_ping_seconds",
        "ping_interval_cv",
        "flight_duration",
        "trajectory_length",
        "mean_altitude",
        "altitude_variance",
        "mean_speed",
        "max_speed",
        "speed_std",
        "vertical_rate_std",
        "heading_variability",
        "enroute_weather_point_count",
        "enroute_weather_coverage_ratio",
        "enroute_temperature_mean",
        "enroute_temperature_min",
        "enroute_temperature_max",
        "enroute_wind_speed_mean",
        "enroute_wind_speed_max",
        "enroute_precipitation_mean",
        "enroute_precipitation_max",
        "enroute_weather_severity_mean",
        "enroute_weather_severity_max",
    ]
    for col in float_cols:
        if col not in df.columns and col.startswith("enroute_"):
            df[col] = pd.NA
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    int_cols = ["middle_gap_count", "holding_pattern_count", "altitude_change_count"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    bool_cols = [
        "dep_anchor_is_partial",
        "arr_anchor_is_partial",
        "is_full_flight",
        "starts_groundish",
        "ends_groundish",
        "has_altitude_spike",
        "is_mappable",
        "takeoff_detected",
        "landing_detected",
        "unstable_descent_flag",
    ]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)

    object_cols = ["trajectory_id", "icao24", "callsign", "trajectory_quality_status"]
    for col in object_cols:
        if col in df.columns:
            df[col] = df[col].astype("object")

    return df
