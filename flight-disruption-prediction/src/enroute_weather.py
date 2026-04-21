"""En-route weather features from retained trajectory sketches.

This module intentionally works from ``trajectory_sketches.parquet`` rather
than full raw ADS-B. The sketch table keeps enough route geometry for maps,
route deviation, and weather-at-aircraft-location without reintroducing the
full point-level storage cost.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.utils import ensure_dir

logger = logging.getLogger(__name__)


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_VARIABLES = (
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
)


@dataclass(frozen=True)
class EnrouteWeatherConfig:
    """Controls for weather lookup at aircraft route-sketch points."""

    enabled: bool = False
    provider: str = "open_meteo"
    cache_dir: str = "data/raw/enroute_weather_cache"
    max_flights: int | None = None
    max_points_per_flight: int = 24
    request_sleep_seconds: float = 0.05
    round_latlon_decimals: int = 2
    round_time: str = "1h"
    timeout_seconds: int = 30
    force_refresh: bool = False


def add_enroute_weather_features(
    df_flights: pd.DataFrame,
    *,
    sketches_path: str | Path,
    config: EnrouteWeatherConfig,
) -> pd.DataFrame:
    """Attach flight-level en-route weather aggregates to an ML feature table."""

    if not config.enabled:
        logger.info("En-route weather disabled; skipping route-point weather enrichment.")
        return df_flights

    sketches_path = Path(sketches_path)
    if not sketches_path.exists():
        logger.warning("Trajectory sketches not found at %s; skipping en-route weather.", sketches_path)
        return df_flights

    if "trajectory_id" not in df_flights.columns:
        logger.warning("Flight table has no trajectory_id; skipping en-route weather.")
        return df_flights

    sketches = pd.read_parquet(sketches_path)
    if sketches.empty or "trajectory_id" not in sketches.columns:
        logger.warning("Trajectory sketches are empty or missing trajectory_id; skipping en-route weather.")
        return df_flights

    wanted_ids = set(df_flights["trajectory_id"].dropna().astype(str))
    sketches = sketches[sketches["trajectory_id"].astype(str).isin(wanted_ids)].copy()
    if config.max_flights is not None and config.max_flights > 0:
        keep_ids = set(df_flights["trajectory_id"].dropna().astype(str).head(config.max_flights))
        sketches = sketches[sketches["trajectory_id"].astype(str).isin(keep_ids)].copy()

    sketches = _prepare_weather_points(sketches, config)
    if sketches.empty:
        logger.warning("No usable route-sketch weather points after cleaning.")
        return df_flights

    cache_dir = Path(config.cache_dir)
    ensure_dir(cache_dir)

    weather_rows: list[dict] = []
    grouped = sketches.groupby(["weather_lat", "weather_lon", "weather_date"], sort=False)
    logger.info(
        "Fetching/caching en-route weather for %s unique lat/lon/date groups from %s sketch points.",
        len(grouped),
        f"{len(sketches):,}",
    )
    for (lat, lon, date_value), group in grouped:
        daily = _load_or_fetch_open_meteo_day(
            latitude=float(lat),
            longitude=float(lon),
            date_str=str(date_value),
            cache_dir=cache_dir,
            config=config,
        )
        if daily.empty:
            continue
        merged = pd.merge(
            group,
            daily,
            left_on="weather_hour",
            right_on="weather_time",
            how="left",
        )
        weather_rows.append(merged)

    if not weather_rows:
        logger.warning("No en-route weather rows were matched.")
        return df_flights

    point_weather = pd.concat(weather_rows, ignore_index=True)
    agg = aggregate_enroute_weather(point_weather)
    out = df_flights.merge(agg, on="trajectory_id", how="left")
    logger.info(
        "Attached en-route weather features for %s/%s flights.",
        agg["trajectory_id"].nunique(),
        df_flights["trajectory_id"].nunique(dropna=True),
    )
    return out


def _prepare_weather_points(sketches: pd.DataFrame, config: EnrouteWeatherConfig) -> pd.DataFrame:
    required = {"trajectory_id", "latitude", "longitude", "timestamp_utc"}
    missing = required - set(sketches.columns)
    if missing:
        logger.warning("Trajectory sketches missing columns for weather lookup: %s", sorted(missing))
        return pd.DataFrame()

    df = sketches.copy()
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["trajectory_id", "timestamp_utc", "latitude", "longitude"])
    df = df[df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)].copy()
    if df.empty:
        return df

    df = df.sort_values(["trajectory_id", "timestamp_utc"], kind="stable")
    if config.max_points_per_flight > 0:
        limited_groups = [
            _limit_points_evenly(group, config.max_points_per_flight)
            for _, group in df.groupby("trajectory_id", sort=False)
        ]
        df = pd.concat(limited_groups, ignore_index=True) if limited_groups else pd.DataFrame()

    df["weather_lat"] = df["latitude"].round(config.round_latlon_decimals)
    df["weather_lon"] = df["longitude"].round(config.round_latlon_decimals)
    df["weather_hour"] = df["timestamp_utc"].dt.round(config.round_time)
    df["weather_date"] = df["weather_hour"].dt.strftime("%Y-%m-%d")
    return df


def _limit_points_evenly(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    idx = np.linspace(0, len(group) - 1, max_points).round().astype(int)
    return group.iloc[np.unique(idx)].copy()


def _cache_path(cache_dir: Path, latitude: float, longitude: float, date_str: str) -> Path:
    lat_tag = f"{latitude:.2f}".replace("-", "m").replace(".", "p")
    lon_tag = f"{longitude:.2f}".replace("-", "m").replace(".", "p")
    return cache_dir / f"open_meteo_{date_str}_{lat_tag}_{lon_tag}.json"


def _load_or_fetch_open_meteo_day(
    *,
    latitude: float,
    longitude: float,
    date_str: str,
    cache_dir: Path,
    config: EnrouteWeatherConfig,
) -> pd.DataFrame:
    path = _cache_path(cache_dir, latitude, longitude, date_str)
    if path.exists() and not config.force_refresh:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return _open_meteo_payload_to_frame(payload)
        except Exception as exc:
            logger.warning("Failed to read cached en-route weather %s: %s", path, exc)

    if config.provider.lower() != "open_meteo":
        raise ValueError(f"Unsupported en-route weather provider: {config.provider}")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": ",".join(OPEN_METEO_VARIABLES),
        "timezone": "UTC",
    }
    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=config.timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        if config.request_sleep_seconds > 0:
            time.sleep(config.request_sleep_seconds)
        return _open_meteo_payload_to_frame(payload)
    except Exception as exc:
        logger.warning("Open-Meteo request failed for %.2f, %.2f on %s: %s", latitude, longitude, date_str, exc)
        return pd.DataFrame()


def _open_meteo_payload_to_frame(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
    times = hourly.get("time")
    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({"weather_time": pd.to_datetime(times, utc=True, errors="coerce")})
    rename = {
        "temperature_2m": "enroute_temperature",
        "precipitation": "enroute_precipitation",
        "wind_speed_10m": "enroute_wind_speed",
        "wind_gusts_10m": "enroute_wind_gust",
        "cloud_cover": "enroute_cloud_cover",
    }
    for source, dest in rename.items():
        values = hourly.get(source)
        df[dest] = pd.to_numeric(pd.Series(values), errors="coerce") if values is not None else np.nan
    return df.dropna(subset=["weather_time"])


def aggregate_enroute_weather(point_weather: pd.DataFrame) -> pd.DataFrame:
    """Aggregate point-level route weather into one row per trajectory."""

    df = point_weather.copy()
    for col in ["enroute_temperature", "enroute_wind_speed", "enroute_precipitation", "enroute_cloud_cover"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["enroute_weather_severity"] = _severity_from_point_weather(df)
    df["has_weather"] = df[
        ["enroute_temperature", "enroute_wind_speed", "enroute_precipitation", "enroute_cloud_cover"]
    ].notna().any(axis=1)

    grouped = df.groupby("trajectory_id", sort=False)
    agg = grouped.agg(
        enroute_sketch_point_count=("trajectory_id", "size"),
        enroute_weather_point_count=("has_weather", "sum"),
        enroute_temperature_mean=("enroute_temperature", "mean"),
        enroute_temperature_min=("enroute_temperature", "min"),
        enroute_temperature_max=("enroute_temperature", "max"),
        enroute_wind_speed_mean=("enroute_wind_speed", "mean"),
        enroute_wind_speed_max=("enroute_wind_speed", "max"),
        enroute_precipitation_mean=("enroute_precipitation", "mean"),
        enroute_precipitation_max=("enroute_precipitation", "max"),
        enroute_weather_severity_mean=("enroute_weather_severity", "mean"),
        enroute_weather_severity_max=("enroute_weather_severity", "max"),
    ).reset_index()
    agg["enroute_weather_coverage_ratio"] = np.where(
        agg["enroute_sketch_point_count"] > 0,
        agg["enroute_weather_point_count"] / agg["enroute_sketch_point_count"],
        np.nan,
    )
    return agg.drop(columns=["enroute_sketch_point_count"])


def _severity_from_point_weather(df: pd.DataFrame) -> pd.Series:
    wind = pd.to_numeric(df.get("enroute_wind_speed"), errors="coerce")
    precip = pd.to_numeric(df.get("enroute_precipitation"), errors="coerce")
    cloud = pd.to_numeric(df.get("enroute_cloud_cover"), errors="coerce")

    conditions = [
        (wind > 55) | (precip > 8.0),
        (wind > 35) | (precip > 3.0) | (cloud > 90),
        (wind > 20) | (precip > 0.5) | (cloud > 70),
    ]
    severity = np.select(conditions, [3, 2, 1], default=0).astype(float)
    all_missing = wind.isna() & precip.isna() & cloud.isna()
    severity[all_missing.to_numpy()] = np.nan
    return pd.Series(severity, index=df.index)
