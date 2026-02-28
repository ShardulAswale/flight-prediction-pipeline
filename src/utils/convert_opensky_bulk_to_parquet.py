"""Convert OpenSky bulk AVRO/CSV files into normalized hourly Parquet partitions."""

from __future__ import annotations

import argparse
import csv
import gzip
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)

TARGET_COLUMNS = [
    "time",
    "icao24",
    "callsign",
    "lat",
    "lon",
    "velocity",
    "heading",
    "vertrate",
    "onground",
    "baroaltitude",
    "geoaltitude",
    "lastposupdate",
    "lastcontact",
    "origin_country",
    "squawk",
]

COLUMN_ALIASES: Dict[str, str] = {
    "time": "time",
    "icao24": "icao24",
    "callsign": "callsign",
    "lat": "lat",
    "latitude": "lat",
    "lon": "lon",
    "longitude": "lon",
    "velocity": "velocity",
    "heading": "heading",
    "true_track": "heading",
    "track": "heading",
    "vertrate": "vertrate",
    "vertical_rate": "vertrate",
    "onground": "onground",
    "on_ground": "onground",
    "baroaltitude": "baroaltitude",
    "baro_altitude": "baroaltitude",
    "geoaltitude": "geoaltitude",
    "geo_altitude": "geoaltitude",
    "lastposupdate": "lastposupdate",
    "last_position": "lastposupdate",
    "lastcontact": "lastcontact",
    "last_contact": "lastcontact",
    "origin_country": "origin_country",
    "origincountry": "origin_country",
    "squawk": "squawk",
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def read_avro(path: Path) -> pd.DataFrame:
    """Read AVRO (optionally gzipped) into a DataFrame."""
    try:
        from fastavro import reader as avro_reader
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("fastavro is required to parse AVRO input files") from exc

    records: List[Dict] = []
    with _open_maybe_gzip(path) as file_obj:
        for row in avro_reader(file_obj):
            records.append(dict(row))
    return pd.DataFrame(records)


def read_csv(path: Path) -> pd.DataFrame:
    """Read CSV (optionally gzipped) into a DataFrame."""
    return pd.read_csv(path, compression="infer")


def parse_hours(hours_input: Optional[str]) -> List[int]:
    """Parse hour selection string."""
    if not hours_input or hours_input.lower() == "all":
        return list(range(24))

    values: List[int] = []
    for part in hours_input.split(","):
        part = part.strip()
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            values.extend(range(int(start_s), int(end_s) + 1))
        else:
            values.append(int(part))

    hours = sorted(set(values))
    for hour in hours:
        if hour < 0 or hour > 23:
            raise ValueError(f"Hour out of range: {hour}")
    return hours


def find_input_files(raw_date_dir: Path, file_format: str, hours: Iterable[int]) -> List[Path]:
    """Locate hourly source files in the raw date directory."""
    candidates = sorted(raw_date_dir.glob("*"))
    selected_hours = {f"{hour:02d}" for hour in hours}

    matches: List[Path] = []
    for path in candidates:
        name_lower = path.name.lower()
        if file_format == "avro" and ".avro" not in name_lower:
            continue
        if file_format == "csv" and ".csv" not in name_lower:
            continue

        # Keep files that include hour marker "HH" in file stem or fallback all.
        stem = path.stem
        if stem.endswith(".avro") or stem.endswith(".csv"):
            stem = Path(stem).stem
        hour_tokens = {stem[-2:], path.name[:2]}
        if selected_hours.intersection(hour_tokens):
            matches.append(path)

    if not matches:
        logger.warning("No files matched explicit hour patterns; using all files matching format in %s", raw_date_dir)
        for path in candidates:
            lower = path.name.lower()
            if (file_format == "avro" and ".avro" in lower) or (file_format == "csv" and ".csv" in lower):
                matches.append(path)

    return sorted(matches)


def _convert_epoch_series(series: pd.Series) -> pd.Series:
    """Convert likely epoch numeric timestamps to UTC pandas datetime."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.dropna().empty:
        return pd.to_datetime(series, errors="coerce", utc=True)

    # Heuristic: values > 1e11 are likely milliseconds.
    unit = "ms" if numeric.dropna().median() > 1e11 else "s"
    return pd.to_datetime(numeric, unit=unit, errors="coerce", utc=True)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source columns to target schema names and log missing fields."""
    rename_map: Dict[str, str] = {}
    for column in df.columns:
        normalized = _normalize_name(column)
        target = COLUMN_ALIASES.get(normalized)
        if target:
            rename_map[column] = target

    out_df = df.rename(columns=rename_map)

    missing = [column for column in TARGET_COLUMNS if column not in out_df.columns]
    if missing:
        logger.info("Missing target columns in source: %s", ", ".join(missing))

    keep_cols = [column for column in TARGET_COLUMNS if column in out_df.columns]
    out_df = out_df[keep_cols].copy()

    for ts_col in ["time", "lastposupdate", "lastcontact"]:
        if ts_col in out_df.columns:
            out_df[ts_col] = _convert_epoch_series(out_df[ts_col])

    if "icao24" in out_df.columns:
        out_df["icao24"] = out_df["icao24"].astype(str).str.strip().str.lower()
    if "callsign" in out_df.columns:
        out_df["callsign"] = out_df["callsign"].astype(str).str.strip()

    return out_df


def _infer_hour_from_filename(path: Path) -> Optional[int]:
    for token in [path.stem[-2:], path.name[:2]]:
        if token.isdigit() and 0 <= int(token) <= 23:
            return int(token)
    return None


def convert_opensky_bulk_to_parquet(
    date: str,
    file_format: str,
    raw_root: str = "data/raw/opensky_state_vectors",
    parquet_root: str = "data/parquet/opensky_state_vectors",
    manifest_dir: str = "outputs/manifests",
    hours: Optional[Iterable[int]] = None,
) -> Path:
    """Convert bulk files to partitioned parquet and write per-hour row manifest."""
    if file_format not in {"avro", "csv"}:
        raise ValueError("file_format must be 'avro' or 'csv'")

    hours = list(hours) if hours is not None else list(range(24))

    raw_date_dir = Path(raw_root) / date
    parquet_date_dir = Path(parquet_root) / date
    parquet_date_dir.mkdir(parents=True, exist_ok=True)

    input_files = find_input_files(raw_date_dir, file_format=file_format, hours=hours)
    if not input_files:
        raise FileNotFoundError(f"No {file_format} files found in {raw_date_dir}")

    hourly_frames: Dict[int, List[pd.DataFrame]] = {}

    for path in input_files:
        logger.info("Reading %s", path)
        df = read_avro(path) if file_format == "avro" else read_csv(path)
        norm_df = normalize_columns(df)

        if "time" in norm_df.columns and norm_df["time"].notna().any():
            source_hours = norm_df["time"].dt.hour.fillna(-1).astype(int)
            for hour, hour_df in norm_df.groupby(source_hours):
                if hour < 0:
                    continue
                hourly_frames.setdefault(int(hour), []).append(hour_df)
        else:
            inferred_hour = _infer_hour_from_filename(path)
            if inferred_hour is None:
                logger.warning("Unable to infer hour for %s; skipping", path)
                continue
            hourly_frames.setdefault(inferred_hour, []).append(norm_df)

    manifest_rows: List[Dict[str, str]] = []

    for hour in sorted(hourly_frames.keys()):
        hour_str = f"{hour:02d}"
        hour_dir = parquet_date_dir / f"hour={hour_str}"
        hour_dir.mkdir(parents=True, exist_ok=True)
        output_path = hour_dir / "part-000.parquet"

        combined = pd.concat(hourly_frames[hour], ignore_index=True)
        combined.to_parquet(output_path, index=False)

        manifest_rows.append(
            {
                "date": date,
                "hour": hour_str,
                "parquet_path": str(output_path),
                "rows": str(len(combined)),
            }
        )
        logger.info("Wrote %s rows to %s", len(combined), output_path)

    parquet_manifest_path = Path(manifest_dir) / f"parquet_manifest_{date}.csv"
    parquet_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with parquet_manifest_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["date", "hour", "parquet_path", "rows"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info("Wrote parquet manifest: %s", parquet_manifest_path)
    return parquet_manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for conversion utility."""
    parser = argparse.ArgumentParser(description="Convert OpenSky bulk AVRO/CSV files to Parquet.")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--format", dest="file_format", choices=["avro", "csv"], default="avro")
    parser.add_argument(
        "--raw-root",
        default="data/raw/opensky_state_vectors",
        help="Root raw directory for downloaded files",
    )
    parser.add_argument(
        "--parquet-root",
        default="data/parquet/opensky_state_vectors",
        help="Root parquet output directory",
    )
    parser.add_argument("--manifest-dir", default="outputs/manifests")
    parser.add_argument("--hours", default="all", help="Hours list/range string")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    hours = parse_hours(args.hours)
    convert_opensky_bulk_to_parquet(
        date=args.date,
        file_format=args.file_format,
        raw_root=args.raw_root,
        parquet_root=args.parquet_root,
        manifest_dir=args.manifest_dir,
        hours=hours,
    )


if __name__ == "__main__":
    main()
