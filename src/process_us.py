"""Process BTS On-Time Performance CSV files into labeled US parquet."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.common import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

US_COLUMNS = [
    "FlightDate",
    "Origin",
    "Dest",
    "CRSDepTime",
    "CRSArrTime",
    "DepTime",
    "ArrTime",
    "ArrDelay",
    "Cancelled",
]


def _hhmm_to_minutes(value) -> float:
    if pd.isna(value):
        return float("nan")
    value_int = int(float(value))
    hh = value_int // 100
    mm = value_int % 100
    return hh * 60 + mm


def _compute_duration(dep_min: pd.Series, arr_min: pd.Series) -> pd.Series:
    duration = arr_min - dep_min
    duration = duration.where(duration >= 0, duration + 24 * 60)
    return duration * 60


def process_us(input_glob: str, output_path: str = "data/us/us_processed.parquet") -> Path:
    files = sorted(Path().glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No files matched glob: {input_glob}")

    frames: List[pd.DataFrame] = []
    for file in files:
        df = pd.read_csv(file, usecols=lambda c: c in US_COLUMNS)
        missing = [c for c in US_COLUMNS if c not in df.columns]
        for col in missing:
            df[col] = pd.NA

        df["FlightDate"] = pd.to_datetime(df["FlightDate"], errors="coerce")

        for col in ["CRSDepTime", "CRSArrTime", "DepTime", "ArrTime"]:
            minutes = df[col].apply(_hhmm_to_minutes)
            base = df["FlightDate"]
            df[f"{col}_dt"] = base + pd.to_timedelta(minutes, unit="m")

        df["scheduled_duration_seconds"] = _compute_duration(
            df["CRSDepTime"].apply(_hhmm_to_minutes), df["CRSArrTime"].apply(_hhmm_to_minutes)
        )
        df["actual_duration_seconds"] = _compute_duration(
            df["DepTime"].apply(_hhmm_to_minutes), df["ArrTime"].apply(_hhmm_to_minutes)
        )

        df["delay_minutes"] = pd.to_numeric(df["ArrDelay"], errors="coerce")
        df["delay_label"] = "Normal"
        df.loc[pd.to_numeric(df["Cancelled"], errors="coerce") == 1, "delay_label"] = "Cancelled"
        df.loc[
            (pd.to_numeric(df["Cancelled"], errors="coerce") != 1)
            & (pd.to_numeric(df["ArrDelay"], errors="coerce") > 15),
            "delay_label",
        ] = "Late"

        df["region"] = "US"
        df["label_source"] = "official"

        processed = pd.DataFrame(
            {
                "flight_id": df.index.astype(str),
                "region": df["region"],
                "departure_airport": df["Origin"],
                "arrival_airport": df["Dest"],
                "actual_departure_time": df["DepTime_dt"],
                "actual_arrival_time": df["ArrTime_dt"],
                "actual_duration_seconds": df["actual_duration_seconds"],
                "baseline_duration_seconds": df["scheduled_duration_seconds"],
                "delay_minutes": df["delay_minutes"],
                "delay_label": df["delay_label"],
                "label_source": df["label_source"],
            }
        )
        frames.append(processed)

    out_df = pd.concat(frames, ignore_index=True)
    out = Path(output_path)
    ensure_dir(out.parent)
    out_df.to_parquet(out, index=False)
    logger.info("Wrote US processed rows=%d to %s", len(out_df), out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process BTS On-Time Performance CSV files")
    parser.add_argument("--input-glob", required=True, help="Glob pattern to BTS CSV files")
    parser.add_argument("--output-path", default="data/us/us_processed.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    process_us(args.input_glob, args.output_path)


if __name__ == "__main__":
    main()
