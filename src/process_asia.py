"""Process OpenSky flight parquet for Asia using baseline-derived labels."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.common import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

ASIA_PREFIXES = (
    "V",  # South/Southeast Asia blocks
    "R",  # East Asia blocks
    "Z",  # China region blocks
    "O",  # Middle East blocks
    "U",  # Central Asia blocks
)


def _load_flights(flights_root: str) -> pd.DataFrame:
    files = sorted(Path(flights_root).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {flights_root}")

    frames: List[pd.DataFrame] = []
    for file in files:
        frames.append(
            pd.read_parquet(
                file,
                columns=["icao24", "callsign", "estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"],
            )
        )
    return pd.concat(frames, ignore_index=True)


def process_asia(
    flights_root: str,
    baseline_path: str,
    output_path: str = "data/asia/asia_processed.parquet",
) -> Path:
    flights = _load_flights(flights_root)
    baseline = pd.read_parquet(baseline_path)

    flights = flights[flights["estDepartureAirport"].fillna("").str.startswith(ASIA_PREFIXES)]
    flights = flights[flights["estArrivalAirport"].fillna("").str.startswith(ASIA_PREFIXES)]

    flights["actual_duration_seconds"] = flights["lastSeen"] - flights["firstSeen"]

    merged = flights.merge(
        baseline[["estDepartureAirport", "estArrivalAirport", "median_duration_seconds"]],
        on=["estDepartureAirport", "estArrivalAirport"],
        how="left",
    )

    merged["delay_minutes"] = (merged["actual_duration_seconds"] - merged["median_duration_seconds"]) / 60.0
    merged["delay_label"] = "Normal"
    merged.loc[merged["estArrivalAirport"].isna() | merged["lastSeen"].isna(), "delay_label"] = "Cancelled"
    merged.loc[
        (merged["delay_label"] != "Cancelled")
        & (merged["actual_duration_seconds"] > merged["median_duration_seconds"] * 1.15),
        "delay_label",
    ] = "Late"

    out_df = pd.DataFrame(
        {
            "flight_id": merged["icao24"].astype(str) + "_" + merged["firstSeen"].astype(str),
            "region": "Asia",
            "departure_airport": merged["estDepartureAirport"],
            "arrival_airport": merged["estArrivalAirport"],
            "actual_departure_time": pd.to_datetime(merged["firstSeen"], unit="s", utc=True, errors="coerce"),
            "actual_arrival_time": pd.to_datetime(merged["lastSeen"], unit="s", utc=True, errors="coerce"),
            "actual_duration_seconds": merged["actual_duration_seconds"],
            "baseline_duration_seconds": merged["median_duration_seconds"],
            "delay_minutes": merged["delay_minutes"],
            "delay_label": merged["delay_label"],
            "label_source": "baseline",
        }
    )

    out = Path(output_path)
    ensure_dir(out.parent)
    out_df.to_parquet(out, index=False)
    logger.info("Wrote Asia processed rows=%d to %s", len(out_df), out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process Asia OpenSky flights with baseline-derived labels")
    parser.add_argument("--flights-root", required=True)
    parser.add_argument("--baseline-path", required=True)
    parser.add_argument("--output-path", default="data/asia/asia_processed.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    process_asia(args.flights_root, args.baseline_path, args.output_path)


if __name__ == "__main__":
    main()
