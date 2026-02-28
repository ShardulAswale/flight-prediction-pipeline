"""Label OpenSky flights as on_time or late using route-duration baselines."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.common import ensure_dir, setup_logging


logger = logging.getLogger(__name__)


def _list_flight_files(parquet_root: Path) -> List[Path]:
    return sorted(parquet_root.glob("date=*/flights.parquet"))


def label_flights(
    flights_parquet_root: str = "data/parquet/flights",
    baseline_path: str = "data/processed/route_baseline.parquet",
    output_path: str = "data/processed/flights_labeled.parquet",
) -> Path:
    """Merge flights with route baseline and derive delay label."""
    flight_files = _list_flight_files(Path(flights_parquet_root))
    if not flight_files:
        raise FileNotFoundError(f"No flights parquet files found under {flights_parquet_root}")

    baseline = pd.read_parquet(baseline_path)

    frames = []
    for file in flight_files:
        df = pd.read_parquet(
            file,
            columns=[
                "icao24",
                "callsign",
                "estDepartureAirport",
                "estArrivalAirport",
                "firstSeen",
                "lastSeen",
            ],
        )
        frames.append(df)

    flights = pd.concat(frames, ignore_index=True)
    flights = flights.dropna(subset=["firstSeen", "lastSeen"])
    flights["actual_duration_seconds"] = flights["lastSeen"] - flights["firstSeen"]
    flights = flights[flights["actual_duration_seconds"] >= 0]

    labeled = flights.merge(
        baseline[["estDepartureAirport", "estArrivalAirport", "median_duration_seconds"]],
        on=["estDepartureAirport", "estArrivalAirport"],
        how="left",
    )
    labeled = labeled.dropna(subset=["median_duration_seconds"])
    labeled["late_threshold_seconds"] = labeled["median_duration_seconds"] * 1.15
    labeled["delay_label"] = labeled["actual_duration_seconds"].gt(labeled["late_threshold_seconds"]).map(
        {True: "late", False: "on_time"}
    )

    out_cols = [
        "icao24",
        "callsign",
        "estDepartureAirport",
        "estArrivalAirport",
        "firstSeen",
        "lastSeen",
        "actual_duration_seconds",
        "median_duration_seconds",
        "delay_label",
    ]
    labeled = labeled[out_cols]

    out_file = Path(output_path)
    ensure_dir(out_file.parent)
    labeled.to_parquet(out_file, index=False)
    logger.info("wrote labeled flights (%d rows): %s", len(labeled), out_file)
    return out_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label flights as on_time/late using route baseline")
    parser.add_argument("--flights-parquet-root", default="data/parquet/flights")
    parser.add_argument("--baseline-path", default="data/processed/route_baseline.parquet")
    parser.add_argument("--output-path", default="data/processed/flights_labeled.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    label_flights(args.flights_parquet_root, args.baseline_path, args.output_path)


if __name__ == "__main__":
    main()
