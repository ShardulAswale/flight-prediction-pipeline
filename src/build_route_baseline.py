"""Build route-level baseline durations from OpenSky flight parquet files."""

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


def build_route_baseline(
    flights_parquet_root: str = "data/parquet/flights",
    output_path: str = "data/processed/route_baseline.parquet",
) -> Path:
    """Compute route baseline median/mean durations across all available flights."""
    root = Path(flights_parquet_root)
    files = _list_flight_files(root)
    if not files:
        raise FileNotFoundError(f"No flights parquet files found under {root}")

    frames = []
    for file in files:
        df = pd.read_parquet(
            file,
            columns=["estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"],
        )
        frames.append(df)

    all_flights = pd.concat(frames, ignore_index=True)
    all_flights = all_flights.dropna(subset=["estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"])  # route present
    all_flights["actual_duration_seconds"] = all_flights["lastSeen"] - all_flights["firstSeen"]
    all_flights = all_flights[all_flights["actual_duration_seconds"] >= 0]

    baseline = (
        all_flights.groupby(["estDepartureAirport", "estArrivalAirport"], as_index=False)
        .agg(
            median_duration_seconds=("actual_duration_seconds", "median"),
            mean_duration_seconds=("actual_duration_seconds", "mean"),
            flight_count=("actual_duration_seconds", "size"),
        )
        .sort_values("flight_count", ascending=False)
    )

    out_file = Path(output_path)
    ensure_dir(out_file.parent)
    baseline.to_parquet(out_file, index=False)
    logger.info("wrote route baseline with %d routes: %s", len(baseline), out_file)
    return out_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build route baseline durations from flights parquet")
    parser.add_argument("--flights-parquet-root", default="data/parquet/flights")
    parser.add_argument("--output-path", default="data/processed/route_baseline.parquet")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    build_route_baseline(args.flights_parquet_root, args.output_path)


if __name__ == "__main__":
    main()
