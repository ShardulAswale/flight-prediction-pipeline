"""Build route median-duration baseline from OpenSky flight parquet files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.common import ensure_dir, setup_logging

logger = logging.getLogger(__name__)


def _list_flight_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.parquet"))


def build_baseline(
    flights_root: str,
    output_path: str,
    min_flights_per_route: int = 1,
) -> Path:
    """Compute route-level median duration baseline."""
    files = _list_flight_files(Path(flights_root))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {flights_root}")

    frames = []
    for file in files:
        df = pd.read_parquet(
            file,
            columns=["estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"],
        )
        frames.append(df)

    flights = pd.concat(frames, ignore_index=True)
    flights = flights.dropna(subset=["estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"])
    flights["actual_duration_seconds"] = flights["lastSeen"] - flights["firstSeen"]
    flights = flights[flights["actual_duration_seconds"] >= 0]

    baseline = (
        flights.groupby(["estDepartureAirport", "estArrivalAirport"], as_index=False)
        .agg(
            median_duration_seconds=("actual_duration_seconds", "median"),
            flight_count=("actual_duration_seconds", "size"),
        )
    )
    baseline = baseline[baseline["flight_count"] >= min_flights_per_route]

    out = Path(output_path)
    ensure_dir(out.parent)
    baseline.to_parquet(out, index=False)
    logger.info("Wrote baseline with %d routes to %s", len(baseline), out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build route baseline from OpenSky flights")
    parser.add_argument("--flights-root", required=True, help="Path containing OpenSky flight parquet files")
    parser.add_argument("--output-path", default="data/processed/route_baseline.parquet")
    parser.add_argument("--min-flights-per-route", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    build_baseline(args.flights_root, args.output_path, args.min_flights_per_route)


if __name__ == "__main__":
    main()
