"""Produce simple QC summary for labeled flight-level dataset."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from src.common import ensure_dir, setup_logging


logger = logging.getLogger(__name__)


def generate_flight_qc(
    labeled_path: str = "data/processed/flights_labeled.parquet",
    output_csv: str = "outputs/qc/flight_summary.csv",
) -> Path:
    """Generate requested summary metrics into outputs/qc/flight_summary.csv."""
    df = pd.read_parquet(labeled_path)

    total_flights = int(len(df))
    unique_routes = int(df[["estDepartureAirport", "estArrivalAirport"]].drop_duplicates().shape[0])
    unique_aircraft = int(df["icao24"].nunique(dropna=True))
    avg_duration = float(df["actual_duration_seconds"].mean()) if total_flights else 0.0
    late_percentage = float((df["delay_label"] == "late").mean() * 100.0) if total_flights else 0.0

    summary = pd.DataFrame(
        [
            {
                "total_flights": total_flights,
                "unique_routes": unique_routes,
                "unique_aircraft": unique_aircraft,
                "avg_duration": round(avg_duration, 3),
                "late_percentage": round(late_percentage, 3),
            }
        ]
    )

    out = Path(output_csv)
    ensure_dir(out.parent)
    summary.to_csv(out, index=False)
    logger.info("wrote flight QC summary: %s", out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate flight-level QC summary csv")
    parser.add_argument("--labeled-path", default="data/processed/flights_labeled.parquet")
    parser.add_argument("--output-csv", default="outputs/qc/flight_summary.csv")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    generate_flight_qc(args.labeled_path, args.output_csv)


if __name__ == "__main__":
    main()
