"""Unify US/EU/Asia processed datasets and generate global QC summary."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.common import ensure_dir, setup_logging

logger = logging.getLogger(__name__)

UNIFIED_COLUMNS = [
    "flight_id",
    "region",
    "departure_airport",
    "arrival_airport",
    "actual_departure_time",
    "actual_arrival_time",
    "actual_duration_seconds",
    "baseline_duration_seconds",
    "delay_minutes",
    "delay_label",
    "label_source",
]


def _load_and_standardize(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for col in UNIFIED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[UNIFIED_COLUMNS].copy()


def unify_datasets(
    us_path: str = "data/us/us_processed.parquet",
    eu_path: str = "data/eu/eu_processed.parquet",
    asia_path: str = "data/asia/asia_processed.parquet",
    output_path: str = "data/unified/global_flights.parquet",
    qc_path: str = "outputs/qc/global_summary.csv",
) -> Path:
    us = _load_and_standardize(us_path)
    eu = _load_and_standardize(eu_path)
    asia = _load_and_standardize(asia_path)

    global_df = pd.concat([us, eu, asia], ignore_index=True)
    out = Path(output_path)
    ensure_dir(out.parent)
    global_df.to_parquet(out, index=False)

    total_flights = len(global_df)
    flights_per_region = global_df["region"].value_counts(dropna=False).to_dict()
    delay_dist = (
        global_df.groupby(["region", "delay_label"]).size().rename("count").reset_index().sort_values(["region", "count"], ascending=[True, False])
    )
    cancelled_pct = float((global_df["delay_label"] == "Cancelled").mean() * 100.0) if total_flights else 0.0

    qc_df = pd.DataFrame(
        [
            {
                "total_flights": total_flights,
                "flights_per_region": json.dumps(flights_per_region),
                "delay_distribution_per_region": json.dumps(
                    {
                        region: grp.set_index("delay_label")["count"].to_dict()
                        for region, grp in delay_dist.groupby("region")
                    }
                ),
                "cancelled_percentage": round(cancelled_pct, 3),
            }
        ]
    )
    qc_out = Path(qc_path)
    ensure_dir(qc_out.parent)
    qc_df.to_csv(qc_out, index=False)

    logger.info("Wrote unified dataset rows=%d to %s", len(global_df), out)
    logger.info("Wrote global QC summary to %s", qc_out)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unify regional processed datasets and emit QC summary")
    parser.add_argument("--us-path", default="data/us/us_processed.parquet")
    parser.add_argument("--eu-path", default="data/eu/eu_processed.parquet")
    parser.add_argument("--asia-path", default="data/asia/asia_processed.parquet")
    parser.add_argument("--output-path", default="data/unified/global_flights.parquet")
    parser.add_argument("--qc-path", default="outputs/qc/global_summary.csv")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    unify_datasets(args.us_path, args.eu_path, args.asia_path, args.output_path, args.qc_path)


if __name__ == "__main__":
    main()
