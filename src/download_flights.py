"""Bulk flight-level downloader using OpenSky flight interval endpoint only."""

from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from opensky_api import OpenSkyApi

from src.common import OpenSkyConfig, chunk_two_hour_windows, date_range, ensure_dir, parse_date, run_with_retry, setup_logging, utc_midnight


logger = logging.getLogger(__name__)


@dataclass
class FlightRow:
    icao24: Optional[str]
    callsign: Optional[str]
    estDepartureAirport: Optional[str]
    estArrivalAirport: Optional[str]
    firstSeen: Optional[int]
    lastSeen: Optional[int]


def _fetch_interval(api: OpenSkyApi, begin: int, end: int):
    """Fetch one 2-hour interval from OpenSky with retries."""

    def _call():
        return api.get_flights_from_interval(begin=begin, end=end)

    return run_with_retry(
        _call,
        max_attempts=5,
        base_sleep_seconds=2.0,
        logger=logger,
        operation_name=f"get_flights_from_interval({begin}, {end})",
    )


def _normalize_flights(raw_flights: List[object]) -> List[Dict]:
    rows: List[Dict] = []
    for flight in raw_flights or []:
        row = FlightRow(
            icao24=getattr(flight, "icao24", None),
            callsign=(getattr(flight, "callsign", None) or "").strip() or None,
            estDepartureAirport=getattr(flight, "estDepartureAirport", None),
            estArrivalAirport=getattr(flight, "estArrivalAirport", None),
            firstSeen=getattr(flight, "firstSeen", None),
            lastSeen=getattr(flight, "lastSeen", None),
        )
        rows.append(asdict(row))
    return rows


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {row["date"]: row for row in rows}


def _write_manifest(path: Path, rows: List[Dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["date", "flights_count", "start_time", "end_time"],
        )
        writer.writeheader()
        writer.writerows(rows)


def download_flights(
    start_date: str,
    end_date: str,
    parquet_root: str = "data/parquet/flights",
    manifest_path: str = "outputs/manifests/flight_download_manifest.csv",
) -> Path:
    """Download flight-level records for each day, split into 2-hour windows."""
    cfg = OpenSkyConfig.from_env()
    api = OpenSkyApi(username=cfg.username, password=cfg.password)

    start = parse_date(start_date)
    end = parse_date(end_date)

    manifest_file = Path(manifest_path)
    manifest_by_day = _load_manifest(manifest_file)

    updated_manifest: List[Dict[str, str]] = []

    for day in date_range(start, end):
        day_str = day.isoformat()
        out_dir = ensure_dir(Path(parquet_root) / f"date={day_str}")
        out_file = out_dir / "flights.parquet"

        if out_file.exists() and out_file.stat().st_size > 0:
            logger.info("[%s] skipping existing parquet: %s", day_str, out_file)
            row = manifest_by_day.get(
                day_str,
                {"date": day_str, "flights_count": "0", "start_time": f"{day_str}T00:00:00Z", "end_time": f"{day_str}T23:59:59Z"},
            )
            updated_manifest.append(row)
            continue

        logger.info("[%s] downloading flights in 2-hour windows", day_str)
        rows: List[Dict] = []
        for idx, (begin, end_ts) in enumerate(chunk_two_hour_windows(day), start=1):
            logger.info("[%s] window %02d/12: %s -> %s", day_str, idx, begin, end_ts)
            flights = _fetch_interval(api, begin=begin, end=end_ts)
            rows.extend(_normalize_flights(flights))

        if rows:
            df = pd.DataFrame(rows).drop_duplicates(
                subset=["icao24", "callsign", "estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"],
                keep="last",
            )
        else:
            df = pd.DataFrame(columns=["icao24", "callsign", "estDepartureAirport", "estArrivalAirport", "firstSeen", "lastSeen"])

        df.to_parquet(out_file, index=False)
        logger.info("[%s] wrote %d flights to %s", day_str, len(df), out_file)

        updated_manifest.append(
            {
                "date": day_str,
                "flights_count": str(len(df)),
                "start_time": datetime.utcfromtimestamp(utc_midnight(day)).isoformat() + "Z",
                "end_time": datetime.utcfromtimestamp(utc_midnight(day) + 86399).isoformat() + "Z",
            }
        )

    _write_manifest(manifest_file, updated_manifest)
    logger.info("wrote manifest: %s", manifest_file)
    return manifest_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download OpenSky flight-level data by 2-hour windows")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--parquet-root", default="data/parquet/flights")
    parser.add_argument("--manifest-path", default="outputs/manifests/flight_download_manifest.csv")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log_level)
    download_flights(
        start_date=args.start_date,
        end_date=args.end_date,
        parquet_root=args.parquet_root,
        manifest_path=args.manifest_path,
    )


if __name__ == "__main__":
    main()
