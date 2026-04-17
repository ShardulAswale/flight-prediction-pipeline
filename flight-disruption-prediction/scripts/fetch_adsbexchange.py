import argparse
import io
import logging
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import ensure_dir, load_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("fetch_adsbexchange")

DEFAULT_URL_TEMPLATE = "https://www.adsbexchange.com/data/adsbexchange_{date}_0000.zip"

OPENSKY_STATE_COLUMNS = [
    "icao24",
    "callsign",
    "origin_country",
    "time_position",
    "last_contact",
    "longitude",
    "latitude",
    "baro_altitude",
    "on_ground",
    "velocity",
    "true_track",
    "vertical_rate",
    "sensors",
    "geo_altitude",
    "squawk",
    "spi",
    "position_source",
    "timestamp",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ADS-B Exchange nightly ZIP dumps or convert local ZIP archives to parquet."
    )
    parser.add_argument("--date", help="Single date to fetch in YYYYMMDD format.")
    parser.add_argument("--from-date", dest="from_date", help="Start date in YYYYMMDD format.")
    parser.add_argument("--to-date", dest="to_date", help="End date in YYYYMMDD format.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to the repo config file.",
    )
    parser.add_argument(
        "--archive-dir",
        help="Directory where downloaded ZIP archives should be stored. Defaults to data/raw/adsbexchange_archives.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where converted parquet files should be written. Defaults to paths.raw_data_dir from config.",
    )
    parser.add_argument(
        "--url-template",
        default=DEFAULT_URL_TEMPLATE,
        help="Download URL template. Use {date} as the YYYYMMDD placeholder.",
    )
    parser.add_argument(
        "--zip-path",
        action="append",
        default=[],
        help="Local ZIP archive to convert. Can be passed multiple times.",
    )
    parser.add_argument(
        "--zip-dir",
        help="Directory containing local ADS-B Exchange ZIP archives to convert.",
    )
    parser.add_argument(
        "--convert",
        action="store_true",
        help="Convert each downloaded ZIP into states_YYYYMMDD.parquet after download.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload archives and overwrite parquet outputs if they already exist.",
    )
    return parser.parse_args()


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y%m%d")


def expand_dates(args: argparse.Namespace) -> list[str]:
    if args.date:
        return [parse_date(args.date).strftime("%Y%m%d")]
    if args.from_date and args.to_date:
        start = parse_date(args.from_date)
        end = parse_date(args.to_date)
        if end < start:
            raise ValueError("--to-date must be on or after --from-date.")
        days = (end - start).days + 1
        return [(start + timedelta(days=offset)).strftime("%Y%m%d") for offset in range(days)]
    raise ValueError("Provide either --date or both --from-date and --to-date.")


def infer_date_from_path(path: Path) -> str:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    raise ValueError(f"Could not infer YYYYMMDD date from archive name: {path.name}")


def resolve_local_archives(args: argparse.Namespace) -> list[Path]:
    archives = [Path(value) for value in args.zip_path]
    if args.zip_dir:
        archives.extend(sorted(Path(args.zip_dir).glob("*.zip")))
    return archives


def normalize_adsbexchange_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {
        "hex": "icao24",
        "icao": "icao24",
        "flight": "callsign",
        "callsign": "callsign",
        "lat": "latitude",
        "lng": "longitude",
        "lon": "longitude",
        "alt": "baro_altitude",
        "altitude": "baro_altitude",
        "track": "true_track",
        "heading": "true_track",
        "speed": "velocity",
        "gs": "velocity",
        "vertrate": "vertical_rate",
        "vertical_rate": "vertical_rate",
        "squawk": "squawk",
        "ts": "timestamp",
        "time": "timestamp",
        "timestamp": "timestamp",
    }
    df = df.rename(columns=rename_map)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").round().astype("Int64")

    for col in OPENSKY_STATE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["icao24"] = df["icao24"].astype("string").str.lower().str.strip()
    df["callsign"] = df["callsign"].astype("string").str.strip()
    df["origin_country"] = df["origin_country"].fillna("ADS-B Exchange")
    df["on_ground"] = df["on_ground"].fillna(False).astype(bool)
    df["time_position"] = df["time_position"].fillna(df["timestamp"])
    df["last_contact"] = df["last_contact"].fillna(df["timestamp"])

    ordered_columns = OPENSKY_STATE_COLUMNS + [col for col in df.columns if col not in OPENSKY_STATE_COLUMNS]
    return df[ordered_columns]


def download_archive(date_str: str, archive_dir: Path, url_template: str, overwrite: bool) -> Path:
    archive_path = archive_dir / f"adsbexchange_{date_str}.zip"
    if archive_path.exists() and not overwrite:
        logger.info("Archive already exists, skipping download: %s", archive_path)
        return archive_path

    url = url_template.format(date=date_str)
    logger.info("Downloading %s", url)
    response = requests.get(url, timeout=120)
    if response.status_code == 404 and url_template == DEFAULT_URL_TEMPLATE:
        raise RuntimeError(
            "ADS-B Exchange returned 404 for the default public URL pattern. "
            "Their historical files are typically provided via secure/custom download links, "
            "not a predictable public path. Use --url-template with the correct provider URL "
            "or download ZIPs manually and rerun with --zip-path/--zip-dir --convert."
        )
    response.raise_for_status()
    archive_path.write_bytes(response.content)
    logger.info("Saved archive to %s", archive_path)
    return archive_path


def convert_archive(archive_path: Path, output_dir: Path, date_str: str, overwrite: bool) -> Path:
    output_path = output_dir / f"states_{date_str}.parquet"
    if output_path.exists() and not overwrite:
        logger.info("Parquet already exists, skipping conversion: %s", output_path)
        return output_path

    dataframes: list[pd.DataFrame] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".csv"):
                continue
            with archive.open(member) as handle:
                csv_bytes = handle.read()
                frame = pd.read_csv(io.BytesIO(csv_bytes))
                dataframes.append(normalize_adsbexchange_df(frame))

    if not dataframes:
        raise RuntimeError(f"No CSV members found in archive: {archive_path}")

    combined = pd.concat(dataframes, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    logger.info("Saved parquet to %s", output_path)
    return output_path


def main() -> None:
    try:
        args = parse_args()
        config = load_config(args.config)

        raw_data_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
        archive_dir = Path(args.archive_dir) if args.archive_dir else raw_data_dir / "adsbexchange_archives"
        output_dir = Path(args.output_dir) if args.output_dir else raw_data_dir
        ensure_dir(archive_dir)
        ensure_dir(output_dir)

        local_archives = resolve_local_archives(args)
        if local_archives:
            if not args.convert:
                raise ValueError("Local ZIP conversion requires --convert.")
            for archive_path in local_archives:
                date_str = infer_date_from_path(archive_path)
                convert_archive(archive_path, output_dir, date_str, args.overwrite)
            return

        dates = expand_dates(args)
        for date_str in dates:
            archive_path = download_archive(date_str, archive_dir, args.url_template, args.overwrite)
            if args.convert:
                convert_archive(archive_path, output_dir, date_str, args.overwrite)
    except requests.RequestException as exc:
        logger.error("ADS-B Exchange download failed: %s", exc)
        raise SystemExit(1) from exc
    except (ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
