import argparse
import logging
import sys
import time
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from fastavro import reader as avro_reader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils import ensure_dir, load_config


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("fetch_opensky_samples")

DEFAULT_URL_TEMPLATE = (
    "https://s3.opensky-network.org/data-samples/states/{date}/{hour:02d}/"
    "states_{date}-{hour:02d}.avro.tar"
)

STATE_VECTOR_COLUMNS = [
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
        description="Download public OpenSky weekly sample state vectors and convert them to parquet."
    )
    parser.add_argument(
        "--sample-date",
        action="append",
        required=True,
        help="Sample day in YYYY-MM-DD format. Pass multiple times for multiple Mondays.",
    )
    parser.add_argument("--hour-start", type=int, default=0, help="First hour to fetch, inclusive.")
    parser.add_argument("--hour-end", type=int, default=23, help="Last hour to fetch, inclusive.")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "config.yaml"),
        help="Path to the repo config file.",
    )
    parser.add_argument(
        "--archive-dir",
        help="Directory where downloaded .avro.tar files should be stored.",
    )
    parser.add_argument(
        "--output-dir",
        help="Base directory for parquet output. Defaults to data/raw/opensky.",
    )
    parser.add_argument(
        "--single-output-file",
        help="Optional parquet path for one combined output file across all requested dates and hours.",
    )
    parser.add_argument(
        "--url-template",
        default=DEFAULT_URL_TEMPLATE,
        help="Template for the OpenSky sample bucket URL. Must contain {date} and {hour}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload archives and overwrite parquet output if it already exists.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel download/conversion workers to use.",
    )
    return parser.parse_args()


def normalise_state_vectors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    rename_map = {
        "time": "timestamp",
        "lat": "latitude",
        "lon": "longitude",
        "baroaltitude": "baro_altitude",
        "geoaltitude": "geo_altitude",
        "heading": "true_track",
        "vertrate": "vertical_rate",
        "onground": "on_ground",
        "lastposupdate": "time_position",
        "lastcontact": "last_contact",
    }
    df = df.rename(columns=rename_map)

    for col in STATE_VECTOR_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    numeric_cols = [
        "timestamp",
        "time_position",
        "last_contact",
        "longitude",
        "latitude",
        "baro_altitude",
        "velocity",
        "true_track",
        "vertical_rate",
        "geo_altitude",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["icao24"] = df["icao24"].astype("string").str.lower().str.strip()
    df["callsign"] = df["callsign"].astype("string").str.strip()
    df["on_ground"] = df["on_ground"].fillna(False).astype(bool)
    df["origin_country"] = df["origin_country"].fillna(pd.NA)

    return df[STATE_VECTOR_COLUMNS]


def archive_filename(date_str: str, hour: int) -> str:
    return f"states_{date_str}-{hour:02d}.avro.tar"


def download_archive(date_str: str, hour: int, archive_dir: Path, url_template: str, overwrite: bool) -> Path | None:
    archive_path = archive_dir / archive_filename(date_str, hour)
    if archive_path.exists() and not overwrite:
        logger.info("Archive already exists, skipping download: %s", archive_path)
        return archive_path

    ensure_dir(archive_dir)
    url = url_template.format(date=date_str, hour=hour)
    part_path = archive_path.with_suffix(archive_path.suffix + ".part")
    max_attempts = 4

    for attempt in range(1, max_attempts + 1):
        try:
            logger.info("Downloading %s (attempt %s/%s)", url, attempt, max_attempts)
            with requests.get(url, timeout=(30, 300), stream=True) as response:
                if response.status_code == 404:
                    logger.warning("Sample file not found for %s hour %02d", date_str, hour)
                    return None

                response.raise_for_status()
                if part_path.exists():
                    part_path.unlink()
                with part_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)

            part_path.replace(archive_path)
            logger.info("Saved archive to %s", archive_path)
            return archive_path
        except (requests.RequestException, OSError) as exc:
            if part_path.exists():
                part_path.unlink()
            if attempt == max_attempts:
                raise
            sleep_seconds = min(60, 5 * attempt)
            logger.warning(
                "Download failed for %s hour %02d on attempt %s/%s: %s. Retrying in %ss.",
                date_str,
                hour,
                attempt,
                max_attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    return None


def load_archive_frame(archive_path: Path, date_str: str, hour: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    with tarfile.open(archive_path, mode="r:*") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".avro"):
                continue
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            records = list(avro_reader(extracted))
            if records:
                frames.append(pd.json_normalize(records))

    if not frames:
        raise RuntimeError(f"No Avro records found in archive: {archive_path}")

    combined = pd.concat(frames, ignore_index=True)
    normalised = normalise_state_vectors(combined)
    normalised["sample_date"] = date_str
    normalised["sample_hour"] = hour
    return normalised


def convert_archive(archive_path: Path, output_dir: Path, date_str: str, hour: int, overwrite: bool) -> Path:
    output_path = (
        output_dir
        / f"year={date_str[:4]}"
        / f"month={date_str[5:7]}"
        / f"states_{date_str.replace('-', '')}_{hour:02d}.parquet"
    )
    if output_path.exists() and not overwrite:
        logger.info("Parquet already exists, skipping conversion: %s", output_path)
        return output_path

    ensure_dir(output_path.parent)
    normalised = load_archive_frame(archive_path, date_str, hour)
    normalised.to_parquet(output_path, index=False)
    logger.info("Saved parquet to %s", output_path)
    return output_path


def main() -> None:
    try:
        args = parse_args()
        if not 0 <= args.hour_start <= 23 or not 0 <= args.hour_end <= 23:
            raise ValueError("Hours must be between 0 and 23.")
        if args.hour_end < args.hour_start:
            raise ValueError("--hour-end must be greater than or equal to --hour-start.")
        if args.max_workers < 1:
            raise ValueError("--max-workers must be at least 1.")

        config = load_config(args.config)
        raw_data_dir = PROJECT_ROOT / config["paths"]["raw_data_dir"]
        archive_dir = Path(args.archive_dir) if args.archive_dir else raw_data_dir / "opensky"
        output_dir = Path(args.output_dir) if args.output_dir else raw_data_dir / "opensky"
        single_output_file = Path(args.single_output_file) if args.single_output_file else None
        ensure_dir(archive_dir)
        ensure_dir(output_dir)
        if single_output_file is not None:
            ensure_dir(single_output_file.parent)
            if single_output_file.exists() and not args.overwrite:
                logger.info("Combined parquet already exists, skipping: %s", single_output_file)
                return
            if single_output_file.exists() and args.overwrite:
                single_output_file.unlink()

        tasks = [
            (sample_date, hour)
            for sample_date in args.sample_date
            for hour in range(args.hour_start, args.hour_end + 1)
        ]
        total_steps = len(tasks)

        for sample_date in args.sample_date:
            logger.info("Queued sample date %s", sample_date)

        downloaded_archives: dict[tuple[str, int], Path] = {}
        download_progress = tqdm(total=total_steps, desc="OpenSky downloads", unit="hour", file=sys.stdout)
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                future_map = {
                    executor.submit(download_archive, sample_date, hour, archive_dir, args.url_template, args.overwrite): (sample_date, hour)
                    for sample_date, hour in tasks
                }

                for future in as_completed(future_map):
                    sample_date, hour = future_map[future]
                    download_progress.set_postfix_str(f"{sample_date} {hour:02d}:00")
                    archive_path = future.result()
                    if archive_path is not None:
                        downloaded_archives[(sample_date, hour)] = archive_path
                    download_progress.update(1)
        finally:
            download_progress.close()

        if not downloaded_archives:
            raise RuntimeError("No OpenSky sample archives were downloaded.")

        if single_output_file is not None:
            writer: pq.ParquetWriter | None = None
            writer_schema: pa.Schema | None = None
            written_batches = 0
            written_rows = 0
            concat_progress = tqdm(total=total_steps, desc="OpenSky concat", unit="hour", file=sys.stdout)
            try:
                for sample_date, hour in tasks:
                    concat_progress.set_postfix_str(f"{sample_date} {hour:02d}:00")
                    archive_path = downloaded_archives.get((sample_date, hour))
                    if archive_path is None:
                        concat_progress.update(1)
                        continue

                    frame = load_archive_frame(archive_path, sample_date, hour)
                    table = pa.Table.from_pandas(frame, preserve_index=False)
                    if writer_schema is None:
                        writer_schema = table.schema
                    else:
                        table = table.cast(writer_schema)

                    if writer is None:
                        writer = pq.ParquetWriter(single_output_file, writer_schema, compression="snappy")
                    writer.write_table(table)
                    written_batches += 1
                    written_rows += table.num_rows
                    concat_progress.update(1)
            finally:
                concat_progress.close()
                if writer is not None:
                    writer.close()

            if written_batches == 0:
                raise RuntimeError("No sample archives were converted into the combined parquet output.")
            logger.info(
                "Saved combined parquet to %s (%s rows across %s batches)",
                single_output_file,
                written_rows,
                written_batches,
            )
        else:
            convert_progress = tqdm(total=total_steps, desc="OpenSky convert", unit="hour", file=sys.stdout)
            try:
                for sample_date, hour in tasks:
                    convert_progress.set_postfix_str(f"{sample_date} {hour:02d}:00")
                    archive_path = downloaded_archives.get((sample_date, hour))
                    if archive_path is not None:
                        convert_archive(archive_path, output_dir, sample_date, hour, args.overwrite)
                    convert_progress.update(1)
            finally:
                convert_progress.close()
    except requests.RequestException as exc:
        logger.error("OpenSky sample download failed: %s", exc)
        raise SystemExit(1) from exc
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
