"""Download OpenSky bulk hourly state-vector files with manifest tracking."""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests


logger = logging.getLogger(__name__)


def parse_hours(hours_input: Optional[str]) -> List[int]:
    """Parse hour input into a sorted list of integers [0..23]."""
    if not hours_input or hours_input.lower() == "all":
        return list(range(24))

    hours: List[int] = []
    for part in hours_input.split(","):
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str), int(end_str)
            if start > end:
                raise ValueError(f"Invalid range {part}: start must be <= end")
            hours.extend(range(start, end + 1))
        else:
            hours.append(int(part))

    unique_hours = sorted(set(hours))
    for hour in unique_hours:
        if hour < 0 or hour > 23:
            raise ValueError(f"Hour out of range: {hour}")
    return unique_hours


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_existing_manifest(manifest_path: Path) -> Dict[str, Dict[str, str]]:
    """Load existing manifest entries keyed by local_path for idempotent skips."""
    if not manifest_path.exists():
        return {}

    records: Dict[str, Dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            records[row["local_path"]] = row
    return records


def render_url(base_url: str, date: str, hour: int, file_format: str) -> str:
    """Render the remote URL from base URL.

    Supports either:
    - template base_url containing placeholders: {date}, {hour}, {format}
    - plain base_url as root path, using <base>/<date>/<hour>.<format>.gz
    """
    if any(token in base_url for token in ("{date}", "{hour}", "{format}")):
        return base_url.format(date=date, hour=f"{hour:02d}", format=file_format)
    return f"{base_url.rstrip('/')}/{date}/{hour:02d}.{file_format}.gz"


def request_stream_with_retry(
    url: str,
    max_attempts: int = 5,
    timeout_seconds: int = 60,
    session: Optional[requests.Session] = None,
) -> Optional[requests.Response]:
    """Stream a URL with retry + exponential backoff on transient failures."""
    session = session or requests.Session()
    for attempt in range(1, max_attempts + 1):
        try:
            response = session.get(url, stream=True, timeout=timeout_seconds)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else (2 ** attempt)
                logger.warning("429 rate limit for %s; retrying in %.1fs", url, wait_seconds)
                response.close()
                time.sleep(wait_seconds)
                continue
            if 500 <= response.status_code <= 599:
                wait_seconds = 2 ** attempt
                logger.warning("Server error %s for %s; retrying in %.1fs", response.status_code, url, wait_seconds)
                response.close()
                time.sleep(wait_seconds)
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            wait_seconds = 2 ** attempt
            logger.warning("Attempt %d/%d failed for %s (%s); retrying in %.1fs", attempt, max_attempts, url, exc, wait_seconds)
            time.sleep(wait_seconds)

    logger.error("Exceeded retry attempts for %s", url)
    return None


def write_manifest(manifest_path: Path, rows: List[Dict[str, str]]) -> None:
    """Write a manifest CSV with deterministic column order."""
    fieldnames = [
        "date",
        "hour",
        "url",
        "local_path",
        "bytes",
        "sha256",
        "downloaded_at_utc",
        "status",
    ]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def download_opensky_bulk(
    base_url: str,
    date: str,
    hours: Iterable[int],
    file_format: str,
    out_dir: str,
    manifest_dir: str = "outputs/manifests",
) -> Path:
    """Download hourly OpenSky bulk files and record a manifest CSV."""
    if file_format not in {"avro", "csv"}:
        raise ValueError("file_format must be 'avro' or 'csv'")

    output_root = Path(out_dir) / date
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(manifest_dir) / f"bulk_files_manifest_{date}.csv"
    existing_rows = load_existing_manifest(manifest_path)
    updated_rows: List[Dict[str, str]] = []

    session = requests.Session()

    for hour in sorted(set(hours)):
        url = render_url(base_url=base_url, date=date, hour=hour, file_format=file_format)
        filename = Path(url).name
        if not filename:
            filename = f"{hour:02d}.{file_format}.gz"
        local_path = output_root / filename

        downloaded_at_utc = datetime.now(timezone.utc).isoformat()

        if local_path.exists() and local_path.stat().st_size > 0:
            current_sha = sha256_file(local_path)
            prior = existing_rows.get(str(local_path))
            if prior and prior.get("sha256") == current_sha:
                updated_rows.append(
                    {
                        "date": date,
                        "hour": f"{hour:02d}",
                        "url": url,
                        "local_path": str(local_path),
                        "bytes": str(local_path.stat().st_size),
                        "sha256": current_sha,
                        "downloaded_at_utc": downloaded_at_utc,
                        "status": "skipped_existing",
                    }
                )
                logger.info("Skipping existing verified file: %s", local_path)
                continue

        response = request_stream_with_retry(url=url, session=session)
        if response is None:
            updated_rows.append(
                {
                    "date": date,
                    "hour": f"{hour:02d}",
                    "url": url,
                    "local_path": str(local_path),
                    "bytes": "0",
                    "sha256": "",
                    "downloaded_at_utc": downloaded_at_utc,
                    "status": "failed",
                }
            )
            continue

        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        with tmp_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_obj.write(chunk)
        response.close()

        if tmp_path.stat().st_size == 0:
            tmp_path.unlink(missing_ok=True)
            updated_rows.append(
                {
                    "date": date,
                    "hour": f"{hour:02d}",
                    "url": url,
                    "local_path": str(local_path),
                    "bytes": "0",
                    "sha256": "",
                    "downloaded_at_utc": downloaded_at_utc,
                    "status": "empty",
                }
            )
            logger.error("Downloaded empty file for %s", url)
            continue

        tmp_path.replace(local_path)
        digest = sha256_file(local_path)
        file_size = local_path.stat().st_size

        updated_rows.append(
            {
                "date": date,
                "hour": f"{hour:02d}",
                "url": url,
                "local_path": str(local_path),
                "bytes": str(file_size),
                "sha256": digest,
                "downloaded_at_utc": downloaded_at_utc,
                "status": "downloaded",
            }
        )
        logger.info("Downloaded %s (%s bytes)", local_path, file_size)

    write_manifest(manifest_path, updated_rows)
    logger.info("Wrote manifest: %s", manifest_path)
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for bulk download utility."""
    parser = argparse.ArgumentParser(description="Download OpenSky bulk state-vector hourly files.")
    parser.add_argument("--base-url", required=True, help="Base URL or URL template for files")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument(
        "--hours",
        default="all",
        help="Hours as comma/range list (e.g. '0-23' or '0,1,2,10-12'). Default: all",
    )
    parser.add_argument("--format", dest="file_format", choices=["avro", "csv"], default="avro")
    parser.add_argument(
        "--out-dir",
        default="data/raw/opensky_state_vectors",
        help="Root output directory for downloaded files",
    )
    parser.add_argument(
        "--manifest-dir",
        default="outputs/manifests",
        help="Directory where download manifest CSV is written",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    hours = parse_hours(args.hours)
    download_opensky_bulk(
        base_url=args.base_url,
        date=args.date,
        hours=hours,
        file_format=args.file_format,
        out_dir=args.out_dir,
        manifest_dir=args.manifest_dir,
    )


if __name__ == "__main__":
    main()
