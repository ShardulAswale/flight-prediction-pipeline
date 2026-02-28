"""Shared utilities for the OpenSky flight-level ingestion pipeline."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Generator, Iterable, Optional, TypeVar


T = TypeVar("T")


def setup_logging(level: str = "INFO") -> None:
    """Configure process-wide logging once."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a date object."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start_date: date, end_date: date) -> Generator[date, None, None]:
    """Yield all dates in [start_date, end_date]."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def utc_midnight(day: date) -> int:
    """Return UNIX timestamp for 00:00:00 UTC of a date."""
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def chunk_two_hour_windows(day: date) -> Iterable[tuple[int, int]]:
    """Yield 2-hour begin/end UNIX timestamps for one UTC day."""
    day_start = utc_midnight(day)
    for offset_hours in range(0, 24, 2):
        begin = day_start + offset_hours * 3600
        end = begin + 2 * 3600
        yield begin, end


def run_with_retry(
    func: Callable[[], T],
    max_attempts: int = 5,
    base_sleep_seconds: float = 2.0,
    logger: Optional[logging.Logger] = None,
    operation_name: str = "operation",
) -> T:
    """Run a callable with bounded retry and exponential backoff."""
    logger = logger or logging.getLogger(__name__)
    last_error: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_attempts:
                break
            sleep_seconds = base_sleep_seconds * (2 ** (attempt - 1))
            logger.warning(
                "%s failed on attempt %d/%d: %s. Sleeping %.1fs before retry.",
                operation_name,
                attempt,
                max_attempts,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise last_error


@dataclass
class OpenSkyConfig:
    """OpenSky authentication config from environment variables."""

    username: Optional[str]
    password: Optional[str]

    @classmethod
    def from_env(cls) -> "OpenSkyConfig":
        return cls(
            username=os.getenv("OPENSKY_USERNAME") or None,
            password=os.getenv("OPENSKY_PASSWORD") or None,
        )
