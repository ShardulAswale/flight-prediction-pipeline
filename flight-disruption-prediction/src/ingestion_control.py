"""Master monthly ingestion control.

This module coordinates sources that naturally arrive in different shapes:

- OpenSky ADS-B: weekly sample dates, hourly archives
- BTS: monthly ZIP/CSV files
- Eurocontrol: monthly parquet files

The control layer builds a month-by-month plan, optionally executes downloads,
trims schedule datasets to the requested date window, and saves reporting
artifacts under ``outputs/ingestion_control``.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_ingestion import BTSCombiner, BTSDownloader, EuroCombiner, EurocontrolDownloader
from src.pipeline_reporting import (
    PerformanceLog,
    output_dir,
    save_bar_chart,
    save_dataframe,
    save_pipeline_flow_diagram,
    save_table_image,
)
from src.utils import ensure_dir

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MonthWindow:
    year: int
    month: int
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"{self.year}-{self.month:02d}"


@dataclass
class MasterIngestionConfig:
    project_root: Path
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    sources: tuple[str, ...] = ("opensky", "bts", "eurocontrol")
    hour_start: int = 0
    hour_end: int = 23
    max_workers: int = 4
    force: bool = False
    execute_downloads: bool = False
    combine_outputs: bool = True
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    outputs_dir: Path = Path("outputs/ingestion_control")
    opensky_url_template: str = (
        "https://s3.opensky-network.org/data-samples/states/{date}/{hour:02d}/"
        "states_{date}-{hour:02d}.avro.tar"
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        for attr in ["raw_dir", "processed_dir", "outputs_dir"]:
            path = getattr(self, attr)
            if not Path(path).is_absolute():
                path = self.project_root / path
            object.__setattr__(self, attr, Path(path))
        object.__setattr__(self, "sources", tuple(str(s).lower() for s in self.sources))
        object.__setattr__(self, "start_date", pd.Timestamp(self.start_date).tz_localize(None))
        object.__setattr__(self, "end_date", pd.Timestamp(self.end_date).tz_localize(None))


def month_windows(start_date: str | date | datetime | pd.Timestamp, end_date: str | date | datetime | pd.Timestamp) -> list[MonthWindow]:
    start = pd.Timestamp(start_date).tz_localize(None).normalize()
    end = pd.Timestamp(end_date).tz_localize(None).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    current = pd.Timestamp(year=start.year, month=start.month, day=1)
    windows: list[MonthWindow] = []
    while current <= end:
        month_start = max(current, start)
        next_month = current + pd.offsets.MonthBegin(1)
        month_end = min(next_month - pd.Timedelta(seconds=1), end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        windows.append(MonthWindow(current.year, current.month, month_start, month_end))
        current = next_month
    return windows


def mondays_in_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> list[str]:
    current = pd.Timestamp(start_dt).normalize()
    while current.weekday() != 0:
        current += pd.Timedelta(days=1)

    values: list[str] = []
    while current <= end_dt.normalize():
        values.append(current.strftime("%Y-%m-%d"))
        current += pd.Timedelta(days=7)
    return values


def build_monthly_plan(config: MasterIngestionConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for win in month_windows(config.start_date, config.end_date):
        if "opensky" in config.sources:
            sample_dates = mondays_in_range(win.start, win.end)
            rows.append(
                {
                    "source": "opensky",
                    "month": win.label,
                    "period_start": win.start.date().isoformat(),
                    "period_end": win.end.date().isoformat(),
                    "storage_grain": "weekly sample dates x hourly archives",
                    "units": len(sample_dates) * (config.hour_end - config.hour_start + 1),
                    "sample_dates": ", ".join(sample_dates),
                    "monthly_output": str(config.processed_dir / "monthly" / "opensky" / f"adsb_{win.year}_{win.month:02d}.parquet"),
                }
            )
        if "bts" in config.sources:
            bts_csv_name = (
                "On_Time_Reporting_Carrier_On_Time_Performance_"
                f"(1987_present)_{win.year}_{win.month}.csv"
            )
            rows.append(
                {
                    "source": "bts",
                    "month": win.label,
                    "period_start": win.start.date().isoformat(),
                    "period_end": win.end.date().isoformat(),
                    "storage_grain": "monthly zip/csv",
                    "units": 1,
                    "sample_dates": "",
                    "monthly_output": str(config.raw_dir / "bts" / bts_csv_name),
                }
            )
        if "eurocontrol" in config.sources:
            rows.append(
                {
                    "source": "eurocontrol",
                    "month": win.label,
                    "period_start": win.start.date().isoformat(),
                    "period_end": win.end.date().isoformat(),
                    "storage_grain": "monthly parquet",
                    "units": 1,
                    "sample_dates": "",
                    "monthly_output": str(config.raw_dir / "eurocontrol" / f"euro_flight_list_{win.year}{win.month:02d}.parquet"),
                }
            )
    return pd.DataFrame(rows)


def _combine_parquet_files(input_files: Sequence[Path], output_file: Path, *, force: bool = False) -> Path | None:
    input_files = [Path(p) for p in input_files if Path(p).exists()]
    if not input_files:
        logger.warning("No parquet files found for %s", output_file)
        return None
    if output_file.exists() and not force:
        logger.info("Combined output already exists: %s", output_file)
        return output_file

    ensure_dir(output_file.parent)
    if output_file.exists():
        output_file.unlink()

    writer: pq.ParquetWriter | None = None
    schema: pa.Schema | None = None
    rows = 0
    try:
        for path in input_files:
            table = pq.read_table(path)
            if schema is None:
                schema = table.schema
            else:
                table = table.cast(schema)
            if writer is None:
                writer = pq.ParquetWriter(output_file, schema, compression="snappy")
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    logger.info("Combined %d parquet files into %s (%d rows)", len(input_files), output_file, rows)
    return output_file


def trim_parquet_by_time(
    input_file: Path,
    output_file: Path,
    *,
    time_candidates: Sequence[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    force: bool = False,
) -> Path | None:
    if not input_file.exists():
        logger.warning("Cannot trim missing parquet: %s", input_file)
        return None
    if output_file.exists() and not force:
        return output_file

    df = pd.read_parquet(input_file)
    time_col = next((col for col in time_candidates if col in df.columns), None)
    if time_col is None:
        logger.warning("No usable time column found in %s; copying without trim.", input_file)
        ensure_dir(output_file.parent)
        df.to_parquet(output_file, index=False)
        return output_file

    start = pd.Timestamp(start_date).tz_localize(None)
    end = pd.Timestamp(end_date).tz_localize(None) + pd.Timedelta(days=1)
    series = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_convert(None)
    trimmed = df[(series >= start) & (series < end)].copy()
    ensure_dir(output_file.parent)
    trimmed.to_parquet(output_file, index=False)
    logger.info(
        "Trimmed %s by %s from %d to %d rows -> %s",
        input_file,
        time_col,
        len(df),
        len(trimmed),
        output_file,
    )
    return output_file


def _run_opensky_month(config: MasterIngestionConfig, win: MonthWindow, perf: PerformanceLog) -> Path | None:
    sample_dates = mondays_in_range(win.start, win.end)
    if not sample_dates:
        logger.warning("No OpenSky sample Mondays for %s", win.label)
        return None

    output_file = config.processed_dir / "monthly" / "opensky" / f"adsb_{win.year}_{win.month:02d}.parquet"
    if output_file.exists() and not config.force:
        logger.info("OpenSky monthly output exists: %s", output_file)
        return output_file

    args = [
        sys.executable,
        str(config.project_root / "scripts" / "fetch_opensky_samples.py"),
        "--hour-start",
        str(config.hour_start),
        "--hour-end",
        str(config.hour_end),
        "--max-workers",
        str(config.max_workers),
        "--archive-dir",
        str(config.raw_dir / "opensky"),
        "--url-template",
        config.opensky_url_template,
        "--single-output-file",
        str(output_file),
    ]
    if config.force:
        args.append("--overwrite")
    for sample_date in sample_dates:
        args.extend(["--sample-date", sample_date])

    with perf.step("opensky_month", month=win.label, sample_dates=len(sample_dates), output=str(output_file)):
        logger.info("Running OpenSky month command: %s", " ".join(args))
        subprocess.run(args, cwd=config.project_root, check=True)
    return output_file


def run_master_ingestion(config: MasterIngestionConfig) -> pd.DataFrame:
    """Run or plan monthly ingestion and save reporting artifacts.

    If ``execute_downloads`` is ``False``, this produces a dry-run plan only.
    """
    ensure_dir(config.raw_dir)
    ensure_dir(config.processed_dir)
    ensure_dir(config.outputs_dir)

    perf = PerformanceLog("master_ingestion", config.outputs_dir / "logs")
    plan = build_monthly_plan(config)
    save_dataframe(plan, "monthly_ingestion_plan", config.outputs_dir)
    save_table_image(plan, "monthly_ingestion_plan", config.outputs_dir, title="Monthly Ingestion Plan", max_rows=40)
    if not plan.empty:
        by_source = plan.groupby("source", as_index=False)["units"].sum()
        save_bar_chart(
            by_source,
            "source",
            "units",
            "monthly_units_by_source",
            config.outputs_dir,
            title="Planned Download/Processing Units by Source",
            ylabel="Units",
        )
    save_pipeline_flow_diagram(
        [
            "Master Control",
            "Monthly Source Plans",
            "Download / Reuse",
            "Source-Specific Trim",
            "Combined Parquets",
            "Feature Pipeline",
        ],
        "ingestion_data_flow",
        config.outputs_dir,
        title="Monthly Ingestion Control Flow",
    )

    if not config.execute_downloads:
        logger.info("Dry run only. Set execute_downloads=True or pass --execute to run downloads.")
        return plan.assign(status="PLANNED")

    status_rows: list[dict] = []
    windows = month_windows(config.start_date, config.end_date)

    opensky_monthlies: list[Path] = []
    if "opensky" in config.sources:
        for win in windows:
            try:
                path = _run_opensky_month(config, win, perf)
                if path:
                    opensky_monthlies.append(path)
                status_rows.append({"source": "opensky", "month": win.label, "status": "SUCCESS", "output": str(path)})
            except Exception as exc:
                logger.exception("OpenSky failed for %s", win.label)
                status_rows.append({"source": "opensky", "month": win.label, "status": "FAILED", "error": repr(exc)})
                raise

        if config.combine_outputs:
            with perf.step("combine_opensky", files=len(opensky_monthlies)):
                _combine_parquet_files(
                    opensky_monthlies,
                    config.processed_dir / "adsb_combined.parquet",
                    force=config.force,
                )

    if "bts" in config.sources:
        downloader = BTSDownloader()
        with perf.step("download_bts", months=len(windows)):
            for win in windows:
                path = downloader.download_month(win.year, win.month, config.raw_dir)
                status_rows.append({"source": "bts", "month": win.label, "status": "SUCCESS" if path else "MISSING", "output": str(path) if path else ""})

        with perf.step("combine_bts"):
            combined = config.processed_dir / "bts_combined_all.parquet"
            BTSCombiner().combine_csvs(config.raw_dir / "bts", combined)
            trim_parquet_by_time(
                combined,
                config.processed_dir / "bts_combined.parquet",
                time_candidates=("scheduled_dep", "scheduled_dep_utc", "service_day_utc"),
                start_date=config.start_date,
                end_date=config.end_date,
                force=True,
            )

    if "eurocontrol" in config.sources:
        downloader = EurocontrolDownloader()
        with perf.step("download_eurocontrol", months=len(windows)):
            for win in windows:
                path = downloader.download_month(win.year, win.month, config.raw_dir)
                status_rows.append({"source": "eurocontrol", "month": win.label, "status": "SUCCESS" if path else "MISSING", "output": str(path) if path else ""})

        with perf.step("combine_eurocontrol"):
            combined = config.processed_dir / "eurocontrol_combined_all.parquet"
            EuroCombiner().combine_parquets(config.raw_dir, combined)
            trim_parquet_by_time(
                combined,
                config.processed_dir / "eurocontrol_combined.parquet",
                time_candidates=("scheduled_dep", "scheduled_dep_utc", "FILED_OFF_BLOCK_TIME", "actual_offblock_time"),
                start_date=config.start_date,
                end_date=config.end_date,
                force=True,
            )

    status = pd.DataFrame(status_rows)
    save_dataframe(status, "monthly_ingestion_status", config.outputs_dir)
    save_table_image(status, "monthly_ingestion_status", config.outputs_dir, title="Monthly Ingestion Status", max_rows=60)
    return status
