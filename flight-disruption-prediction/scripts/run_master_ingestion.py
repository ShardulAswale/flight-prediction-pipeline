"""Run the monthly master ingestion control.

Examples
--------
Dry-run plan only:
    python scripts/run_master_ingestion.py --start-date 2022-01-01 --end-date 2022-06-30

Execute downloads/rebuilds:
    python scripts/run_master_ingestion.py --start-date 2022-01-01 --end-date 2022-06-30 --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion_control import MasterIngestionConfig, run_master_ingestion
from src.pipeline_reporting import configure_pretty_logging
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monthly master ingestion controller.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--start-date", help="Inclusive start date, e.g. 2022-01-01.")
    parser.add_argument("--end-date", help="Inclusive end date, e.g. 2022-06-30.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Sources to include: opensky bts eurocontrol. Defaults to config ingestion.master_control.sources.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run downloads/rebuilds. Default is dry-run plan only.")
    parser.add_argument("--force", action="store_true", help="Overwrite/rebuild existing outputs.")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--hour-start", type=int, default=None)
    parser.add_argument("--hour-end", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_pretty_logging(PROJECT_ROOT / "logs", "master_ingestion")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    logger = logging.getLogger("master_ingestion")

    config = load_config(args.config)
    ingest_cfg = config.get("ingestion", {})
    master_cfg = ingest_cfg.get("master_control", {}) if isinstance(ingest_cfg.get("master_control", {}), dict) else {}
    paths_cfg = config.get("paths", {})

    start_date = args.start_date or master_cfg.get("start_date") or f"{ingest_cfg.get('year', 2022)}-{int(ingest_cfg.get('start_month', 1)):02d}-01"
    if args.end_date:
        end_date = args.end_date
    elif master_cfg.get("end_date"):
        end_date = master_cfg["end_date"]
    else:
        year = int(ingest_cfg.get("year", 2022))
        end_month = int(ingest_cfg.get("end_month", 12))
        end_date = (pd.Timestamp(year=year, month=end_month, day=1) + pd.offsets.MonthEnd(0)).date().isoformat()

    control = MasterIngestionConfig(
        project_root=PROJECT_ROOT,
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        sources=tuple(args.sources or master_cfg.get("sources", ["opensky", "bts", "eurocontrol"])),
        hour_start=int(args.hour_start if args.hour_start is not None else master_cfg.get("hour_start", 0)),
        hour_end=int(args.hour_end if args.hour_end is not None else master_cfg.get("hour_end", 23)),
        max_workers=int(args.max_workers if args.max_workers is not None else master_cfg.get("max_workers", 4)),
        force=bool(args.force),
        execute_downloads=bool(args.execute),
        combine_outputs=bool(master_cfg.get("combine_outputs", True)),
        raw_dir=PROJECT_ROOT / paths_cfg.get("raw_data_dir", "data/raw"),
        processed_dir=PROJECT_ROOT / paths_cfg.get("processed_data_dir", "data/processed"),
        outputs_dir=PROJECT_ROOT / master_cfg.get("outputs_dir", "outputs/ingestion_control"),
        opensky_url_template=master_cfg.get(
            "opensky_url_template",
            "https://s3.opensky-network.org/data-samples/states/{date}/{hour:02d}/states_{date}-{hour:02d}.avro.tar",
        ),
    )

    logger.info(
        "Monthly ingestion control: %s -> %s | sources=%s | execute=%s",
        control.start_date.date(),
        control.end_date.date(),
        control.sources,
        control.execute_downloads,
    )
    result = run_master_ingestion(control)
    logger.info("Ingestion control complete. Rows in status/plan: %d", len(result))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
