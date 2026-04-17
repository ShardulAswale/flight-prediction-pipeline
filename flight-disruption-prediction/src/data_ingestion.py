import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import logging
from pathlib import Path
from typing import Optional, List
import argparse
import sys
import time
import random
import zipfile
from datetime import datetime, date, timedelta

# Add parent dir to path to allow imports if running as script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import ensure_dir, load_config

logger = logging.getLogger(__name__)

class ManifestManager:
    """Manages the download manifest checkpoint table."""
    def __init__(self, manifest_path: str = "data/raw/manifest.csv"):
        self.manifest_path = Path(manifest_path)
        ensure_dir(self.manifest_path.parent)
        if not self.manifest_path.exists():
            pd.DataFrame(columns=[
                'source', 'year', 'month', 'day', 'status', 'row_count', 'downloaded_at', 'file_path'
            ]).to_csv(self.manifest_path, index=False)
            
    def log_download(self, source: str, year: int, month: int, status: str, 
                     row_count: int, file_path: Optional[Path], day: int = 0):
        new_row = {
            'source': source,
            'year': year,
            'month': month,
            'day': day,
            'status': status,
            'row_count': row_count,
            'downloaded_at': datetime.utcnow().isoformat(),
            'file_path': str(file_path) if file_path else ""
        }
        pd.DataFrame([new_row]).to_csv(self.manifest_path, mode='a', header=False, index=False)
        logger.info(f"Manifest updated: {source} {year}-{month:02d}-{day:02d} -> {status}")
    
    def is_downloaded(self, source: str, year: int, month: int, day: int = 0) -> bool:
        """Check if a specific source/date has already been successfully downloaded."""
        try:
            df = pd.read_csv(self.manifest_path)
            mask = (
                (df['source'] == source) & 
                (df['year'] == year) & 
                (df['month'] == month) & 
                (df['day'] == day) &
                (df['status'] == 'SUCCESS')
            )
            return mask.any()
        except Exception:
            return False

def retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0):
    """Decorator for exponential backoff with jitter."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Max retries ({max_retries}) reached. Last error: {e}")
                        raise
                    delay = min(max_delay, (base_delay * 2 ** (retries - 1)))
                    jitter = random.uniform(0, 0.1 * delay)
                    sleep_time = delay + jitter
                    logger.warning(f"Attempt {retries} failed: {e}. Retrying in {sleep_time:.2f}s...")
                    time.sleep(sleep_time)
        return wrapper
    return decorator


class EurocontrolDownloader:
    """Downloader for Eurocontrol OPDI (Open Performance Data Initiative) Parquet files."""
    
    BASE_URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/flight_list_{YYYYMM}.parquet"

    def __init__(self, manifest_mgr: ManifestManager = None):
        self.manifest = manifest_mgr or ManifestManager()

    @retry_with_backoff(max_retries=3)
    def _download_file(self, url: str, file_path: Path):
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            raise ValueError(f"Received HTML instead of Parquet from {url}")

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
    def download_month(self, year: int, month: int, base_output_dir: str | Path) -> Optional[Path]:
        """Download Eurocontrol flight list Parquet for a specific month with partitioning."""
        yyyymm = f"{year}{month:02d}"
        url = self.BASE_URL.format(YYYYMM=yyyymm)
        
        output_dir = Path(base_output_dir) / 'eurocontrol' / f'year={year}' / f'month={month:02d}'
        ensure_dir(output_dir)
        
        file_path = output_dir / f"euro_flight_list_{yyyymm}.parquet"
        
        if file_path.exists():
            logger.info(f"File {file_path.name} already exists. Skipping download.")
            return file_path

        logger.info(f"Downloading Eurocontrol data from {url}...")
        try:
            self._download_file(url, file_path)
            
            try:
                df = pd.read_parquet(file_path, columns=['ADEP'] if 'ADEP' in pd.read_parquet(file_path, columns=[]).columns else [])
                row_count = len(pd.read_parquet(file_path))
            except Exception:
                row_count = 0
                logger.warning(f"Could not read row count from downloaded file {file_path}")

            self.manifest.log_download('eurocontrol', year, month, 'SUCCESS', row_count, file_path)
            logger.info(f"Successfully downloaded Eurocontrol data to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to download Eurocontrol data: {e}")
            self.manifest.log_download('eurocontrol', year, month, 'FAILED', 0, None)
            return None

class BTSDownloader:
    """Downloader for BTS (US Bureau of Transportation Statistics) On-Time Performance data."""

    BASE_URL = (
        "https://transtats.bts.gov/PREZIP/"
        "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
    )
    
    def __init__(self, manifest_mgr: ManifestManager = None):
        self.manifest = manifest_mgr or ManifestManager()

    @retry_with_backoff(max_retries=3)
    def _download_file(self, url: str, file_path: Path):
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def download_month(self, year: int, month: int, base_output_dir: str | Path) -> Optional[Path]:
        output_dir = Path(base_output_dir) / 'bts' / f'year={year}' / f'month={month:02d}'
        ensure_dir(output_dir)
        
        file_path = output_dir / f"bts_on_time_{year}_{month:02d}.zip"
        csv_name = f"On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{year}_{month}.csv"
        csv_path = output_dir / csv_name
        
        if csv_path.exists() and csv_path.stat().st_size > 1000000:
            logger.info(f"CSV {csv_path.name} already exists. Skipping download.")
            return csv_path

        if file_path.exists() and file_path.stat().st_size > 1000000:
            logger.info(f"ZIP {file_path.name} already exists. Reusing archive.")
        else:
            url = self.BASE_URL.format(year=year, month=month)
            logger.info(f"Downloading BTS data from {url}...")
            try:
                self._download_file(url, file_path)
            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else "unknown"
                logger.error(f"Failed to download BTS data for {year}-{month:02d}: HTTP {status_code}")
                self.manifest.log_download('bts', year, month, 'FAILED', 0, None)
                return None
            except Exception as e:
                logger.error(f"Failed to download BTS data for {year}-{month:02d}: {e}")
                self.manifest.log_download('bts', year, month, 'FAILED', 0, None)
                return None

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                csv_members = [name for name in zip_ref.namelist() if name.lower().endswith('.csv')]
                if not csv_members:
                    raise ValueError(f"No CSV found inside {file_path.name}")

                member_name = csv_members[0]
                with zip_ref.open(member_name) as src, open(csv_path, 'wb') as dst:
                    dst.write(src.read())

            try:
                row_count = len(pd.read_csv(csv_path, usecols=['FL_DATE']))
            except Exception:
                row_count = 0
                logger.warning(f"Could not determine row count for {csv_path.name}")

            self.manifest.log_download('bts', year, month, 'SUCCESS', row_count, csv_path)
            logger.info(f"Successfully prepared BTS CSV at {csv_path}")
            return csv_path
        except Exception as e:
            logger.error(f"Failed to extract BTS archive {file_path.name}: {e}")
            self.manifest.log_download('bts', year, month, 'FAILED', 0, None)
            return None

class BTSCombiner:
    """Combines manually downloaded BTS CSV files into a complete parquet dataset.
    T5: Wires ScheduleNormalizer into the output.
    """
    
    REQUIRED_COLS = [
        'FL_DATE', 'FlightDate',
        'OP_UNIQUE_CARRIER', 'IATA_CODE_Reporting_Airline', 'Reporting_Airline',
        'OP_CARRIER_FL_NUM', 'Flight_Number_Reporting_Airline',
        'ORIGIN', 'Origin', 'DEST', 'Dest',
        'OriginState', 'DestState',
        'CANCELLED', 'Cancelled',
        'DepDelay', 'DepDelayMinutes',
        'CRSElapsedTime', 'ActualElapsedTime',
        'CRS_DEP_TIME', 'CRSDepTime', 'DEP_TIME', 'DepTime',
        'CRS_ARR_TIME', 'CRSArrTime', 'ARR_TIME', 'ArrTime'
    ]

    def combine_csvs(self, input_dir: str | Path, output_file: str | Path) -> Optional[Path]:
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            return None
            
        csv_files = sorted(input_dir.rglob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {input_dir}")
            return None
            
        logger.info(f"Combining {len(csv_files)} BTS CSV files from {input_dir}")
        
        # T5: Use ScheduleNormalizer instead of raw preprocess_bts
        from src.normalization import ScheduleNormalizer
        
        processed_dfs = []
        for file in csv_files:
            try:
                logger.info(f"  Processing {file.name}...")
                df = pd.read_csv(file, usecols=lambda x: x in self.REQUIRED_COLS, low_memory=False)
                # T5: Normalize directly using ScheduleNormalizer
                df_proc = ScheduleNormalizer.normalize_bts(df)
                processed_dfs.append(df_proc)
                logger.info(f"    Loaded {len(df_proc):,} records.")
                del df
            except Exception as e:
                logger.error(f"  Failed to process {file.name}: {e}")
                
        if not processed_dfs:
            return None
            
        logger.info("Concatenating, deduplicating, and saving combined dataset...")
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        
        # Deterministic deduplication
        if 'flight_key' in combined_df.columns:
            start_len = len(combined_df)
            combined_df = combined_df.sort_values(by=['flight_key', 'scheduled_dep']).drop_duplicates(subset=['flight_key'], keep='first')
            logger.info(f"Deduplication removed {start_len - len(combined_df):,} duplicate rows.")
            
        ensure_dir(output_file.parent)
        combined_df.to_parquet(output_file, index=False)
        logger.info(f"Successfully saved combined BTS dataset to {output_file}")
        return output_file

class EuroCombiner:
    """Combines downloaded Eurocontrol monthly Parquet files into a single dataset.
    T6: Wires ScheduleNormalizer into the output.
    """
    
    def combine_parquets(self, input_dir: str | Path, output_file: str | Path) -> Optional[Path]:
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        input_dir = Path(input_dir)
        output_file = Path(output_file)
        
        parquet_files = sorted(input_dir.rglob("euro_flight_list_*.parquet"))
        if not parquet_files:
            logger.warning(f"No Eurocontrol Parquet files found in {input_dir}")
            return None
            
        logger.info(f"Combining {len(parquet_files)} Eurocontrol files sequentially...")
        
        # T6: Use ScheduleNormalizer instead of preprocess_eurocontrol
        from src.normalization import ScheduleNormalizer
        
        ensure_dir(output_file.parent)
        
        writer = None
        total_rows = 0
        seen_keys = set()
        
        for file in parquet_files:
            try:
                logger.info(f"  Loading & appending {file.name}...")
                df = pd.read_parquet(file)
                # T6: Normalize using ScheduleNormalizer
                df_proc = ScheduleNormalizer.normalize_eurocontrol(df)
                
                # Deterministic deduplication
                if 'flight_key' in df_proc.columns:
                    df_proc = df_proc[~df_proc['flight_key'].isin(seen_keys)]
                    df_proc = df_proc.drop_duplicates(subset=['flight_key'], keep='first')
                    seen_keys.update(df_proc['flight_key'].tolist())
                
                if len(df_proc) == 0:
                    continue
                    
                table = pa.Table.from_pandas(df_proc, preserve_index=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_file, table.schema)
                    
                writer.write_table(table)
                total_rows += len(df_proc)
                
                del df, df_proc, table
            except Exception as e:
                logger.error(f"  Failed to process {file.name}: {e}")
                
        if writer:
            writer.close()
            logger.info(f"Successfully saved combined Eurocontrol dataset ({total_rows:,} records) to {output_file}")
            return output_file
        else:
            logger.error("Failed to write any Eurocontrol data.")
            return None

class OpenSkyBackfiller:
    """T0A: Backfill ADS-B state vectors from OpenSky API for a date range."""
    
    API_URL = "https://opensky-network.org/api/states/all"
    
    COLUMNS = [
        "icao24", "callsign", "origin_country", "time_position", 
        "last_contact", "longitude", "latitude", "baro_altitude", 
        "on_ground", "velocity", "true_track", "vertical_rate", 
        "sensors", "geo_altitude", "squawk", "spi", "position_source"
    ]
    
    def __init__(self, manifest_mgr: ManifestManager = None,
                 client_id: str = None, client_secret: str = None,
                 bbox: dict = None, requests_per_minute: int = 12,
                 snapshot_interval_minutes: int = 120):
        self.manifest = manifest_mgr or ManifestManager()
        self.client_id = client_id
        self.client_secret = client_secret
        self.bbox = bbox or {}
        self.min_interval = 60.0 / requests_per_minute  # seconds between requests
        self.snapshot_interval_minutes = snapshot_interval_minutes
        self.history_seconds = 3600 if client_id and client_secret else 0
        self.token = None
        self._authenticate()
    
    def _authenticate(self):
        """Fetch OAuth2 token from OpenSky."""
        if not self.client_id or not self.client_secret:
            return
        auth_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        try:
            response = requests.post(auth_url, data=data, timeout=15)
            response.raise_for_status()
            self.token = response.json().get("access_token")
            logger.info("Successfully fetched OpenSky access token.")
        except requests.RequestException as e:
            logger.error(f"OpenSky API authentication failed: {e}")
    
    @retry_with_backoff(max_retries=3, base_delay=5.0)
    def _fetch_snapshot(self, timestamp_utc: int, day_begin_utc: int, day_end_utc: int) -> Optional[pd.DataFrame]:
        """Fetch one snapshot for a timestamp, carrying day begin/end params."""
        params = {
            'time': timestamp_utc,
            'begin': day_begin_utc,
            'end': day_end_utc,
        }
        params.update(self.bbox)

        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        response = requests.get(self.API_URL, params=params, headers=headers, timeout=30)

        if response.status_code == 401 and self.client_id:
            self._authenticate()
            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
                response = requests.get(self.API_URL, params=params, headers=headers, timeout=30)

        # Some OpenSky tiers reject begin/end on states endpoint. Fall back to time-only query.
        if response.status_code == 403:
            fallback_params = {'time': timestamp_utc}
            fallback_params.update(self.bbox)
            response = requests.get(self.API_URL, params=fallback_params, headers=headers, timeout=30)

        response.raise_for_status()
        data = response.json()

        if data and "states" in data and data["states"]:
            df = pd.DataFrame(data["states"], columns=self.COLUMNS)
            df['timestamp'] = data.get("time", timestamp_utc)
            return df
        return None

    def _fetch_day(self, target_date: date) -> Optional[pd.DataFrame]:
        """Fetch state vectors for a full UTC day using begin/end boundaries."""
        day_begin = int(datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0).timestamp())
        day_end = day_begin + 86400
        step_seconds = max(300, self.snapshot_interval_minutes * 60)

        snapshots: List[pd.DataFrame] = []
        for ts in range(day_begin, day_end, step_seconds):
            snap = self._fetch_snapshot(ts, day_begin, day_end)
            if snap is not None and not snap.empty:
                snapshots.append(snap)
            # API rate limiting (max ~12 req/min by default)
            time.sleep(self.min_interval)

        if not snapshots:
            return None

        day_df = pd.concat(snapshots, ignore_index=True)
        dedup_cols = [c for c in ['icao24', 'callsign', 'last_contact', 'longitude', 'latitude', 'timestamp'] if c in day_df.columns]
        if dedup_cols:
            day_df = day_df.drop_duplicates(subset=dedup_cols, keep='first')
        return day_df
    
    def backfill_date_range(self, start_date: str, end_date: str, output_base: str = "data/raw/opensky"):
        """Fetch ADS-B data day by day for the given date range.
        
        Args:
            start_date: 'YYYY-MM-DD' start date.
            end_date: 'YYYY-MM-DD' end date.
            output_base: Base output directory.
        """
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        
        current = start
        total_days = (end - start).days + 1
        completed = 0
        
        logger.info(f"Starting ADS-B backfill: {start_date} to {end_date} ({total_days} days)")
        
        while current <= end:
            year, month, day = current.year, current.month, current.day
            
            # Check manifest — skip already downloaded days
            if self.manifest.is_downloaded('opensky', year, month, day):
                logger.info(f"Day {current} already downloaded (manifest). Skipping.")
                current += timedelta(days=1)
                completed += 1
                continue
            
            # Partition path
            out_dir = Path(output_base) / f"year={year}" / f"month={month:02d}"
            ensure_dir(out_dir)
            out_file = out_dir / f"states_{current.strftime('%Y%m%d')}.parquet"
            
            if out_file.exists():
                logger.info(f"File {out_file} already exists. Skipping.")
                self.manifest.log_download('opensky', year, month, 'SUCCESS', 0, out_file, day)
                current += timedelta(days=1)
                completed += 1
                continue
            
            try:
                logger.info(f"Fetching ADS-B data for {current} ({completed+1}/{total_days})...")
                df = self._fetch_day(current)
                
                if df is not None and not df.empty:
                    df.to_parquet(out_file, index=False)
                    row_count = len(df)
                    self.manifest.log_download('opensky', year, month, 'SUCCESS', row_count, out_file, day)
                    logger.info(f"  Saved {row_count} states to {out_file}")
                else:
                    self.manifest.log_download('opensky', year, month, 'EMPTY', 0, None, day)
                    logger.warning(f"  No data returned for {current}")
                    
            except Exception as e:
                logger.error(f"  Failed to fetch {current}: {e}")
                self.manifest.log_download('opensky', year, month, 'FAILED', 0, None, day)
            
            current += timedelta(days=1)
            completed += 1
        
        logger.info(f"Backfill complete. Processed {completed}/{total_days} days.")


def main():
    """CLI for data fetching and combining."""
    parser = argparse.ArgumentParser(description="Flight Data Ingestion CLI")
    parser.add_argument("--source", choices=[
        "bts", "euro", "combine_bts", "combine_euro", "opensky_backfill", "both"
    ], default="both", help="Data source to download or process")
    parser.add_argument("--year", type=int, required=False, help="Year to download")
    parser.add_argument("--month", type=int, required=False, help="Month to download (1-12)")
    parser.add_argument("--input_dir", type=str, help="Input directory for combining files")
    parser.add_argument("--output_file", type=str, help="Output file path for combined dataset")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save downloaded files")
    # T0A: Backfill CLI args
    parser.add_argument("--backfill-from", type=str, help="Start date for backfill (YYYY-MM-DD)")
    parser.add_argument("--backfill-to", type=str, help="End date for backfill (YYYY-MM-DD)")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    manifest_mgr = ManifestManager()
    
    # T0A: OpenSky backfill
    if args.source == "opensky_backfill":
        if not args.backfill_from or not args.backfill_to:
            logger.error("--backfill-from and --backfill-to are required for opensky_backfill")
            return
        
        # Load config for credentials and bbox
        try:
            config = load_config()
            opensky_cfg = config.get('opensky', {})
        except Exception:
            opensky_cfg = {}
        
        from dotenv import load_dotenv
        load_dotenv()
        
        backfiller = OpenSkyBackfiller(
            manifest_mgr=manifest_mgr,
            client_id=os.environ.get('OPENSKY_CLIENT_ID') or opensky_cfg.get('clientId'),
            client_secret=os.environ.get('OPENSKY_CLIENT_SECRET') or opensky_cfg.get('clientSecret'),
            bbox=opensky_cfg.get('bbox', {}),
        )
        backfiller.backfill_date_range(args.backfill_from, args.backfill_to)
        return
    
    if args.source == "combine_bts":
        combiner = BTSCombiner()
        input_path = args.input_dir or "data/raw/bts"
        output_path = args.output_file or "data/processed/bts_combined.parquet"
        combiner.combine_csvs(input_path, output_path)
        return
        
    if args.source == "combine_euro":
        combiner = EuroCombiner()
        input_path = args.input_dir or "data/raw/eurocontrol"
        output_path = args.output_file or "data/processed/eurocontrol_combined.parquet"
        combiner.combine_parquets(input_path, output_path)
        return

    if not args.year or not args.month:
        logger.error("--year and --month are required for downloading data.")
        return
        
    if args.source in ["euro", "both"]:
        euro = EurocontrolDownloader(manifest_mgr)
        euro.download_month(args.year, args.month, args.output_dir)
        
    if args.source in ["bts", "both"]:
        bts = BTSDownloader(manifest_mgr)
        bts.download_month(args.year, args.month, args.output_dir)

if __name__ == "__main__":
    main()
