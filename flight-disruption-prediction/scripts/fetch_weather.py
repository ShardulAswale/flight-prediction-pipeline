"""
T0D: Fetch real METAR weather data from Iowa Environmental Mesonet (IEM) ASOS API.
Downloads hourly weather observations for specified airports and date range.

Usage:
    python scripts/fetch_weather.py --start 2022-05-01 --end 2022-05-31
    python scripts/fetch_weather.py --start 2022-05-01 --end 2022-05-31 --stations KJFK,KLAX,EGLL
"""
import argparse
import logging
import sys
import os
import time
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Broad fallback set: busy US and European ICAO airport codes.
DEFAULT_STATIONS = [
    'KATL', 'KORD', 'KDFW', 'KDEN', 'KJFK', 'KLAX', 'KSFO', 'KLAS', 'KSEA', 'KCLT',
    'KMCO', 'KMIA', 'KPHX', 'KIAH', 'KBOS', 'KEWR', 'KMSP', 'KDTW', 'KPHL', 'KLGA',
    'KBWI', 'KSLC', 'KSAN', 'KIAD', 'KDCA', 'KMDW', 'KTPA', 'KFLL', 'KPDX', 'PHNL',
    'KSTL', 'KBNA', 'KAUS', 'KMSY', 'KHOU', 'KDAL', 'KOAK', 'KSMF', 'KRDU', 'KSJC',
    'KCLE', 'KPIT', 'KCVG', 'KIND', 'KMCI', 'KSAT', 'KJAX', 'KMEM', 'KBDL', 'KRSW',
    'EGLL', 'EGKK', 'EGSS', 'EGCC', 'EIDW', 'EHAM', 'EBBR', 'ELLX', 'LFPG', 'LFPO',
    'EDDF', 'EDDM', 'EDDL', 'EDDB', 'LSZH', 'LOWW', 'LEMD', 'LEBL', 'LEPA', 'LIRF',
    'LIMC', 'LIPZ', 'LTFM', 'LTBA', 'ESSA', 'ENGM', 'EKCH', 'EFHK', 'EPWA', 'LKPR',
]

IEM_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.yaml'


def infer_stations_from_pipeline_data(max_stations: int = 50) -> list[str]:
    """Infer the busiest airport codes from the combined BTS/Eurocontrol datasets."""
    config = load_config(str(CONFIG_PATH))
    ingest_cfg = config.get('ingestion', {})

    candidate_files = [
        PROJECT_ROOT / ingest_cfg.get('bts_combined_file', 'data/processed/bts_combined.parquet'),
        PROJECT_ROOT / ingest_cfg.get('euro_combined_file', 'data/processed/eurocontrol_combined.parquet'),
    ]

    airport_counts = pd.Series(dtype='int64')
    for file_path in candidate_files:
        if not file_path.exists():
            continue
        try:
            df = pd.read_parquet(file_path, columns=['origin', 'destination'])
            airports = pd.concat([df['origin'], df['destination']], ignore_index=True).dropna().astype(str).str.upper().str.strip()
            airport_counts = airport_counts.add(airports.value_counts(), fill_value=0)
        except Exception as exc:
            logger.warning(f"Could not infer stations from {file_path}: {exc}")

    inferred = airport_counts.sort_values(ascending=False).head(max_stations).index.tolist()
    if inferred:
        logger.info(f"Inferred {len(inferred)} station codes from pipeline schedule data.")
    return inferred


def fetch_station_data(station: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch ASOS data from IEM for a single station."""
    params = {
        'station': station,
        'data': 'tmpf,dwpf,sknt,vsby,p01i',
        'tz': 'Etc/UTC',
        'format': 'comma',
        'latlon': 'no',
        'elev': 'no',
        'missing': 'M',
        'trace': 'T',
        'direct': 'no',
        'report_type': '3',  # METAR
        'year1': start_date[:4],
        'month1': start_date[5:7],
        'day1': start_date[8:10],
        'year2': end_date[:4],
        'month2': end_date[5:7],
        'day2': end_date[8:10],
    }
    
    try:
        response = requests.get(IEM_URL, params=params, timeout=60)
        response.raise_for_status()
        
        # IEM returns CSV with comment header lines starting with #
        lines = response.text.strip().split('\n')
        data_lines = [l for l in lines if not l.startswith('#')]
        
        if len(data_lines) < 2:
            logger.warning(f"No data returned for {station}")
            return pd.DataFrame()
        
        from io import StringIO
        df = pd.read_csv(StringIO('\n'.join(data_lines)), na_values=['M', 'T'])
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch data for {station}: {e}")
        return pd.DataFrame()


def process_and_save(stations: list, start_date: str, end_date: str, output_path: str):
    """Fetch data for all stations and save as unified Parquet."""
    all_dfs = []
    
    for i, station in enumerate(stations):
        logger.info(f"Fetching {station} ({i+1}/{len(stations)})...")
        df = fetch_station_data(station, start_date, end_date)
        
        if df.empty:
            continue
        
        # Map IEM columns to our schema
        col_map = {
            'station': 'airport_code',
            'valid': 'timestamp',
            'tmpf': 'temperature',      # Temperature in Fahrenheit
            'dwpf': 'dewpoint',          # Dewpoint in Fahrenheit
            'sknt': 'wind_speed',        # Wind speed in knots
            'vsby': 'visibility',        # Visibility in miles
            'p01i': 'precipitation',     # Precipitation in inches
        }
        
        available_cols = {k: v for k, v in col_map.items() if k in df.columns}
        df_mapped = df.rename(columns=available_cols)
        
        # Keep only our schema columns
        keep_cols = ['airport_code', 'timestamp', 'wind_speed', 'visibility', 
                     'temperature', 'precipitation']
        existing = [c for c in keep_cols if c in df_mapped.columns]
        df_mapped = df_mapped[existing].copy()
        
        # Ensure correct types
        df_mapped['timestamp'] = pd.to_datetime(df_mapped['timestamp'], utc=True, errors='coerce')
        for num_col in ['wind_speed', 'visibility', 'temperature', 'precipitation']:
            if num_col in df_mapped.columns:
                df_mapped[num_col] = pd.to_numeric(df_mapped[num_col], errors='coerce')
        
        all_dfs.append(df_mapped)
        
        # Polite rate limiting
        time.sleep(1)
    
    if not all_dfs:
        logger.error("No data fetched for any station.")
        return
    
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.dropna(subset=['timestamp'])
    
    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    
    logger.info(f"Saved {len(combined):,} weather observations to {output}")
    logger.info(f"  Stations: {combined['airport_code'].nunique()}")
    logger.info(f"  Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}")
    logger.info(f"  Columns: {list(combined.columns)}")


def main():
    parser = argparse.ArgumentParser(description="Fetch METAR weather data from IEM ASOS API")
    parser.add_argument('--start', type=str, default='2022-05-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2022-05-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--stations', type=str, default=None,
                        help='Comma-separated ICAO station codes (default: top 50)')
    parser.add_argument('--max-stations', type=int, default=80,
                        help='Maximum number of inferred station codes when --stations is not supplied')
    parser.add_argument('--output', type=str, default='data/raw/metar.parquet',
                        help='Output Parquet file path')
    
    args = parser.parse_args()
    
    if args.stations:
        stations = [station.strip().upper() for station in args.stations.split(',') if station.strip()]
    else:
        inferred = infer_stations_from_pipeline_data(max_stations=args.max_stations)
        stations = inferred or DEFAULT_STATIONS
    
    process_and_save(stations, args.start, args.end, args.output)


if __name__ == "__main__":
    main()
