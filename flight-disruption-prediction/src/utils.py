import os
import random
import numpy as np
import yaml
import pandas as pd
from pathlib import Path
import zipfile
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_dir(directory: str | Path):
    """Ensure that a directory exists."""
    os.makedirs(directory, exist_ok=True)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)

def unzip_file(zip_path: str | Path, extract_to: str | Path) -> Optional[Path]:
    """Unzip a file and return the path to the first CSV found."""
    zip_path = Path(zip_path)
    extract_to = Path(extract_to)
    ensure_dir(extract_to)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            for file in extract_to.glob("*.csv"):
                return file
    except Exception as e:
        logger.error(f"Failed to unzip {zip_path}: {e}")
    return None

def preprocess_eurocontrol(df: pd.DataFrame) -> pd.DataFrame:
    """Map Eurocontrol OPDI fields to standard schema."""
    mapping = {
        'flt_id': 'callsign',
        'adep': 'origin_airport',
        'ades': 'destination_airport',
        'first_seen': 'actual_dep',
        'last_seen': 'actual_arr',
    }
    
    df_out = df.rename(columns=mapping)
    
    if 'scheduled_dep' not in df_out.columns:
        df_out['scheduled_dep'] = df_out.get('actual_dep')
    if 'scheduled_arr' not in df_out.columns:
        df_out['scheduled_arr'] = df_out.get('actual_arr')
        
    df_out['cancelled'] = 0
    return df_out

def preprocess_bts(df: pd.DataFrame) -> pd.DataFrame:
    """Map BTS TranStats fields to standard schema. Fully vectorized for speed."""
    mapping = {
        'FL_DATE': 'date',
        'ORIGIN': 'origin_airport',
        'DEST': 'destination_airport',
        'CANCELLED': 'cancelled'
    }
    
    # Build callsign from carrier + flight number
    if 'OP_UNIQUE_CARRIER' in df.columns and 'OP_CARRIER_FL_NUM' in df.columns:
        df['callsign'] = df['OP_UNIQUE_CARRIER'].astype(str) + df['OP_CARRIER_FL_NUM'].astype(str)
    
    df_out = df.rename(columns=mapping)
    
    # ── Vectorized datetime construction ─────────────────────────────────
    # BTS times are HHMM integers (e.g. 1430 = 14:30, 2400 = 00:00 next day)
    if 'date' in df_out.columns:
        base_date = pd.to_datetime(df_out['date'], errors='coerce')
        
        time_pairs = {
            'scheduled_dep': 'CRS_DEP_TIME',
            'actual_dep':    'DEP_TIME',
            'scheduled_arr': 'CRS_ARR_TIME',
            'actual_arr':    'ARR_TIME',
        }
        
        for std_name, raw_col in time_pairs.items():
            if raw_col in df_out.columns:
                t = pd.to_numeric(df_out[raw_col], errors='coerce').fillna(-1).astype(int)
                # Clamp 2400 → 0000
                t = t.where(t != 2400, 0)
                hours = t // 100
                minutes = t % 100
                # Build timedelta, invalid entries get NaT
                td = pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m')
                td = td.where((hours >= 0) & (hours < 24) & (minutes >= 0) & (minutes < 60))
                df_out[std_name] = base_date + td
    
    return df_out
