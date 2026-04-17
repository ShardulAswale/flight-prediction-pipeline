import time
import logging
from typing import List, Optional, Dict, Any
import datetime
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.utils import ensure_dir

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenSkyClient:
    """Client for ingesting ADS-B state vectors from OpenSky Network."""
    
    API_URL = "https://opensky-network.org/api/states/all"
    
    # Columns expected in the OpenSky response
    COLUMNS = [
        "icao24", "callsign", "origin_country", "time_position", 
        "last_contact", "longitude", "latitude", "baro_altitude", 
        "on_ground", "velocity", "true_track", "vertical_rate", 
        "sensors", "geo_altitude", "squawk", "spi", "position_source"
    ]
    
    def __init__(self, polling_interval: int = 10, client_id: Optional[str] = None, client_secret: Optional[str] = None, bbox: Optional[Dict[str, float]] = None):
        """
        Initialize the OpenSky Client.
        
        Args:
           polling_interval: Time parameter (seconds) between requests.
           client_id: Optional OpenSky API Client ID.
           client_secret: Optional OpenSky API Client Secret.
           bbox: Optional dictionary with lamin, lomin, lamax, lomax for geographic filtering.
        """
        self.polling_interval = polling_interval
        self.client_id = client_id
        self.client_secret = client_secret
        self.bbox = bbox
        self.token: Optional[str] = None
        self.last_status_code: Optional[int] = None
        self._authenticate()

    def _authenticate(self) -> None:
        """Fetch OAuth2 token from OpenSky Keycloak if credentials are provided."""
        if not self.client_id or not self.client_secret:
            return
            
        auth_url = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        try:
            # We don't use self.auth anymore, we use the OAuth form POST
            response = requests.post(auth_url, data=data, timeout=15)
            response.raise_for_status()
            self.token = response.json().get("access_token")
            logger.info("Successfully fetched OpenSky access token.")
        except requests.RequestException as e:
            logger.error(f"OpenSky API authentication failed: {e}")

    def fetch_states(self, timestamp: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch state vectors for a specific Unix timestamp.
        
        Args:
            timestamp: Unix timestamp. If None, fetch current states.
            
        Returns:
            DataFrame containing states or None if failed.
        """
        params: Dict[str, Any] = {}
        if timestamp:
            params['time'] = timestamp
        
        # Apply geographic filtering if configured
        for k, v in (self.bbox or {}).items():
            params[k] = v
            
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
            
        try:
            self.last_status_code = None
            response = requests.get(self.API_URL, params=params, headers=headers, timeout=15)
            self.last_status_code = response.status_code
            
            # Handle token expiration dynamically
            if response.status_code == 401 and self.client_id:
                logger.info("Token might have expired. Re-authenticating...")
                self._authenticate()
                if self.token:
                    headers["Authorization"] = f"Bearer {self.token}"
                    response = requests.get(self.API_URL, params=params, headers=headers, timeout=15)
                    self.last_status_code = response.status_code
                    
            response.raise_for_status()
            data = response.json()
            
            if data and "states" in data and data["states"]:
                df = pd.DataFrame(data["states"], columns=self.COLUMNS)
                # Ensure timestamp is recorded
                df['timestamp'] = data.get("time", timestamp or int(time.time()))
                return df
            return None
            
        except requests.RequestException as e:
            if e.response is not None:
                self.last_status_code = e.response.status_code
            logger.warning(f"Failed to fetch states at {timestamp}: {e}")
            return None

    def fetch_states_range(self, start_time: int, end_time: int) -> pd.DataFrame:
        """
        Fetch state vectors over a specific time range.
        
        Args:
            start_time: Start Unix timestamp.
            end_time: End Unix timestamp.
            
        Returns:
            DataFrame containing all collected state vectors.
        """
        all_states: List[pd.DataFrame] = []
        timestamps = list(range(start_time, end_time, self.polling_interval))
        
        logger.info(f"Fetching states from {start_time} to {end_time} ({len(timestamps)} requests)")
        
        for ts in tqdm(timestamps, desc="Fetching OpenSky Data"):
            df = self.fetch_states(ts)
            if getattr(self, 'last_status_code', None) == 403:
                logger.warning("Encountered repeated HTTP 403 while fetching range; stopping further requests.")
                break
            if df is not None:
                all_states.append(df)
            time.sleep(1) # Polite sleep for rate limiting, anonymous API rate limit is 400 requests/day
            
        if all_states:
            return pd.concat(all_states, ignore_index=True)
        return pd.DataFrame(columns=self.COLUMNS + ["timestamp"])

    def save_raw_states(self, df: pd.DataFrame, output_dir: str | Path, date_str: str):
        """
        Save the concatenated raw states to a partitioned Parquet file.
        
        Args:
            df: DataFrame of states.
            output_dir: Directory to save the file.
            date_str: Date string for the filename (YYYYMMDD).
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to save.")
            return
            
        ensure_dir(output_dir)
        output_path = Path(output_dir) / f"states_{date_str}.parquet"
        
        logger.info(f"Saving {len(df)} records to {output_path}")
        df.to_parquet(output_path, index=False)
