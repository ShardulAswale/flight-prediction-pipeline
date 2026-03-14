import time
import argparse
import logging
from src.utils import load_config
from src.opensky_client import OpenSkyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Collect OpenSky ADS-B Data")
    parser.add_argument('--hours', type=int, help='Override collection hours')
    args = parser.parse_args()

    config = load_config()
    hours = args.hours or config['opensky']['collection_hours']
    interval = config['opensky']['polling_interval_seconds']
    
    client = OpenSkyClient(
        polling_interval=interval,
        client_id=config['opensky'].get('clientId'),
        client_secret=config['opensky'].get('clientSecret'),
        bbox=config['opensky'].get('bbox')
    )
    
    end_time = int(time.time())
    start_time = end_time - (hours * 3600)
    
    logger.info(f"Starting collection for {hours} hours...")
    df = client.fetch_states_range(start_time, end_time)
    
    date_str = time.strftime("%Y%m%d", time.localtime(end_time))
    client.save_raw_states(df, config['paths']['raw_data_dir'], date_str)
    
    logger.info("Data collection complete.")

if __name__ == "__main__":
    main()
