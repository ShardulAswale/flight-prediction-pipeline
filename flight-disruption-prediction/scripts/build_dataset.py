import glob
import logging
import pandas as pd
from pathlib import Path
from src.utils import load_config, ensure_dir
from src.trajectory_builder import TrajectoryBuilder
from src.feature_engineering import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config = load_config()
    
    raw_dir = Path(config['paths']['raw_data_dir'])
    processed_dir = Path(config['paths']['processed_data_dir'])
    ensure_dir(processed_dir)
    
    # 1. Load Raw States
    raw_files = glob.glob(str(raw_dir / "states_*.parquet"))
    if not raw_files:
        logger.error("No raw state files found. Run collect_data.py first.")
        return
        
    logger.info(f"Loading {len(raw_files)} raw state files...")
    df_raw = pd.concat([pd.read_parquet(f) for f in raw_files], ignore_index=True)
    logger.info(f"Loaded {len(df_raw)} total state vectors.")
    
    # 2. Trajectory Reconstruction
    logger.info("Starting trajectory reconstruction...")
    tb = TrajectoryBuilder(max_gap_minutes=config['trajectory']['max_gap_minutes'])
    df_traj = tb.build_flight_segments(df_raw)
    
    trajectories_file = processed_dir / config['paths']['trajectories_file']
    tb.save_trajectories(df_traj, trajectories_file)
    logger.info(f"Saved trajectories to {trajectories_file}")
    
    # 3. Feature Extraction
    logger.info("Starting feature extraction...")
    fe = FeatureExtractor()
    df_features = fe.extract_features(df_traj)
    
    features_file = processed_dir / config['paths']['features_file']
    fe.save_features(df_features, features_file)
    logger.info(f"Saved trajectory features to {features_file}")
    
    # 4. Final ML Dataset Creation
    # In a full project, this might involve merging labels, but here the features 
    # are the base ML dataset.
    ml_dataset_file = processed_dir / config['paths']['ml_dataset_file']
    df_features.to_parquet(ml_dataset_file, index=False)
    logger.info(f"ML dataset successfully built at {ml_dataset_file}")

if __name__ == "__main__":
    main()
