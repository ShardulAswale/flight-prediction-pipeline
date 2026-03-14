import os
import random
import numpy as np
import yaml
from pathlib import Path

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
