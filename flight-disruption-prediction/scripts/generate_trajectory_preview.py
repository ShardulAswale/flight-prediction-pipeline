"""Generate a map and altitude chart from compact trajectory sketches.

Examples
--------
Generate a preview for the first available full flight:
    python scripts/generate_trajectory_preview.py

Generate a preview for a specific trajectory id:
    python scripts/generate_trajectory_preview.py --trajectory-id AAL1540_...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.trajectory_preview import generate_trajectory_preview
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save route map and altitude chart from trajectory sketches.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--trajectory-id", default=None, help="Specific trajectory_id to preview.")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "trajectory_previews"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config.get("paths", {})
    processed = PROJECT_ROOT / paths.get("processed_data_dir", "data/processed")
    sketches_path = processed / paths.get("trajectory_sketches_file", "trajectory_sketches.parquet")
    features_path = processed / paths.get("features_file", "trajectory_features.parquet")

    result = generate_trajectory_preview(
        sketches_path=sketches_path,
        features_path=features_path,
        out_dir=Path(args.out_dir),
        trajectory_id=args.trajectory_id,
    )
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
