# Flight Disruption Prediction Pipeline

Machine Learning Approaches to Probabilistic Flight Disruption Prediction Using ADS-B Trajectory Data.

## Project Goal
This project provides a reproducible Python pipeline to collect ADS-B trajectory data from OpenSky, process it into flight segments, extract features, and build a clean machine learning dataset for flight disruption prediction.

## Data Pipeline
1. **OpenSky Ingestion**: Collects raw state vectors from the OpenSky API.
2. **Trajectory Reconstruction**: Groups state vectors by aircraft and segments them into trajectories based on time gaps.
3. **Feature Extraction**: Computes statistical and physical features for each trajectory.
4. **Dataset Builder**: Combines all steps to generate the final ML dataset.

## How to Run the Notebooks
To run the notebooks individually, start Jupyter in the project root:
```bash
jupyter notebook notebooks/
```
1. Run `01_opensky_ingestion.ipynb` to download the raw data.
2. Run `02_trajectory_reconstruction.ipynb` to build flight trajectories.
3. Run `03_feature_extraction.ipynb` to calculate ML features.
4. Run `04_visualisation.ipynb` to visualise trajectories on an interactive map.

## How to Build the Dataset
Alternatively, you can run the full pipeline using scripts:
```bash
python scripts/collect_data.py
python scripts/build_dataset.py
```

## Structure
- `data/raw`: Raw state vector parquet files.
- `data/processed`: Trajectories, features, and final ml_dataset.
- `src`: Core modules for ingestion, Reconstruction, engineering, and visualisation.
- `notebooks`: Orchestration notebooks for step-by-step execution.
- `configs/config.yaml`: Configuration parameters for execution.
- `scripts`: Executable scripts for automating steps.
