# Flight Disruption Prediction Pipeline

Machine-learning pipeline for predicting flight disruption using ADS-B surveillance signals, airline schedule labels, weather, airport, and congestion features.

## Current Architecture

The project now uses a **schedule-aware direct feature pipeline** by default.

Instead of reconstructing and storing a huge point-level `trajectories.parquet`, the pipeline:

1. Loads schedule data first.
2. Builds callsign + time windows for flights.
3. Filters ADS-B pings only to relevant scheduled windows.
4. Downsamples pings inside each flight window.
5. Saves compact route geometry to `trajectory_sketches.parquet` for maps, route deviation, and en-route weather.
6. Computes compact flight-level feature vectors directly.
7. Merges schedule labels and weather.
8. Trains models and saves explainability artifacts.

This keeps the route-signal idea while avoiding unnecessary compute and massive raw point-level intermediate files.

## Main Commands

Create a monthly ingestion plan without downloading:

```powershell
python scripts/run_master_ingestion.py --start-date 2022-01-01 --end-date 2022-06-30
```

Execute monthly ingestion/downloads:

```powershell
python scripts/run_master_ingestion.py --start-date 2022-01-01 --end-date 2022-06-30 --execute
```

Run the default schedule-aware pipeline:

```powershell
python main.py --stage features --force
python main.py --stage merge --force
python main.py --stage weather --force
python main.py --stage label --force
python main.py --stage validate --force
python main.py --stage train --force
```

Generate saved visual reports:

```powershell
python scripts/generate_pipeline_report.py
```

Run tests:

```powershell
python -m pytest tests -q --basetemp .pytest_tmp_run
```

## Notebook Order

The notebooks are thin control and reporting layers. Heavy actions are behind explicit flags so a casual `Run All` does not redownload or retrain by accident.

1. `01_data_ingestion.ipynb`: monthly master ingestion plan/control.
2. `02_trajectory_reconstruction.ipynb`: ADS-B trace strategy and quality review.
3. `03_feature_extraction.ipynb`: schedule-aware feature extraction control and summary.
4. `04_dataset_integration.ipynb`: merge, weather, labels, validation control.
5. `05_visualisation.ipynb`: saved pipeline visual report.
6. `06_feature_selection.ipynb`: signal audit and feature diagnostics.
7. `07_model_training.ipynb`: training control and artifact preview.
8. `08_model_evaluation.ipynb`: model comparison, SHAP, evaluation outputs.
9. `09_frontend_validation.ipynb`: Streamlit readiness checks.

## Output Layout

- `data/raw/`: downloaded source files.
- `data/processed/trajectory_features.parquet`: one row per observed scheduled flight.
- `data/processed/trajectory_sketches.parquet`: compact point-level route geometry, usually one point every 10 minutes plus endpoints/phase detail.
- `data/processed/`: other compact parquet outputs used by the pipeline.
- `outputs/ingestion_control/`: monthly ingestion plans, flow charts, and status tables.
- `outputs/pipeline_report/`: saved report charts and tables.
- `outputs/notebook_reports/`: notebook-specific CSV/PNG outputs.
- `logs/`: pretty logs, validation reports, quality reports, timing metadata.
- `models/`: trained models, preprocessors, feature lists, and model reports.

## Notes

- `data/`, `models/`, `outputs/`, and `logs/` are local artifacts and should generally stay out of Git.
- The old full point-level trajectory reconstruction path still exists for debugging, but it is no longer the default training path.
- Weather coverage depends on available METAR airport coverage and may remain partial unless more airports are fetched.
- Airport METAR weather remains the default weather layer. En-route weather lookup from `trajectory_sketches.parquet` is available through `enroute_weather.enabled`, but it is disabled by default because it can create many external API requests.
