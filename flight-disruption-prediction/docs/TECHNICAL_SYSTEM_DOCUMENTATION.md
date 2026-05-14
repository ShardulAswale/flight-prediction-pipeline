# Technical System Documentation - Flight Disruption Prediction

Generated for dissertation support. Evidence checked against the repository and generated artifacts in `C:/Code/flight-disruption-prediction` on 2026-04-28.

This document describes the project as it exists in code and outputs. It separates implemented behaviour from partial work, recommendations, and claims that should not be made without more evidence.

## 1. Executive summary

### Project aim

The project builds a machine-learning pipeline to predict flight disruption risk using flight schedule data, ADS-B aircraft state vectors, airport weather, congestion variables, and trajectory-derived features. The current model target is binary:

| Target class | Meaning |
|---|---|
| `Normal` | Flight is not late beyond the selected delay threshold and is not cancelled. |
| `Disrupted` | Flight is either `Late` or `Cancelled` in the original labels. |

The original labels are still preserved in the final dataset as `label`, `label_original`, and `disruption_subtype`, but the trained target is controlled by `training.target_mode: binary_disrupted` in `configs/config.yaml`.

### Current working architecture

The current default system is the schedule-aware direct feature pipeline. It is not the original full point-level trajectory reconstruction pipeline.

Current flow:

1. Load schedule data first.
2. Build callsign and time windows for scheduled flights.
3. Filter the large ADS-B table to relevant scheduled callsigns and windows.
4. Downsample ADS-B pings inside each matched flight window.
5. Compute one compact feature vector per observed scheduled flight.
6. Save route sketches separately for visualisation and optional future route/weather features.
7. Merge with schedule labels.
8. Add airport METAR weather and contextual features.
9. Label flights.
10. Validate quality.
11. Train logistic regression, random forest, and XGBoost.
12. Save metrics, plots, model artifacts, and UI-ready prediction files.

This design is implemented mainly in `main.py`, `src/schedule_aware_features.py`, `src/feature_engineering.py`, `src/merge.py`, `src/weather.py`, `src/feature_enrichment.py`, `src/labeling.py`, `src/quality_gates.py`, and `src/model_training.py`.

### Verified dataset scale

The following counts were verified from generated parquet metadata and logs:

| Dataset | Path | Verified rows | Notes |
|---|---:|---:|---|
| ADS-B combined | `data/processed/adsb_combined.parquet` | 1,296,709,098 | Large OpenSky state-vector input. |
| BTS combined | `data/processed/bts_combined.parquet` | 2,683,015 | US schedule, delay, and cancellation source. |
| Eurocontrol combined | `data/processed/eurocontrol_combined.parquet` | 4,339,106 | Downloaded and normalised, but not used in the current final labelled model because current merge config uses BTS only. |
| Trajectory features | `data/processed/trajectory_features.parquet` | 400,222 | One row per observed scheduled flight before final merge/label filtering. |
| Trajectory sketches | `data/processed/trajectory_sketches.parquet` | 6,462,924 | Compact route geometry, not the model target table. |
| Final ML dataset | `data/processed/ml_dataset.parquet` | 393,339 | Final labelled modelling dataset. |
| METAR weather | `data/raw/metar.parquet` | 344,583 | Airport-level weather observations. |

Final original label distribution from `logs/label_distribution.json`:

| Original label | Count | Percent |
|---|---:|---:|
| `Normal` | 307,701 | 78.23% |
| `Late` | 85,103 | 21.64% |
| `Cancelled` | 535 | 0.14% |

Binary target distribution:

| Binary label | Count |
|---|---:|
| `Normal` | 307,701 |
| `Disrupted` | 85,638 |

### Current model result summary

The current saved `models/model_comparison.json` ranks XGBoost as the best model. The final training run was regenerated on 2026-04-28 with subtype weights `Normal=1.0`, `Late=1.0`, and `Cancelled=1.0`, matching the current `configs/config.yaml`.

| Model | Accuracy | Weighted F1 | Macro F1 | Disrupted recall | Disrupted precision | Cohen's kappa | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---|
| XGBoost | 0.7964 | 0.8047 | 0.7406 | 0.7118 | 0.5496 | 0.4843 | 0.6994 |
| Random forest | 0.8189 | 0.7818 | 0.6552 | 0.2781 | 0.8390 | 0.3411 | 0.6609 |
| Logistic regression | 0.6530 | 0.6779 | 0.6018 | 0.6297 | 0.3610 | 0.2303 | 0.4123 |

Interpretation:

- XGBoost gives the best overall balance in the saved current artifact.
- Logistic regression can achieve high disrupted recall, but with poor precision and accuracy.
- Random forest has high disrupted precision and high overall accuracy, but it misses many disrupted flights.
- PR-AUC is populated in the corrected final binary run. The ROC curve should be cited only from the regenerated figure that uses `Disrupted` as the positive class.

### Main caveats for dissertation writing

| Caveat | Dissertation impact |
|---|---|
| Current model is binary disruption prediction, not reliable cancellation-specific prediction. | Avoid claiming the system can predict cancellations as a separate operational class. |
| Current final labelled dataset uses BTS only after merge. | Do not claim Eurocontrol contributes to final supervised training unless rerun. |
| Older ROC/PR-AUC artifacts used the wrong positive-class framing. | Use only the regenerated final ROC and PR-AUC outputs where `Disrupted` is the positive class. |
| Route deviation code exists but current trained feature list does not include deviation features. | Do not claim route-deviation features improved the final model. |
| En-route weather code exists but `enroute_weather.enabled: false`. | Do not claim en-route weather was used in final training. |
| UI probabilities are raw model probabilities. | Do not describe them as calibrated probabilities. |

Suitable dissertation use:

| Chapter | Use |
|---|---|
| Chapter 3 Methodology | Pipeline design, preprocessing, feature engineering, modelling choices. |
| Chapter 4 Results | Dataset counts, model metrics, weighted-run comparison, generated figures. |
| Chapter 5 Critical Discussion | Runtime limitations, class imbalance, PR-AUC logging issue, partial route/weather work, cancellation limitation. |

## 2. Part 1: Current Working System and Experiments

### 2.1 Repository structure

| Area | Main files/folders | Purpose | Implementation status |
|---|---|---|---|
| Pipeline entrypoint | `main.py` | Orchestrates ingestion, features, merge, weather, label, validate, train, and drift stages. | Implemented. |
| Configuration | `configs/config.yaml` | Controls dates, source files, feature mode, weather, training target, weights, enabled models, and paths. | Implemented. |
| Source modules | `src/` | Core reusable pipeline logic. | Implemented with some optional/partial modules. |
| Scripts | `scripts/` | CLI helpers for ingestion, weather fetching, reports, EDA, and trajectory previews. | Implemented. |
| Notebooks | `notebooks/` | Dissertation-facing control, visualisation, and experiment notebooks. | Implemented as thin wrappers/reports. |
| Streamlit UI | `app.py`, `pages/`, `predict_app.py`, `standalone/predict_app.py` | Dashboard and prediction explorer. | Implemented, recently simplified for stability. |
| Models | `models/` | Saved trained models, preprocessors, label encoder, feature list, best XGBoost parameters, comparison JSON. | Implemented. |
| Outputs | `outputs/` | Figures, tables, EDA outputs, pipeline reports, threshold experiments, SHAP outputs. | Implemented. |
| Logs | `logs/` | Run metadata, quality reports, label distribution, matching reports. | Implemented. |
| Snapshots | `weighted_1_1_1_run`, `weighted_1_2_5_run`, `weighted_1_2_10_run` | Preserved comparison runs for sample-weight experiments. | Implemented. |
| Tests | `tests/` | Unit/integration tests for validation, labels, normalisation, quality gates, and pipeline behaviours. | Implemented. |

### 2.2 Major source files and modules

| File | Main responsibility | Important functions/classes | Dissertation use |
|---|---|---|---|
| `main.py` | Pipeline orchestration. | `stage_ingest`, `stage_features`, `stage_merge`, `stage_weather`, `stage_label`, `stage_validate`, `stage_train`, `stage_drift`. | Chapter 3 pipeline methodology. |
| `src/data_ingestion.py` | Download and combine Eurocontrol, BTS, and OpenSky backfill data. | `ManifestManager`, `EurocontrolDownloader`, `BTSDownloader`, `BTSCombiner`, `EuroCombiner`, `OpenSkyBackfiller`. | Chapter 3 data acquisition. |
| `src/ingestion_control.py` | Master monthly ingestion planning across sources. | `MasterIngestionConfig`, `month_windows`, `build_monthly_plan`. | Chapter 3 data collection design. |
| `src/normalization.py` | Standardises source schedule data into shared columns. | Source-specific normalisation helpers. | Chapter 3 preprocessing. |
| `src/schedule_aware_features.py` | Current default ADS-B feature extraction path. | `ScheduleAwareFeatureConfig`, `extract_schedule_aware_features`. | Chapter 3 feature engineering and optimisation. |
| `src/trajectory_builder.py` | Legacy and diagnostic full trajectory reconstruction. | Quality-aware segmentation, parallel reconstruction, prototype helpers. | Chapter 5 development evolution and limitations. |
| `src/feature_engineering.py` | Computes flight-level features from ADS-B traces. | `TrajectoryAccumulator`, `FeatureExtractor`. | Chapter 3 feature extraction. |
| `src/merge.py` | Matches ADS-B-derived flight rows to schedule rows. | `DataMerger`. | Chapter 3 dataset integration. |
| `src/weather.py` | Adds airport METAR weather. | `WeatherIntegrator`. | Chapter 3 weather integration. |
| `src/feature_enrichment.py` | Adds temporal, airport, congestion, route distance, and destination weather features. | `add_temporal_features`, `add_airport_features`, `add_congestion_features`, `add_destination_weather`, `enrich_all`. | Chapter 3 engineered features. |
| `src/labeling.py` | Generates original and binary labels. | `LabelGenerator.generate_labels`, `strip_leaky_columns`, `standardize_dataset`. | Chapter 3 target construction. |
| `src/schemas.py` | Defines schemas, feature columns, categories, and drop columns. | `ML_FEATURE_COLUMNS`, `ML_DROP_COLUMNS`, schema dictionaries. | Chapter 3 reproducibility. |
| `src/quality_gates.py` | Validates missingness, duplicates, label balance, and feature quality. | Quality gate functions/classes. | Chapter 4 data quality and Chapter 5 limitations. |
| `src/model_training.py` | Prepares data, trains models, evaluates, saves artifacts and plots. | `ModelTrainer`, `prepare_data`, `evaluate`, model training methods. | Chapter 3 modelling and Chapter 4 results. |
| `src/route_deviation.py` | Computes route deviation features from route geometries. | `add_route_deviation_features`. | Partial work; Chapter 5 future improvement. |
| `src/enroute_weather.py` | Optional Open-Meteo en-route weather from sketch points. | `EnrouteWeatherConfig`, `add_enroute_weather_features`. | Partial work; Chapter 5 future improvement. |
| `src/pipeline_reporting.py` | Produces saved report figures and tables. | Reporting helpers. | Chapter 4 visual outputs. |
| `src/streamlit_utils.py` | Streamlit utility loading and caching functions. | `read_parquet_limited`, `read_parquet_filtered`, `load_model`, `load_json`. | UI implementation evidence. |
| `src/drift_detector.py` | Optional monthly drift analysis. | `DriftDetector`. | Partial/optional monitoring work. |

### 2.3 Scripts

| Script | Purpose | Status |
|---|---|---|
| `scripts/run_master_ingestion.py` | Builds and optionally executes a monthly ingestion plan. | Implemented. |
| `scripts/fetch_opensky_samples.py` | Downloads OpenSky sample archives, converts AVRO tar files, combines ADS-B parquet output. | Implemented and central to ADS-B acquisition. |
| `scripts/fetch_adsbexchange.py` | Alternative ADS-B Exchange fetch/conversion utility. | Implemented but not evidenced as current final source. |
| `scripts/fetch_weather.py` | Fetches airport METAR weather data. | Implemented. |
| `scripts/generate_pipeline_report.py` | Generates saved tables/charts for reporting. | Implemented. |
| `scripts/generate_trajectory_preview.py` | Generates route map and altitude chart from trajectory sketches. | Implemented. |
| `scripts/eda_signal_audit.py` | Audits feature group signal and generates EDA outputs. | Implemented. |
| `scripts/build_dataset.py`, `scripts/collect_data.py` | Earlier data build/collection utilities. | Implemented but secondary to the current main pipeline. |

### 2.4 Notebooks

The notebooks are control and reporting layers. Heavy stages are gated so a casual notebook run does not redownload or retrain everything.

| Notebook | Purpose | Main output/use |
|---|---|---|
| `01_data_ingestion.ipynb` | Master ingestion control and dataset inventory. | Ingestion plan/status figures and inventory tables. |
| `02_trajectory_reconstruction.ipynb` | ADS-B trace strategy and quality review. | Trace diagnostics, flow diagrams, quality summaries, route preview. |
| `03_feature_extraction.ipynb` | Schedule-aware feature extraction control and summaries. | Feature preview, quality counts, trajectory sketch summary. |
| `04_dataset_integration.ipynb` | Merge, weather, label, and validation control. | Final label distribution, dataset inventory, dataset sizes. |
| `05_visualisation.ipynb` | Pipeline visualisation report. | Saved visual report figures from `outputs/pipeline_report/`. |
| `06_feature_selection.ipynb` | Signal audit and feature diagnostics. | EDA feature group outputs. |
| `07_model_training.ipynb` | Training control and artifact preview. | Model artifacts and training summaries. |
| `08_model_evaluation.ipynb` | Model comparison and explainability review. | Confusion matrices, ROC curves, SHAP, ranking. |
| `09_frontend_validation.ipynb` | Streamlit readiness checks. | Syntax and artifact readiness tables. |
| `10_threshold_hyperparameter_experiments.ipynb` | Threshold, hyperparameter, and weighting experiments. | Threshold experiment CSV/parquet and comparison plots. |

### 2.5 Data sources

| Source | Current role | Evidence | Notes |
|---|---|---|---|
| OpenSky ADS-B sample archive | Main raw movement signal source. | `data/processed/adsb_combined.parquet`, `scripts/fetch_opensky_samples.py`, `configs/config.yaml`. | Current ADS-B table is very large: 1.296B rows. |
| BTS | Main final schedule, delay, and cancellation label source. | `data/processed/bts_combined.parquet`, `src/data_ingestion.py`, `src/labeling.py`, merge report. | Final labelled dataset source counts show `bts: 393,339`. |
| Eurocontrol | Downloaded and normalised schedule source. | `data/processed/eurocontrol_combined.parquet`. | Current `merge.schedule_sources: ["bts"]`, so Eurocontrol is not part of the final labelled model. |
| METAR/IEM ASOS | Airport weather source. | `data/raw/metar.parquet`, `src/weather.py`, `scripts/fetch_weather.py`. | Airport weather only; not en-route weather in current run. |
| Open-Meteo en-route weather | Optional future/partial source. | `src/enroute_weather.py`, `configs/config.yaml`. | Disabled by default: `enroute_weather.enabled: false`. |

### 2.6 Current configuration

Important current values from `configs/config.yaml`:

| Config area | Current value | Meaning |
|---|---|---|
| Ingestion date window | `2022-01-01` to `2022-06-30` | Six-month data window. |
| OpenSky sample schedule | Mondays, all 24 hours | Weekly sampled ADS-B strategy. |
| Feature mode | `schedule_aware_direct` | Current default feature pipeline avoids full `trajectories.parquet`. |
| Schedule source for features | `bts` | Feature windows are based on BTS schedules. |
| Merge schedule source | `bts` | Final labelled merge uses BTS only. |
| Downsample interval | 60 seconds | Feature extraction pings are reduced before feature computation. |
| Sketch interval | 600 seconds | Saved route sketches keep coarser route geometry for maps. |
| Training target mode | `binary_disrupted` | Training uses `Normal` vs `Disrupted`. |
| Enabled models | Logistic regression, random forest, XGBoost | These are the three model families currently trained. |
| XGBoost trials | 10 | Hyperparameter search is limited for runtime. |
| En-route weather | Disabled | Code exists but final run does not use it. |
| Current subtype weights in config | Normal 1.0, Late 1.0, Cancelled 1.0 | The config currently shows unweighted subtype multipliers. |

Final-run alignment:

- `configs/config.yaml` currently shows `1/1/1` subtype weights.
- `models/model_comparison.json` was regenerated on 2026-04-28 using the same `1/1/1` weights.
- The older `weighted_*_run` folders remain useful as experiment snapshots, not as the active final model artifacts.

### 2.7 Pipeline stages

#### Stage: ingestion

Implemented in `main.py::stage_ingest`, `src/data_ingestion.py`, `src/ingestion_control.py`, and `scripts/run_master_ingestion.py`.

Purpose:

- Acquire and combine source data for the configured period.
- Build monthly ingestion plans.
- Download BTS and Eurocontrol schedule data.
- Download OpenSky ADS-B sample archives.
- Save combined processed parquet files.

Primary outputs:

| Output | Meaning |
|---|---|
| `data/processed/adsb_combined.parquet` | Combined ADS-B state vectors. |
| `data/processed/bts_combined.parquet` | Combined BTS schedule/delay/cancellation data. |
| `data/processed/eurocontrol_combined.parquet` | Combined Eurocontrol schedule data. |
| `outputs/ingestion_control/monthly_ingestion_plan.csv` | Planned source-month units. |
| `outputs/ingestion_control/monthly_ingestion_status.csv` | Ingestion execution status. |
| `outputs/ingestion_control/logs/master_ingestion_performance.json` | Ingestion performance summary. |

Dissertation placement:

- Chapter 3 Methodology: data acquisition and storage design.
- Chapter 5 Critical Discussion: download time, network bottlenecks, and source compatibility.

#### Stage: features

Implemented in `main.py::stage_features` and `src/schedule_aware_features.py`.

Current default behaviour:

- Checks `features.mode`.
- If mode is `schedule_aware_direct`, it does not build full point-level `trajectories.parquet`.
- It loads BTS schedule windows.
- It partitions ADS-B by scheduled callsign.
- It extracts features from ADS-B pings inside schedule windows.
- It saves compact route sketches if enabled.

Primary outputs:

| Output | Rows | Meaning |
|---|---:|---|
| `data/processed/trajectory_features.parquet` | 400,222 | Flight-level feature vectors before final supervised filtering. |
| `data/processed/trajectory_sketches.parquet` | 6,462,924 | Downsampled route geometry for route maps, altitude charts, and optional future features. |

Important distinction:

- The system still has full trajectory reconstruction code in `src/trajectory_builder.py`.
- That path is legacy/diagnostic in the current default configuration.
- The current training path uses schedule-aware direct features, not a full `trajectories.parquet` file.

Dissertation placement:

- Chapter 3 Methodology: feature extraction pipeline.
- Chapter 5 Critical Discussion: why the design changed after runtime bottlenecks.

#### Stage: merge

Implemented in `main.py::stage_merge` and `src/merge.py`.

Current behaviour:

- Loads `trajectory_features.parquet`.
- Uses `merge.schedule_sources` from config.
- Current config uses BTS only.
- Matches ADS-B-derived features to schedule rows using callsign/time tolerance logic.
- Deduplicates by trajectory and flight key.
- Saves `data/processed/ml_dataset_merged.parquet`.

Verified merge report from `logs/matching_regression_report.json`:

| Metric | Value |
|---|---:|
| ADS-B feature input rows | 400,222 |
| Candidate rows | 2,228,928 |
| Time-filtered rows | 786,692 |
| Final merged rows | 393,339 |
| Retention percentage | 98.28% |
| Duplicates resolved | 393,353 |
| BTS baseline in scope | 439,449 |
| Matched rows | 393,339 |
| Recovery percent | 89.51% |

Dissertation placement:

- Chapter 3 Methodology: record linkage and matching.
- Chapter 4 Results: merge recovery and final dataset size.

#### Stage: weather

Implemented in `main.py::stage_weather`, `src/weather.py`, and `src/feature_enrichment.py`.

Current behaviour:

- Loads `ml_dataset_merged.parquet`.
- Loads `data/raw/metar.parquet` if available.
- Adds origin airport weather.
- Adds destination airport weather.
- Adds weather severity fields.
- Adds temporal, airport, congestion, and route distance enrichments.
- Saves `data/processed/ml_dataset_weather.parquet`.

What is implemented but not active:

- En-route weather from route sketches exists in `src/enroute_weather.py`.
- It is disabled by `enroute_weather.enabled: false`.
- Therefore, final dissertation results should describe weather as airport METAR weather unless the en-route path is explicitly enabled and rerun.

Dissertation placement:

- Chapter 3 Methodology: airport weather feature integration.
- Chapter 5 Critical Discussion: limitation of airport-only weather.

#### Stage: label

Implemented in `main.py::stage_label` and `src/labeling.py`.

Current behaviour:

- Computes `delay_minutes` using BTS `DepDelay` when available, otherwise timestamp differences.
- Assigns original labels:
  - `Cancelled` if cancellation flag is set.
  - `Late` if delay is greater than 15 minutes.
  - `Normal` otherwise.
- Creates `label_original`.
- Creates binary `label_binary`:
  - `Late` and `Cancelled` become `Disrupted`.
  - `Normal` remains `Normal`.
- Creates `disruption_subtype` to preserve whether a disrupted flight was late or cancelled.
- Saves label distribution to `logs/label_distribution.json`.

Important leakage control:

- `src/labeling.py` removes `actual_dep` and `actual_arr` before the final dataset is standardised.
- `src/schemas.py` excludes `label_original`, `label_binary`, and `disruption_subtype` from training features.
- `delay_minutes` remains in the final dataset for reporting but is dropped by model preparation because it is target-derived.

Dissertation placement:

- Chapter 3 Methodology: target construction.
- Chapter 5 Critical Discussion: cancellation sparsity and binary target decision.

#### Stage: validate

Implemented in `main.py::stage_validate`, `src/quality_gates.py`, and `src/data_validator.py`.

Current output from `logs/quality_gates_report.json`:

| Gate | Result | Evidence |
|---|---|---|
| Missingness | Failed | Missingness above 30% for `route_gc_distance_km` and destination weather fields. |
| Duplicates | Passed | Duplicate count is 0 using `flight_key`. |
| Label balance | Failed | `Cancelled` is only 0.14%. |
| Overall | Failed | Because missingness and label balance failed. |
| Training eligible rows | 393,269 | 99.98% of rows are training eligible. |

Interpretation:

- The dataset is usable for binary disruption modelling, but the quality gate output should be honestly discussed.
- The failed label-balance gate supports the decision not to make reliable cancellation-specific claims.

Dissertation placement:

- Chapter 4 Results: data quality results.
- Chapter 5 Critical Discussion: class imbalance and missingness limitations.

#### Stage: train

Implemented in `main.py::stage_train` and `src/model_training.py`.

Current behaviour:

- Loads `data/processed/ml_dataset.parquet`.
- Selects target based on `training.target_mode`.
- Uses `label_binary` for `binary_disrupted` mode.
- Drops identifiers, target columns, known leakage columns, and configured excluded features.
- Imputes missing values.
- Scales features for models that need it.
- Trains logistic regression, random forest, and XGBoost.
- Saves models and preprocessors.
- Saves metrics and figures.
- Generates SHAP explanations.

Current training feature list:

| Feature group | Features |
|---|---|
| Trajectory shape/behaviour | `flight_duration`, `trajectory_length`, `mean_speed`, `max_speed`, `speed_std`, `mean_altitude`, `altitude_variance`, `vertical_rate_std`, `heading_variability`, `holding_pattern_count`, `altitude_change_count`, `unstable_descent_flag`, `is_full_flight` |
| Trajectory quality/completeness | `trajectory_quality_score`, `dep_anchor_confidence`, `arr_anchor_confidence`, `middle_gap_count`, `gap_fraction_of_flight`, `max_inter_ping_seconds`, `ping_interval_cv` |
| Temporal | `dep_hour`, `dep_day_of_week`, `dep_month`, `is_peak_hour` |
| Airport/congestion | `origin_flight_count`, `dest_flight_count`, `origin_flights_1hr`, `dest_flights_1hr` |
| Route distance | `route_gc_distance_km` |
| Origin weather | `wind_speed`, `visibility`, `temperature`, `precipitation`, `weather_severity` |
| Destination weather | `wind_speed_dest`, `visibility_dest`, `temperature_dest`, `precipitation_dest`, `weather_severity_dest` |

Saved model files:

| File | Meaning |
|---|---|
| `models/logistic_regression.pkl` | Trained logistic regression model. |
| `models/random_forest.pkl` | Trained random forest model. |
| `models/xgboost.pkl` | Trained XGBoost model. |
| `models/imputer.pkl` | Saved imputation transformer. |
| `models/scaler.pkl` | Saved scaler. |
| `models/label_encoder.pkl` | Saved label encoder. |
| `models/feature_list.json` | Ordered feature list used by UI and inference. |
| `models/best_params_xgboost.json` | Best saved XGBoost hyperparameters. |
| `models/model_comparison.json` | Model ranking and metrics. |
| `models/feature_importance.json` | Feature importance values. |

Saved XGBoost parameters:

| Parameter | Value |
|---|---:|
| `max_depth` | 7 |
| `learning_rate` | 0.2609 |
| `n_estimators` | 405 |
| `subsample` | 0.7730 |
| `colsample_bytree` | 0.5106 |
| `min_child_weight` | 5 |
| `tree_method` | `hist` |
| `device` | `cuda` |

Dissertation placement:

- Chapter 3 Methodology: model design and training procedure.
- Chapter 4 Results: model comparison and feature importance.

### 2.8 Evaluation metrics

The code uses the following metrics in `src/model_training.py`:

| Metric | Meaning | Use in this project |
|---|---|---|
| Accuracy | Fraction of correct predictions. | Useful but can be misleading with imbalance. |
| Weighted F1 | F1 weighted by class support. | Captures overall performance while accounting for class frequency. |
| Macro F1 | Average F1 across classes equally. | Better for minority-class sensitivity than weighted F1. |
| Cohen's kappa | Agreement beyond chance. | Useful for imbalanced classification. |
| Per-class recall | Share of each actual class correctly found. | Important for disrupted recall. |
| Per-class precision | Share of predicted class that is correct. | Important for false alarm cost. |
| Confusion matrix | Actual vs predicted counts. | Used for class-level error diagnosis. |
| ROC curves | Ranking/separation quality. | Saved as `outputs/roc_curves.png`. |
| PR-AUC | Precision-recall area under curve. | Populated in the corrected final binary run. |

Important issue resolved:

- Older ROC/PR-AUC logic was unsafe because it relied on a generic class index rather than the intended positive class.
- The corrected final outputs explicitly use `Disrupted` as the positive class.
- The regenerated binary ROC curve reports AUC values of 0.6930 for logistic regression, 0.8383 for random forest, and 0.8547 for XGBoost.

### 2.9 Current model outputs and interpretation

Current saved `models/model_comparison.json`:

| Model | Accuracy | Weighted F1 | Macro F1 | Disrupted recall | Disrupted precision | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| XGBoost | 0.7964 | 0.8047 | 0.7406 | 0.7118 | 0.5496 | Best saved balance between detecting disruptions and maintaining usable precision. |
| Random forest | 0.8189 | 0.7818 | 0.6552 | 0.2781 | 0.8390 | Conservative: predicts fewer disruptions, so precision is high but many disruptions are missed. |
| Logistic regression | 0.6530 | 0.6779 | 0.6018 | 0.6297 | 0.3610 | Aggressive: catches more disruptions than random forest but produces many false positives. |

What this means for an end-user product:

- If the goal is a balanced risk estimate, XGBoost is the most defensible model from the current artifacts.
- If the goal is to avoid missing disruptions at almost any cost, lower thresholds or stronger weights can raise recall, but precision and trustworthiness drop.
- If the goal is to show only high-confidence disruption alerts, random forest behaviour may be useful, but it misses too many disrupted flights for broad risk monitoring.

### 2.10 Experiments performed

#### Experiment 1: Initial full trajectory reconstruction

Evidence:

- Legacy path remains in `src/trajectory_builder.py`.
- `main.py::stage_features` still supports `trajectory_reconstruction` mode.
- Notebook `02_trajectory_reconstruction.ipynb` includes optional full reconstruction.

Purpose:

- Build complete flight trajectories from ADS-B pings.
- Segment aircraft traces into flight-level paths.
- Compute trajectory-level features.

Outcome:

- Technically implemented but not current default.
- Runtime and disk behaviour made it impractical for six months of data on available hardware.
- The project moved to schedule-aware direct extraction.

Dissertation use:

- Chapter 5 Critical Discussion: explain the abandoned/modified approach and why it changed.

#### Experiment 2: Schedule-aware direct extraction

Evidence:

- `features.mode: schedule_aware_direct` in config.
- `src/schedule_aware_features.py`.
- `data/processed/trajectory_features.parquet` and `data/processed/trajectory_sketches.parquet`.

Purpose:

- Reduce unnecessary ADS-B compute by filtering by scheduled callsigns and time windows before feature extraction.

Outcome:

- Produced 400,222 feature rows and 6,462,924 sketch points from 1.296B raw ADS-B pings.
- Became current default feature pipeline.

Dissertation use:

- Chapter 3 Methodology: final feature extraction design.
- Chapter 5 Critical Discussion: optimisation and practical engineering constraints.

#### Experiment 3: EDA signal audit

Evidence:

- `scripts/eda_signal_audit.py`.
- `outputs/eda_signal_audit/summary.json`.
- `outputs/eda_signal_audit/feature_group_model_results.csv`.

Verified summary:

| Result | Value |
|---|---:|
| Rows used | 83,289 |
| Disrupted rate | 0.2143 |
| Best feature group | `combined_available` |
| Best ROC-AUC | 0.6901 |

Feature group findings:

| Feature group | ROC-AUC | Average precision | Disrupted recall | Interpretation |
|---|---:|---:|---:|---|
| Combined available | 0.6901 | 0.3357 | 0.5991 | Best signal comes from combining feature groups. |
| Temporal derived | 0.6753 | 0.3140 | 0.6221 | Time features are strong. |
| Congestion derived | 0.6290 | 0.2807 | 0.5366 | Congestion has useful signal. |
| Airport/route categorical | 0.6080 | 0.2506 | 0.5894 | Route/airport context helps. |
| Trajectory existing | 0.5735 | 0.2333 | 0.3592 | Trajectory features alone are weaker but still above random. |
| Weather existing | 0.5412 | 0.1930 | 0.2910 | Airport weather alone is weak. |

Dissertation use:

- Chapter 4 Results: feature group signal audit.
- Chapter 5 Discussion: why combined features were used and why weather/trajectory alone are insufficient.

#### Experiment 4: Binary target merge

Evidence:

- `src/labeling.py` creates `label_binary` and `disruption_subtype`.
- `src/model_training.py` uses `training.target_mode`.
- Current config uses `binary_disrupted`.

Purpose:

- Avoid unreliable cancellation-specific prediction because cancellations are extremely rare.
- Preserve `Late` and `Cancelled` as subtypes while training the model to predict general disruption risk.

Outcome:

- Binary target has 85,638 disrupted flights instead of only 535 cancelled flights.
- This is more statistically defensible than trying to model cancellations separately.

Important wording:

- The purpose of merging is target stability and class sparsity reduction.
- It should not be described as being done simply to increase recall.
- Recall is controlled by model, class/sample weighting, threshold, and business decision policy.

#### Experiment 5: Subtype sample weights

Evidence:

- `training.subtype_sample_weights` in config.
- `src/model_training.py` composes balanced sample weights with subtype multipliers.
- Snapshot folders: `weighted_1_1_1_run`, `weighted_1_2_5_run`, `weighted_1_2_10_run`.
- `outputs/threshold_experiments/weighted_run_comparison.csv`.

Comparison:

| Weights | Model | Accuracy | Disrupted recall | Disrupted precision | Weighted F1 | Macro F1 | Cohen's kappa |
|---|---|---:|---:|---:|---:|---:|---:|
| 1/1/1 | XGBoost | 0.8057 | 0.7072 | 0.5676 | 0.8125 | 0.7490 | 0.5001 |
| 1/2/5 | XGBoost | 0.7898 | 0.7230 | 0.5373 | 0.7995 | 0.7359 | 0.4760 |
| 1/2/10 | XGBoost | 0.7693 | 0.7444 | 0.5043 | 0.7824 | 0.7195 | 0.4473 |
| 1/1/1 | Random forest | 0.8189 | 0.2781 | 0.8390 | 0.7818 | 0.6552 | 0.3411 |
| 1/2/10 | Random forest | 0.8159 | 0.2604 | 0.8440 | 0.7761 | 0.6447 | 0.3234 |
| 1/1/1 | Logistic regression | 0.6530 | 0.6297 | 0.3610 | 0.6779 | 0.6018 | 0.2303 |
| 1/2/10 | Logistic regression | 0.3888 | 0.9145 | 0.2655 | 0.3753 | 0.3879 | 0.0773 |

Interpretation:

- Weighting late/cancelled subtypes increases XGBoost disrupted recall from 0.7072 to 0.7444.
- The same change reduces XGBoost accuracy, disrupted precision, weighted F1, macro F1, and kappa.
- The best balanced XGBoost run is `1/1/1` based on macro F1 and kappa.
- The highest XGBoost disrupted recall is `1/2/10`.
- The final choice should be framed as a business tradeoff, not a universally better model.

#### Experiment 6: Threshold and hyperparameter experiments

Evidence:

- `notebooks/10_threshold_hyperparameter_experiments.ipynb`.
- `outputs/threshold_experiments/threshold_hyperparameter_experiments.csv`.
- `outputs/threshold_experiments/metrics_vs_threshold.png`.
- `outputs/threshold_experiments/accuracy_recall_prauc_combined.png`.
- Heatmaps for logistic regression, random forest, and XGBoost.

Purpose:

- Explore the tradeoff between accuracy and disrupted recall.
- Compare threshold choices for different business scenarios.
- Test hyperparameter settings for faster models and XGBoost.

Important interpretation:

- Accuracy and recall crossing on a threshold plot does not mean the model is optimal by itself.
- The crossing point is only a visual tradeoff point where overall correctness and disrupted capture rate are similar.
- The final threshold should depend on the cost of false positives vs missed disruptions.

#### Experiment 7: UI prediction explorer

Evidence:

- `pages/6_Predictions_Explorer.py`.
- `standalone/predict_app.py`.
- `app.py` and `predict_app.py` entrypoints.

Current behaviour:

- Loads model artifacts and the final dataset.
- Provides route-first selection using origin-destination pairs.
- Allows selection of a specific historical flight on that route.
- Runs inference using saved feature list, imputer, scaler, label encoder, and model.
- Shows class probabilities.

Important limitation:

- The displayed probabilities are raw model probabilities, not calibrated probabilities.
- The selected flights are existing rows from the historical dataset, not future flight forecasts unless future schedule and feature inputs are provided.

### 2.11 Figures, tables, and output files

Important generated outputs for dissertation use:

| Artifact | Path | Meaning | Suggested dissertation use |
|---|---|---|---|
| Dataset sizes table | `outputs/pipeline_report/dataset_sizes_table.csv` | Size of major datasets. | Chapter 4 data overview. |
| Dataset inventory | `outputs/pipeline_report/dataset_inventory.csv` | Data artifact inventory. | Appendix or Chapter 4. |
| Label distribution | `outputs/pipeline_report/label_distribution.png` and `.csv` | Class balance. | Chapter 4 target distribution. |
| Monthly final counts | `outputs/pipeline_report/monthly_final_flight_counts.png` and `.csv` | Month coverage of final data. | Chapter 4 data coverage. |
| Missingness plot | `outputs/pipeline_report/ml_dataset_missingness.png` | Missing columns. | Chapter 5 limitations. |
| Weather coverage | `outputs/pipeline_report/weather_coverage.png` and `.csv` | Weather availability. | Chapter 4/5. |
| Model ranking | `outputs/pipeline_report/model_ranking.png` and `.csv` | Model comparison. | Chapter 4 model results. |
| Confusion matrices | `outputs/confusion_matrix_*.png` | Class-level errors. | Chapter 4 evaluation. |
| ROC curves | `outputs/roc_curves.png` | Model discrimination. | Chapter 4 evaluation. |
| SHAP summary | `outputs/shap_summary.png` | Feature explanations. | Chapter 4 explainability. |
| Feature importance by category | `outputs/feature_importance_by_category.png` | Grouped importance. | Chapter 4 feature analysis. |
| Top feature importance | `outputs/pipeline_report/top_feature_importance.png` and `.csv` | Ranked features. | Chapter 4 feature interpretation. |
| Trajectory route map | `outputs/trajectory_previews/...route_map.html` | Example route geometry. | Chapter 3/4 visual explanation. |
| Altitude profile | `outputs/trajectory_previews/...altitude_profile.png` | Example altitude trace. | Chapter 3 feature illustration. |
| Threshold experiment results | `outputs/threshold_experiments/threshold_hyperparameter_experiments.csv` | Threshold/hyperparameter metrics. | Chapter 4 experiments. |
| Weighted-run comparison | `outputs/threshold_experiments/weighted_run_comparison.csv` | Weighting scheme comparison. | Chapter 4/5 tradeoff analysis. |

The `important outputs/` folder contains a curated copy of many report-relevant figures, tables, logs, and threshold experiment outputs. It is useful for dissertation assembly but should not be treated as a separate source of truth from the original outputs.

### 2.12 Fully implemented, partially implemented, and missing parts

#### Fully implemented in code and evidenced by outputs

| Component | Evidence |
|---|---|
| Six-month ADS-B/BTS/Eurocontrol ingestion outputs | Processed parquet files exist. |
| Schedule-aware direct ADS-B feature extraction | `trajectory_features.parquet` and `trajectory_sketches.parquet`. |
| BTS merge into final labelled ML dataset | `matching_regression_report.json`, `ml_dataset.parquet`. |
| Airport METAR weather integration | `data/raw/metar.parquet`, weather features in feature list. |
| Original and binary label creation | `label_distribution.json`, `label_binary` in dataset. |
| Quality gates | `quality_gates_report.json`. |
| Logistic regression, random forest, and XGBoost training | Saved model files and `model_comparison.json`. |
| SHAP and feature importance outputs | `outputs/shap_summary.png`, `feature_importance.json`. |
| Threshold and weight experiments | Notebook 10 and threshold experiment outputs. |
| Streamlit prediction explorer | `pages/6_Predictions_Explorer.py`, `standalone/predict_app.py`. |

#### Partially implemented or optional

| Component | Current state | Risk |
|---|---|---|
| Full point-level trajectory reconstruction | Code exists but is not current default. | Do not present it as the final training path. |
| Route deviation features | Code exists and schemas include columns, but current trained feature list does not include deviation columns. | Do not claim model gains from route deviation. |
| En-route weather | Code exists but config disables it. | Do not claim en-route weather was used. |
| Drift detection | `src/drift_detector.py` and `stage_drift` exist. | Only discuss if outputs exist for the run being written about. |
| PR-AUC reporting | Fixed for the final binary run by using `Disrupted` as the explicit positive class. | Use only regenerated final PR-AUC/ROC artifacts. |
| UI future prediction | Historical flight selection works. | True future prediction requires future schedules and feature availability. |

#### Missing or not yet implemented

| Missing item | Why it matters | Recommended action |
|---|---|---|
| Calibrated probabilities | UI currently shows raw probabilities. | Add calibration or explicitly label as raw model probability. |
| Reliable cancellation-specific prediction | Cancelled class has only 535 final rows. | Avoid separate cancellation model unless more cancellation data is obtained. |
| Final run provenance | Current active config and model comparison now agree on `1/1/1`; older snapshots still exist. | Label the 2026-04-28 run as final and treat `weighted_*_run` folders as experimental comparisons. |
| Older binary ROC/PR-AUC artifacts | Some older artifacts used the wrong class framing. | Use the corrected final evidence bundle only. |
| Lagged route cancellation/delay history | Route history can leak if computed using full dataset. | Implement time-aware lagged features only. |
| External validation period | Current results are from Jan-Jun 2022. | Add later holdout if available. |

### 2.13 Mismatch between code and possible dissertation claims

| Claim | Supported by current evidence? | Correct wording |
|---|---|---|
| The final model predicts `Normal`, `Late`, and `Cancelled`. | No, current training target is binary. | The final model predicts `Normal` vs `Disrupted`; original subtypes are retained for analysis. |
| The model can reliably predict cancellations. | No. Cancelled has 535 rows and prior 3-class results were weak for cancellation. | Cancellation-specific prediction is not reliable with current data; cancellations are folded into disruption risk. |
| Eurocontrol contributes to final training labels. | No, current final merged rows are BTS only. | Eurocontrol was downloaded and normalised but is not part of the current final supervised training set. |
| Route deviation features drive the final model. | No, current trained feature list does not include deviation columns. | Route deviation code exists but is not evidenced as used in the current trained model. |
| En-route weather was used. | No, it is disabled. | Airport METAR weather was used; en-route weather is a future/optional extension. |
| PR-AUC was unavailable because classes were merged. | No. | PR-AUC was not populated in the saved artifact due to evaluation/logging implementation. |
| UI gives calibrated probability of disruption. | No. | UI gives raw model probabilities. |
| The route-level cancellation risk rises exponentially from previous cancellations. | No implemented evidence. | Route/congestion features exist, but no exponential route cancellation feature is implemented. |

### 2.14 Runtime and run metadata evidence

Selected run metadata from `logs/`:

| Stage/run | Start | End | Duration | Rows/models |
|---|---|---|---:|---|
| Label | 2026-04-25 12:13:13 | 2026-04-25 12:14:04 | 50.7 seconds | 393,339 rows |
| Validate | 2026-04-25 12:14:05 | 2026-04-25 12:14:08 | 2.9 seconds | 393,339 rows |
| Train baseline run | 2026-04-25 12:14:09 | 2026-04-25 13:55:36 | 6,087.6 seconds | 3 models |
| Train weighted `1/1/1` snapshot | 2026-04-26 17:27:14 | 2026-04-26 17:35:26 | 491.6 seconds | 3 models |
| Train weighted `1/2/5` snapshot | 2026-04-26 18:05:44 | 2026-04-26 18:13:11 | 447.6 seconds | 3 models |
| Train weighted `1/2/10` snapshot | 2026-04-26 18:17:46 | 2026-04-26 18:23:24 | 337.8 seconds | 3 models |
| Final active `1/1/1` train run after ROC/PR-AUC fix | 2026-04-28 17:41:37 | 2026-04-28 17:48:09 | 392.2 seconds | 3 models |

Interpretation:

- Later training runs were much faster than the earlier full training run, likely because expensive explainability or cached/intermediate work was reduced or already available.
- For dissertation claims, use run metadata as evidence but avoid claiming a single universal runtime because hardware, GPU availability, and cached artifacts affect duration.

## 3. Part 2: Development Story and Project Evolution

### 3.1 Initial approach

The initial technical direction was to reconstruct aircraft trajectories from ADS-B state vectors and then use trajectory behaviour to predict disruption. This was a reasonable starting point because ADS-B contains direct operational signals such as position, altitude, speed, heading, vertical rate, and timing gaps. These signals can describe flight progress and abnormal behaviour better than schedule data alone.

Implemented evidence:

- `src/trajectory_builder.py` contains full trajectory segmentation and quality-aware reconstruction logic.
- `src/feature_engineering.py` computes trajectory-derived features.
- `02_trajectory_reconstruction.ipynb` documents the trace strategy and retains optional full reconstruction.

Dissertation use:

- Chapter 3: explain the original hypothesis that movement behaviour contains disruption signal.
- Chapter 5: reflect on why the original full reconstruction was technically expensive.

### 3.2 Problems and blockers

The full reconstruction approach faced practical bottlenecks:

| Blocker | Evidence/impact |
|---|---|
| Very large ADS-B input | Current combined ADS-B table has 1.296B rows and about 44 GB on disk. |
| Long local runtime | Full trajectory reconstruction was observed to run for many hours without producing useful final output. |
| Low CPU utilisation at points | The workload included I/O, parquet writes, grouping, and Python overhead, not only pure CPU work. |
| Large intermediate files | The old approach risked producing huge point-level trajectory outputs. |
| Colab limitations | Colab had fewer CPUs and memory limits; Drive I/O was slower than local SSD. |

This led to a design change: filter the ADS-B data using schedule knowledge before doing trajectory-style feature extraction.

Dissertation use:

- Chapter 5 Critical Discussion: practical constraints and engineering tradeoffs.

### 3.3 Move to schedule-aware direct features

The key change was to stop treating all ADS-B points as equally relevant. Instead, the system first uses schedule rows to identify callsigns and time windows, then processes only ADS-B pings that could correspond to those flights.

| Previous design | Current design |
|---|---|
| Reconstruct all aircraft trajectories first. | Start from scheduled flights and filter ADS-B by callsign/time. |
| Write large point-level trajectory intermediates. | Save compact one-row-per-flight features. |
| Route geometry could be lost if only features are saved. | Save separate coarse `trajectory_sketches.parquet`. |
| Expensive for six months of data. | More practical on local hardware. |

Implemented evidence:

- `features.mode: schedule_aware_direct`.
- `src/schedule_aware_features.py`.
- `trajectory_features.parquet` and `trajectory_sketches.parquet`.

Dissertation use:

- Chapter 3: final methodology.
- Chapter 5: justified methodological adaptation.

### 3.4 Keeping route geometry without excessive compute

A concern during development was that computing only aggregate feature vectors would throw away route geometry. The current compromise is route sketches:

- Feature extraction can downsample pings for computation.
- Route sketches preserve coarse point-level geometry at a lower frequency.
- Sketches are used for maps and altitude charts.
- Sketches can support route-deviation and en-route weather features later.

Implemented evidence:

- `trajectory_sketches.parquet` has 6,462,924 sketch points.
- Config includes `sketch_interval_seconds: 600`.
- `scripts/generate_trajectory_preview.py` and `src/trajectory_preview.py` support route map and altitude profile generation.

Dissertation use:

- Chapter 3: explain how route information is retained.
- Chapter 4: include route map and altitude chart as explanatory visuals.

### 3.5 Binary target decision

The original labels include `Normal`, `Late`, and `Cancelled`, but cancellations are very rare in the final matched dataset. The current final target therefore merges `Late` and `Cancelled` into `Disrupted`.

Reasoning:

- `Cancelled` has only 535 final rows, or 0.14%.
- A three-class cancellation predictor would be unreliable with so few examples.
- A binary disruption target is more defensible for an operational risk prototype.
- The original subtype is still preserved for analysis and sample weighting.

Important critical reflection:

- The binary merge should be described as a response to class sparsity.
- It should not be described as automatically improving recall.
- Recall is controlled by model training, thresholds, class/sample weighting, and business decision policy.

Dissertation use:

- Chapter 3: target engineering.
- Chapter 5: limitation of cancellation prediction.

### 3.6 Weighting and threshold experiments

After the binary target was created, experiments were run to examine whether the model should treat late and cancelled subtypes differently inside the disrupted class.

Three preserved run snapshots exist:

| Snapshot | Meaning |
|---|---|
| `weighted_1_1_1_run` | No subtype multiplier beyond class balancing. |
| `weighted_1_2_5_run` | Late receives 2x subtype multiplier, cancelled receives 5x. |
| `weighted_1_2_10_run` | Late receives 2x subtype multiplier, cancelled receives 10x. |

Main finding:

- XGBoost recall improves as cancelled/late weighting increases.
- XGBoost precision and overall metrics decline as weighting increases.
- The best choice depends on business cost:
  - choose `1/1/1` for balanced validation performance.
  - choose `1/2/10` if missed disruptions are more costly than false alarms.

Threshold experiments then explored similar tradeoffs at the decision threshold level. This is important because a company use case may prefer higher recall even if accuracy is lower, while an end-user product may prefer fewer false alarms.

Dissertation use:

- Chapter 4: show comparison table and threshold plots.
- Chapter 5: discuss business-dependent model selection.

### 3.7 UI simplification

The Streamlit prediction page initially had stability and memory problems because large parquet files and richer UI elements made repeated predictions unreliable. The current direction is a simpler flow:

1. Select a route such as origin-to-destination.
2. Select a historical flight from that route.
3. Predict using the saved model pipeline.
4. Show the class probability table.
5. Allow the user to choose another flight.

Implemented evidence:

- `pages/6_Predictions_Explorer.py`.
- `standalone/predict_app.py`.
- Route-first selection logic and cached artifact loading.

Remaining limitation:

- The UI is still a prototype over historical rows.
- It is not yet a full future-flight forecasting product.

Dissertation use:

- Chapter 3: artefact/prototype design.
- Chapter 5: usability and deployment limitations.

### 3.8 Key milestones

| Milestone | Evidence |
|---|---|
| Basic ingestion and raw source combination | `src/data_ingestion.py`, processed source parquet files. |
| Full trajectory reconstruction attempt | `src/trajectory_builder.py`, `02_trajectory_reconstruction.ipynb`. |
| Move to schedule-aware direct pipeline | `src/schedule_aware_features.py`, config feature mode. |
| Route sketches retained | `trajectory_sketches.parquet`, preview outputs. |
| BTS final supervised dataset built | `matching_regression_report.json`, `ml_dataset.parquet`. |
| Binary target implemented | `src/labeling.py`, `label_binary`, `disruption_subtype`. |
| Model training and explainability | `models/`, `outputs/shap_summary.png`, `outputs/model_comparison.png`. |
| Threshold and weighting experiments | `notebooks/10_threshold_hyperparameter_experiments.ipynb`, weighted snapshots. |
| Streamlit UI stabilised | `pages/6_Predictions_Explorer.py`, `standalone/predict_app.py`. |

### 3.9 Remaining limitations

| Limitation | Impact | Dissertation framing |
|---|---|---|
| Cancelled class is tiny | No reliable cancellation-specific model. | Critical limitation and justification for binary target. |
| Current final training uses BTS only | Geographic/source scope is narrower than all downloaded data. | Be precise about final supervised source. |
| Weather is airport-level | May miss en-route weather disruption mechanisms. | Future work. |
| PR-AUC missing in saved comparison | Results table incomplete. | Fix or omit PR-AUC from final reported model table. |
| Earlier config/model artifact mismatch | Previously confused final result interpretation. | Resolved by rerunning training on 2026-04-28 with `1/1/1` weights; keep older weighted folders as experiment snapshots. |
| UI uses historical rows | Not true future prediction yet. | Prototype, not production deployment. |
| Some missingness remains high | Destination weather and route distance missingness affect quality gate. | Data quality limitation. |

## 4. Dissertation mapping table

| Dissertation section | Use this project evidence | Suggested content |
|---|---|---|
| Chapter 1 Introduction | Project aim, aviation disruption context, need for predictive support. | Introduce disruption prediction as a data-driven decision-support problem. |
| Chapter 2 Literature Review | Not directly generated by code. | Compare with literature on delay prediction, ADS-B, weather, class imbalance, and explainability. |
| Chapter 3 Methodology: Data | `configs/config.yaml`, source parquet counts, `src/data_ingestion.py`, `scripts/run_master_ingestion.py`. | Explain OpenSky, BTS, Eurocontrol, METAR, six-month period, storage layout. |
| Chapter 3 Methodology: Preprocessing | `src/normalization.py`, `src/schedule_aware_features.py`, `src/merge.py`, `src/weather.py`. | Explain source normalisation, schedule windows, ADS-B filtering, merge, weather integration. |
| Chapter 3 Methodology: Feature Engineering | `src/feature_engineering.py`, `src/feature_enrichment.py`, feature list. | Explain trajectory, temporal, congestion, weather, route-distance, and quality features. |
| Chapter 3 Methodology: Target | `src/labeling.py`, label counts. | Explain `Normal`, `Late`, `Cancelled`, and binary `Disrupted` target. |
| Chapter 3 Methodology: Modelling | `src/model_training.py`, `models/best_params_xgboost.json`. | Explain train/test split, imputation, scaling, model families, XGBoost tuning, sample weights. |
| Chapter 3 Methodology: Artefact | Streamlit files. | Explain prediction explorer as a prototype interface. |
| Chapter 4 Results: Data | Dataset tables and monthly coverage figures. | Present row counts, class distribution, merge recovery, data quality. |
| Chapter 4 Results: Models | `model_comparison.json`, confusion matrices, ROC curves, SHAP, feature importance. | Compare model performance and explain feature influence. |
| Chapter 4 Results: Experiments | Threshold and weighted-run outputs. | Present accuracy/recall/precision tradeoffs. |
| Chapter 5 Critical Discussion | Quality gates, missingness, class imbalance, runtime evolution, partial features. | Critically evaluate limitations and design decisions. |
| Chapter 5 Future Work | Route deviation, en-route weather, calibration, future-flight inference, lagged route history. | Describe improvements that are not yet fully implemented. |
| Appendix | Important output folder, notebook reports, logs, config. | Include supporting tables and generated figures. |

## 5. Missing features and improvement plan

### 5.1 Must fix before final submission

| Priority | Issue | Why it matters | Action |
|---|---|---|---|
| High | Final-run provenance | Final results need clear provenance. | State that the final active run is the 2026-04-28 `1/1/1` training run; cite weighted folders only as experiments. |
| High | ROC/PR-AUC figure provenance | Older ROC figures can be misleading. | Use only the corrected ROC figure labelled `Positive class: Disrupted`. |
| High | Claims about cancellation prediction | Current data cannot support reliable cancellation-specific prediction. | State binary disruption prediction only. |
| High | Eurocontrol final role | Downloaded but not used in current final supervised dataset. | Word dissertation carefully or rerun with verified Eurocontrol labels. |
| Medium | Route deviation missing from trained feature list | Route-deviation claims would be unsupported. | Either integrate and rerun or present as future work. |
| Medium | En-route weather disabled | En-route weather claims would be unsupported. | Either enable and rerun or present as future work. |
| Medium | Raw probabilities in UI | Raw probabilities may be misinterpreted. | Label as raw or add calibration. |

### 5.2 Recommended improvements after dissertation core is stable

| Improvement | Benefit |
|---|---|
| Probability calibration | Makes UI risk percentages more defensible. |
| Time-based holdout with later months | Stronger real-world validation. |
| Lagged route/airport disruption history | Adds operational context without target leakage if implemented correctly. |
| En-route weather from trajectory sketches | Tests whether weather along the actual path improves prediction. |
| Route deviation from historical corridor baselines | Could capture flights that are unusually far from normal route geometry. |
| Better PR and calibration plots | Stronger Chapter 4 evaluation. |
| Lightweight inference table | More robust Streamlit experience and lower memory load. |

### 5.3 Recommended final model decision wording

Final selected setting:

- Use the regenerated 2026-04-28 `1/1/1` XGBoost model as the preferred final result because `1/1/1` was the best balanced weighting strategy in the preserved comparison and now matches the active config.

If discussing high disruption recall as an alternative:

- Use the `1/2/10` XGBoost result and explicitly state that it sacrifices precision and accuracy to catch more disrupted flights.

Avoid saying:

- "The weighting made the model better" without specifying the metric.

Use instead:

- "Increasing disruption subtype weights improved disrupted recall but reduced precision and overall agreement, so the preferred setting depends on the operational cost of missed disruptions versus false alarms."

## 6. Final checklist before writing Chapter 4 and Chapter 5

### Evidence checklist

| Check | Status from current evidence |
|---|---|
| Dataset row counts verified | Yes. |
| Label counts verified | Yes. |
| Final target mode identified | Yes: binary `Normal` vs `Disrupted`. |
| Current model ranking available | Yes. |
| Weighted-run comparisons available | Yes. |
| Threshold experiment outputs available | Yes. |
| Quality gate report available | Yes. |
| Feature list available | Yes: 39 features. |
| SHAP and feature importance outputs available | Yes. |
| UI prototype exists | Yes. |
| PR-AUC available in saved comparison | Yes, after the 2026-04-28 ROC/PR-AUC fix. |
| Config/model artifact fully aligned | Yes, current active config and model comparison now use `1/1/1` weights. |

### Chapter 4 writing checklist

- Use `ml_dataset.parquet` row count: 393,339.
- Use label distribution: 307,701 normal, 85,103 late, 535 cancelled.
- State binary target distribution: 307,701 normal and 85,638 disrupted.
- Present merge recovery from `matching_regression_report.json`.
- Present quality gate failures honestly.
- Present model comparison with accuracy, weighted F1, macro F1, disrupted recall, disrupted precision, and kappa.
- Do not present PR-AUC from `model_comparison.json` unless fixed and rerun.
- Use confusion matrices, ROC curves, SHAP summary, feature importance, missingness, and label distribution figures.
- Use weighted-run comparison to explain performance tradeoffs.

### Chapter 5 writing checklist

- Explain why full trajectory reconstruction was replaced.
- Discuss class imbalance and the cancellation limitation.
- Explain that binary disruption prediction is a pragmatic design decision.
- Discuss raw probability limitations and lack of calibration.
- Discuss airport-only weather and disabled en-route weather.
- Discuss why Eurocontrol is not part of final supervised training in the current run.
- Discuss route deviation as an implemented code path but not an evidenced final model feature.
- Discuss final model choice as a business tradeoff, not just highest accuracy.

### Final submission risk checklist

| Risk | Required action before final submission |
|---|---|
| Older weighted snapshots can be confused with the active final model. | Clearly label `weighted_*_run` folders as experiments and the 2026-04-28 `1/1/1` run as the active final run. |
| Older ROC curve used the wrong positive-class framing. | Use the regenerated ROC curve with `Disrupted` as the positive class. |
| Figures from old runs may show stale one-month or outdated data. | Use regenerated figures from current six-month outputs only. |
| Cancellation-specific claims unsupported. | Reword to disruption-risk claims. |
| Eurocontrol overclaim risk. | State it was acquired/normalised but not used in current final labelled training. |
| UI future prediction overclaim risk. | State UI predicts on existing/historical flight records unless future features are supplied. |

