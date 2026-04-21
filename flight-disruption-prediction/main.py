"""
Flight Disruption Prediction Pipeline — Main Orchestrator
T2: Fixed broken imports
T4: Wired DataValidator into pipeline boundaries
T8: Refactored with --stage CLI flag
T9: Resumable stage support with --force flag
T16: Strict column ordering on final export
T31: Drift detection stage
"""
import logging
import argparse
import json
import os
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime

from src.feature_engineering import FeatureExtractor
from src.schedule_aware_features import ScheduleAwareFeatureConfig, extract_schedule_aware_features
from src.trajectory_builder import build_notebook_context, reconstruct_full_dataset, reconstruct_full_dataset_parallel
from src.merge import DataMerger
from src.weather import WeatherIntegrator
from src.labeling import LabelGenerator
from src.data_ingestion import EurocontrolDownloader, BTSCombiner, EuroCombiner, ManifestManager
from src.utils import load_config, ensure_dir, set_seed
from src.config_validator import validate_environment
from src.data_validator import DataValidator
from src.schemas import (
    FEATURES_SCHEMA, CANONICAL_SCHEDULE_SCHEMA,
    ML_DATASET_SCHEMA, ML_DATASET_COLUMN_ORDER, ML_FEATURE_COLUMNS
)
from src.quality_gates import QualityGateRunner
from src.normalization import ScheduleNormalizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Stage output file mapping (T9) ──────────────────────────────────────
STAGE_OUTPUTS = {
    'ingest': 'data/processed/eurocontrol_combined.parquet',
    'features': 'data/processed/trajectory_features.parquet',
    'merge': 'data/processed/ml_dataset_merged.parquet',
    'weather': 'data/processed/ml_dataset_weather.parquet',
    'label': 'data/processed/ml_dataset_labeled.parquet',
    'validate': 'logs/quality_gates_report.json',
    'train': 'models/model_comparison.json',
    'drift': 'logs/drift_report.json',
}


def _resolve_weather_input(paths: dict) -> Path:
    """Resolve the configured weather input path with a sensible project-root fallback."""
    configured = Path(paths.get('metar_data_file', 'data/raw/metar_2025.parquet'))
    if configured.exists():
        return configured
    fallback = Path('data/raw/metar_2025.parquet')
    return fallback if fallback.exists() else configured


def _log_run_metadata(stage: str, start_time: datetime, status: str, 
                      row_count: int = 0, details: dict = None):
    """Write run metadata to logs/ (T8)."""
    log_dir = Path('logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    
    end_time = datetime.now()
    run_meta = {
        'stage': stage,
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'duration_seconds': (end_time - start_time).total_seconds(),
        'status': status,
        'row_count': row_count,
    }
    if details:
        run_meta.update(details)
    
    run_file = log_dir / f"run_{end_time.strftime('%Y%m%d_%H%M%S')}_{stage}.json"
    with open(run_file, 'w') as f:
        json.dump(run_meta, f, indent=4, default=str)
    logger.info(f"Run metadata saved to {run_file}")


def _latest_merge_qa_metrics(source_dataset: str) -> dict:
    qa_dir = Path('logs/merge_qa')
    candidates = sorted(qa_dir.glob(f'merge_qa_report_{source_dataset}_*.json'), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return {}
    with open(candidates[-1], 'r') as f:
        payload = json.load(f)
    return payload.get('metrics', {})


def _stage_schema(schema: dict, df: pd.DataFrame, required_cols: list[str] | None = None) -> dict:
    """Return a stage-appropriate schema subset to avoid false missing-column noise."""
    required_cols = required_cols or []
    keep_cols = set(df.columns).union(required_cols)
    return {col: rules for col, rules in schema.items() if col in keep_cols}


def _write_matching_regression_report(merged_df: pd.DataFrame, labeled_df: pd.DataFrame):
    """Persist a compact source-matching summary for regression tracking."""
    report_path = Path('logs/matching_regression_report.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    source_counts = {}
    if 'source_dataset' in merged_df.columns:
        source_counts = {
            str(k): int(v)
            for k, v in merged_df['source_dataset'].value_counts(dropna=False).to_dict().items()
        }

    label_distribution = {}
    if 'label' in labeled_df.columns:
        label_distribution = {
            str(k): int(v)
            for k, v in labeled_df['label'].value_counts(dropna=False).to_dict().items()
        }

    labels_by_source = {}
    if {'source_dataset', 'label'}.issubset(labeled_df.columns):
        labels_by_source = {
            str(source): {str(label): int(count) for label, count in counts.items()}
            for source, counts in labeled_df.groupby('source_dataset')['label'].value_counts(dropna=False).unstack(fill_value=0).to_dict('index').items()
        }

    bts_baseline_rows = None
    bts_baseline_late = None
    bts_baseline_cancelled = None
    feature_path = Path('data/processed/trajectory_features.parquet')
    bts_path = Path('data/processed/bts_combined.parquet')
    if feature_path.exists() and bts_path.exists():
        df_adsb_dates = pd.read_parquet(feature_path, columns=['timestamp'])
        adsb_days = (
            pd.to_datetime(df_adsb_dates['timestamp'], unit='s', utc=True, errors='coerce')
            .dt.normalize()
            .dropna()
            .drop_duplicates()
        )
        df_bts = pd.read_parquet(
            bts_path,
            columns=['service_day_utc', 'scheduled_dep_utc', 'scheduled_arr_utc', 'DepDelay', 'Cancelled'],
        )
        scheduled_dep_days = pd.to_datetime(df_bts['scheduled_dep_utc'], utc=True, errors='coerce').dt.normalize()
        scheduled_arr_days = pd.Series(pd.NaT, index=df_bts.index, dtype='datetime64[ns, UTC]')
        if 'scheduled_arr_utc' in df_bts.columns:
            scheduled_arr_days = pd.to_datetime(df_bts['scheduled_arr_utc'], utc=True, errors='coerce').dt.normalize()
        else:
            scheduled_arr_days = scheduled_dep_days
        scoped_bts = df_bts[
            scheduled_dep_days.isin(set(adsb_days.tolist())) | scheduled_arr_days.isin(set(adsb_days.tolist()))
        ].copy()
        bts_baseline_rows = int(len(scoped_bts))
        if 'Cancelled' in scoped_bts.columns:
            cancelled_flag = pd.to_numeric(scoped_bts['Cancelled'], errors='coerce').fillna(0).astype(int)
        else:
            cancelled_flag = pd.Series(0, index=scoped_bts.index, dtype='int64')
        delay = pd.to_numeric(scoped_bts.get('DepDelay'), errors='coerce')
        bts_baseline_cancelled = int(cancelled_flag.eq(1).sum())
        bts_baseline_late = int(((delay > 15) & cancelled_flag.ne(1)).sum())

    bts_matches = source_counts.get('bts', 0)
    report = {
        'generated_at': datetime.now().isoformat(),
        'merged_rows_total': int(len(merged_df)),
        'merged_source_counts': source_counts,
        'merge_candidate_metrics': {
            'bts': _latest_merge_qa_metrics('bts'),
            'eurocontrol': _latest_merge_qa_metrics('eurocontrol'),
        },
        'labeled_rows_total': int(len(labeled_df)),
        'label_distribution': label_distribution,
        'label_distribution_by_source': labels_by_source,
        'delay_minutes_range': {
            'min': float(pd.to_numeric(labeled_df.get('delay_minutes'), errors='coerce').min())
            if 'delay_minutes' in labeled_df.columns and not labeled_df['delay_minutes'].dropna().empty else None,
            'max': float(pd.to_numeric(labeled_df.get('delay_minutes'), errors='coerce').max())
            if 'delay_minutes' in labeled_df.columns and not labeled_df['delay_minutes'].dropna().empty else None,
        },
        'bts_baseline_in_scope': {
            'rows': bts_baseline_rows,
            'late': bts_baseline_late,
            'cancelled': bts_baseline_cancelled,
            'matched_rows': int(bts_matches),
            'recovery_pct': round((bts_matches / bts_baseline_rows) * 100, 2) if bts_baseline_rows else None,
        },
    }

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4, default=str)
    logger.info(f"Matching regression report saved to {report_path}")


def _should_skip(stage: str, force: bool) -> bool:
    """T9: Check if stage output already exists. Skip unless --force."""
    if force:
        return False
    output = STAGE_OUTPUTS.get(stage)
    if output and Path(output).exists():
        logger.info(f"Stage '{stage}' skipped — output already exists at {output}. Use --force to re-run.")
        return True
    return False


def stage_ingest(config: dict, force: bool = False):
    """Stage: Ingest — Download and combine datasets."""
    if _should_skip('ingest', force):
        return
    
    start = datetime.now()
    paths = config['paths']
    ingest_cfg = config.get('ingestion', {})
    raw_dir = Path(paths['raw_data_dir'])
    ensure_dir(raw_dir)
    
    year = ingest_cfg.get('year', 2025)
    start_month = ingest_cfg.get('start_month', 1)
    end_month = ingest_cfg.get('end_month', 11)
    
    # Download Eurocontrol data
    logger.info(f"Downloading Eurocontrol OPDI data for {year} months {start_month}-{end_month}...")
    euro_dl = EurocontrolDownloader()
    for m in range(start_month, end_month + 1):
        euro_dl.download_month(year, m, raw_dir)
    
    # BTS Combination
    bts_csv_dir = Path(ingest_cfg.get('bts_csv_dir', 'data/raw/bts'))
    bts_combined_file = Path(ingest_cfg.get('bts_combined_file', 'data/processed/bts_combined.parquet'))
    
    if not bts_combined_file.exists() or force:
        logger.info("Combining BTS CSV files into a single parquet...")
        combiner = BTSCombiner()
        combiner.combine_csvs(bts_csv_dir, bts_combined_file)
    else:
        logger.info(f"BTS combined file already exists: {bts_combined_file}")

    # Eurocontrol Combination
    euro_combined_file = Path(ingest_cfg.get('euro_combined_file', 'data/processed/eurocontrol_combined.parquet'))
    if not euro_combined_file.exists() or force:
        logger.info("Combining Eurocontrol Parquet files into a single parquet...")
        combiner = EuroCombiner()
        combiner.combine_parquets(raw_dir, euro_combined_file)
    else:
        logger.info(f"Eurocontrol combined file already exists: {euro_combined_file}")
    
    _log_run_metadata('ingest', start, 'SUCCESS')


def stage_features(config: dict, validator: DataValidator, force: bool = False):
    """Stage: Extract ADS-B flight-level features."""
    if _should_skip('features', force):
        return
    
    start = datetime.now()
    paths = config['paths']
    
    ingest_cfg = config.get('ingestion', {})
    traj_path = Path(paths['processed_data_dir']) / paths['trajectories_file']
    adsb_combined_path = Path(ingest_cfg.get('adsb_combined_file', 'data/processed/adsb_combined.parquet'))
    features_path = Path(paths['processed_data_dir']) / paths['features_file']

    if not adsb_combined_path.exists():
        logger.error(f"No trajectory data found. Run data acquisition first.")
        _log_run_metadata('features', start, 'FAILED')
        return

    feature_cfg = config.get('features', {})
    feature_mode = str(feature_cfg.get('mode', 'trajectory_reconstruction')).lower()

    if feature_mode in {'schedule_aware_direct', 'schedule_aware', 'direct'}:
        logger.info("Running schedule-aware direct ADS-B feature extraction...")
        direct_cfg = ScheduleAwareFeatureConfig(
            batch_size=int(feature_cfg.get('batch_size', config.get('trajectory', {}).get('batch_size', 200_000))),
            partitions=int(feature_cfg.get('partitions', config.get('trajectory', {}).get('reconstruction_partitions', 16))),
            pre_departure_hours=float(feature_cfg.get('pre_departure_hours', 2.0)),
            post_arrival_hours=float(feature_cfg.get('post_arrival_hours', 3.0)),
            fallback_duration_hours=float(feature_cfg.get('fallback_duration_hours', 3.0)),
            downsample_interval_seconds=int(feature_cfg.get('downsample_interval_seconds', 60)),
            phase_detail_minutes=int(feature_cfg.get('phase_detail_minutes', 15)),
            phase_interval_seconds=int(feature_cfg.get('phase_interval_seconds', 30)),
            max_points_per_flight=int(feature_cfg.get('max_points_per_flight', 300)),
            min_points_per_flight=int(feature_cfg.get('min_points_per_flight', 10)),
            save_trajectory_sketches=bool(feature_cfg.get('save_trajectory_sketches', True)),
            sketch_interval_seconds=int(feature_cfg.get('sketch_interval_seconds', 600)),
            sketch_phase_interval_seconds=int(feature_cfg.get('sketch_phase_interval_seconds', 120)),
            sketch_phase_detail_minutes=int(feature_cfg.get('sketch_phase_detail_minutes', 15)),
            sketch_max_points_per_flight=int(feature_cfg.get('sketch_max_points_per_flight', 80)),
            sketch_output_file=str(feature_cfg.get('sketch_output_file', 'trajectory_sketches.parquet')),
            max_schedule_rows=feature_cfg.get('max_schedule_rows'),
            schedule_sources=tuple(feature_cfg.get('schedule_sources', ['bts'])),
            cleanup_work_dir=bool(feature_cfg.get('cleanup_work_dir', False)),
            force_repartition=bool(force or feature_cfg.get('force_repartition', False)),
        )
        if direct_cfg.max_schedule_rows is not None:
            direct_cfg = ScheduleAwareFeatureConfig(
                **{
                    **direct_cfg.__dict__,
                    'max_schedule_rows': int(direct_cfg.max_schedule_rows),
                }
            )

        schedule_paths = {
            'bts': ingest_cfg.get('bts_combined_file', 'data/processed/bts_combined.parquet'),
            'eurocontrol': ingest_cfg.get('euro_combined_file', 'data/processed/eurocontrol_combined.parquet'),
        }
        df_features = extract_schedule_aware_features(
            adsb_path=adsb_combined_path,
            schedule_paths=schedule_paths,
            output_path=features_path,
            work_dir=Path(paths['processed_data_dir']) / '_schedule_aware_work',
            config=direct_cfg,
            max_gap_minutes=config.get('trajectory', {}).get('max_gap_minutes', 15),
        )

        validator.validate(df_features, FEATURES_SCHEMA, 'trajectory_features')
        logger.info(f"Features saved to {features_path} ({len(df_features)} rows)")
        _log_run_metadata(
            'features',
            start,
            'SUCCESS',
            len(df_features),
            {'feature_mode': feature_mode, 'schedule_sources': list(direct_cfg.schedule_sources)},
        )
        return

    require_traj_cols = {
        'trajectory_quality_status',
        'trajectory_quality_score',
        'is_full_flight',
        'starts_groundish',
        'ends_groundish',
        'has_altitude_spike',
        'is_mappable',
    }
    needs_rebuild = force or (not traj_path.exists())
    if traj_path.exists() and not force:
        traj_schema = set(pq.ParquetFile(traj_path).schema.names)
        if not require_traj_cols.issubset(traj_schema):
            logger.info("Existing trajectories are missing quality-aware fields; rebuilding.")
            needs_rebuild = True

    if needs_rebuild:
        logger.info("Running quality-aware trajectory reconstruction...")
        ctx = build_notebook_context(Path(__file__).resolve().parent)
        ctx.config = config
        ctx.adsb_input_path = adsb_combined_path.resolve()
        ctx.traj_path = traj_path.resolve()
        trajectory_cfg = config.get('trajectory', {})
        reconstruction_workers = int(trajectory_cfg.get('reconstruction_workers', 1))
        reconstruction_partitions = int(trajectory_cfg.get('reconstruction_partitions', max(reconstruction_workers, 1)))
        reconstruction_fn = reconstruct_full_dataset_parallel if reconstruction_workers > 1 else reconstruct_full_dataset
        reconstruction_kwargs = {
            'batch_size': trajectory_cfg.get('batch_size', 200_000),
            'max_gap_minutes': trajectory_cfg.get('max_gap_minutes', 15),
        }
        if reconstruction_fn is reconstruct_full_dataset_parallel:
            reconstruction_kwargs.update(
                {
                    'workers': reconstruction_workers,
                    'partitions': reconstruction_partitions,
                }
            )
        output_path, df_traj_quality = reconstruction_fn(ctx, **reconstruction_kwargs)
        logger.info("Reconstructed trajectories written to %s (%d quality summaries).", output_path, len(df_traj_quality))
    else:
        logger.info(f"Using existing quality-aware trajectories from {traj_path}...")
    
    logger.info("Extracting flight-level behavioral features...")
    fe = FeatureExtractor(max_gap_minutes=config.get('trajectory', {}).get('max_gap_minutes', 15))
    if hasattr(fe, 'extract_features_parallel'):
        df_features = fe.extract_features_parallel(
            traj_path,
            num_workers=max(1, min(6, (os.cpu_count() or 2) - 1)),
            partition_batch_size=200_000,
            worker_batch_size=200_000,
        )
    elif hasattr(fe, 'extract_features_streaming'):
        df_features = fe.extract_features_streaming(traj_path)
    else:
        df_traj = pd.read_parquet(traj_path)
        df_features = fe.extract_features(df_traj)

    if not df_features.empty:
        allowed_statuses = {'full_flight', 'partial_end_missing', 'partial_start_missing'}
        before_filter = len(df_features)
        if 'trajectory_quality_status' in df_features.columns:
            df_features = df_features[df_features['trajectory_quality_status'].isin(allowed_statuses)].copy()
        if 'num_points' in df_features.columns:
            df_features = df_features[df_features['num_points'] >= 10].copy()
        if 'flight_duration' in df_features.columns:
            df_features = df_features[df_features['flight_duration'].between(300, 86400, inclusive='both')].copy()
        logger.info(
            "Filtered flight-level features from %d to %d rows using completeness gates.",
            before_filter,
            len(df_features),
        )
        if 'trajectory_quality_status' in df_features.columns and not df_features.empty:
            logger.info("Post-filter trajectory status distribution: %s", df_features['trajectory_quality_status'].value_counts().to_dict())
    
    # T4: Validate features
    validator.validate(df_features, FEATURES_SCHEMA, 'trajectory_features')
    
    # Save features
    ensure_dir(features_path.parent)
    df_features.to_parquet(features_path, index=False)
    logger.info(f"Features saved to {features_path} ({len(df_features)} rows)")
    
    _log_run_metadata('features', start, 'SUCCESS', len(df_features))


def stage_merge(config: dict, validator: DataValidator, force: bool = False):
    """Stage: Merge ADS-B features with schedule data."""
    if _should_skip('merge', force):
        return
    
    start = datetime.now()
    paths = config['paths']
    ingest_cfg = config.get('ingestion', {})
    merge_cfg = config.get('merge', {})
    
    # Load features
    features_path = Path(paths['processed_data_dir']) / paths['features_file']
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}. Run --stage features first.")
        return
    
    df_adsb_features = pd.read_parquet(features_path)
    initial_adsb_count = len(df_adsb_features)
    
    # T10: Per-region tolerance
    tolerance_hours = merge_cfg.get('tolerance_hours', 2)
    merge_sources = {str(source).lower() for source in merge_cfg.get('schedule_sources', ['eurocontrol', 'bts'])}
    merger = DataMerger(tolerance_hours=tolerance_hours)
    merged_dfs = []
    
    euro_combined_file = Path(ingest_cfg.get('euro_combined_file', 'data/processed/eurocontrol_combined.parquet'))
    if 'eurocontrol' in merge_sources and euro_combined_file.exists():
        logger.info(f"  Merger: Processing Eurocontrol data from {euro_combined_file}...")
        df_euro = pd.read_parquet(euro_combined_file)
        if 'region' not in df_euro.columns:
            df_euro['region'] = 'EU'

        df_matched = merger.merge_datasets(df_adsb_features, df_euro)
        df_matched = df_matched.dropna(subset=['scheduled_dep']) if 'scheduled_dep' in df_matched.columns else df_matched
        if not df_matched.empty:
            merged_dfs.append(df_matched)

        del df_euro, df_matched
    elif 'eurocontrol' not in merge_sources:
        logger.info("  Merger: Skipping Eurocontrol because merge.schedule_sources excludes it.")

    bts_combined_file = Path(ingest_cfg.get('bts_combined_file', 'data/processed/bts_combined.parquet'))
    if 'bts' in merge_sources and bts_combined_file.exists():
        logger.info(f"  Merger: Processing BTS data from {bts_combined_file}...")
        df_bts = pd.read_parquet(bts_combined_file)
        if 'region' not in df_bts.columns:
            df_bts['region'] = 'US'
        
        df_matched = merger.merge_datasets(df_adsb_features, df_bts)
        df_matched = df_matched.dropna(subset=['scheduled_dep']) if 'scheduled_dep' in df_matched.columns else df_matched
        if not df_matched.empty:
            merged_dfs.append(df_matched)
            
        del df_bts, df_matched
    elif 'bts' not in merge_sources:
        logger.info("  Merger: Skipping BTS because merge.schedule_sources excludes it.")
        
    if merged_dfs:
        df_merged = pd.concat(merged_dfs, ignore_index=True)
        if 'source_dataset' in df_merged.columns:
            source_priority = {'bts': 0, 'eurocontrol': 1}
            df_merged['source_priority'] = df_merged['source_dataset'].map(source_priority).fillna(2)
            sort_cols = ['source_priority']
            if 'time_diff_minutes' in df_merged.columns:
                sort_cols.append('time_diff_minutes')
            df_merged = df_merged.sort_values(sort_cols, kind='stable')

        if 'trajectory_id' in df_merged.columns:
            df_merged = df_merged.drop_duplicates(subset=['trajectory_id'], keep='first')
        if 'flight_key' in df_merged.columns:
            df_merged = df_merged.drop_duplicates(subset=['flight_key'], keep='first')
        df_merged = df_merged.drop(columns=['source_priority'], errors='ignore')

        logger.info(f"Incremental merge found {len(df_merged):,} matched flights.")
        if 'source_dataset' in df_merged.columns:
            logger.info("Merged source counts: %s", df_merged['source_dataset'].value_counts().to_dict())
        if 'time_diff_minutes' in df_merged.columns and not df_merged['time_diff_minutes'].dropna().empty:
            logger.info(
                "Merged time_diff_minutes range: min=%.2f max=%.2f",
                df_merged['time_diff_minutes'].min(),
                df_merged['time_diff_minutes'].max(),
            )
    else:
        logger.warning("Merge resulted in empty dataframe. Returning unmatched ADS-B features.")
        df_merged = df_adsb_features
    
    # Validate merged dataset against the columns expected at merge time.
    validator.validate(
        df_merged,
        _stage_schema(
            ML_DATASET_SCHEMA,
            df_merged,
            required_cols=['flight_key', 'scheduled_dep', 'scheduled_arr', 'origin', 'destination', 'label_source'],
        ),
        'merged_dataset',
    )
    
    # Save intermediate merge output
    merge_path = Path('data/processed/ml_dataset_merged.parquet')
    ensure_dir(merge_path.parent)
    df_merged.to_parquet(merge_path, index=False)
    
    _log_run_metadata('merge', start, 'SUCCESS', len(df_merged), {
        'initial_adsb': initial_adsb_count,
        'merged_count': len(df_merged)
    })


def stage_weather(config: dict, validator: DataValidator, force: bool = False):
    """Stage: Integrate METAR weather data."""
    if _should_skip('weather', force):
        return
    
    start = datetime.now()
    paths = config['paths']
    
    merge_path = Path('data/processed/ml_dataset_merged.parquet')
    if not merge_path.exists():
        logger.error("Merged dataset not found. Run --stage merge first.")
        return
    
    df_merged = pd.read_parquet(merge_path)
    
    # Try parquet first (T0D output), then CSV
    weather_path = _resolve_weather_input(paths)

    if weather_path.exists() and weather_path.suffix.lower() == '.parquet':
        logger.info(f"Loading METAR weather data from {weather_path}...")
        df_weather = pd.read_parquet(weather_path)
    elif weather_path.exists():
        logger.info(f"Loading METAR weather data from {weather_path}...")
        df_weather = pd.read_csv(weather_path)
    else:
        logger.warning("No weather data found. Skipping weather integration.")
        # Save as-is
        out = Path('data/processed/ml_dataset_weather.parquet')
        df_merged.to_parquet(out, index=False)
        _log_run_metadata('weather', start, 'SKIPPED', len(df_merged))
        return
    
    weather_integrator = WeatherIntegrator()
    df_weather_enriched = weather_integrator.add_weather_features(df_merged, df_weather)

    # Task 3.1: Drop unverified Eurocontrol rows — they have no delay info
    if 'label_source' in df_weather_enriched.columns:
        unverified_mask = df_weather_enriched['label_source'].astype('string').eq('unverified_euro')
        n_drop = int(unverified_mask.sum())
        if n_drop > 0:
            df_weather_enriched = df_weather_enriched[~unverified_mask].copy()
            logger.info(f"Dropped {n_drop:,} unverified Eurocontrol rows (no delay data).")

    # ── Feature enrichment (Tasks 3.2, 3.3, 3.4, 2.3) ──
    from src.feature_enrichment import enrich_all
    df_weather_enriched = enrich_all(df_weather_enriched, df_weather)

    enroute_cfg = config.get('enroute_weather', {}) if isinstance(config.get('enroute_weather', {}), dict) else {}
    if bool(enroute_cfg.get('enabled', False)):
        try:
            from src.enroute_weather import EnrouteWeatherConfig, add_enroute_weather_features

            sketches_file = Path(paths.get('trajectory_sketches_file', 'trajectory_sketches.parquet'))
            if not sketches_file.is_absolute():
                sketches_file = Path(paths['processed_data_dir']) / sketches_file
            max_flights = enroute_cfg.get('max_flights')
            max_flights = int(max_flights) if max_flights not in (None, '') else None

            df_weather_enriched = add_enroute_weather_features(
                df_weather_enriched,
                sketches_path=sketches_file,
                config=EnrouteWeatherConfig(
                    enabled=True,
                    provider=str(enroute_cfg.get('provider', 'open_meteo')),
                    cache_dir=str(enroute_cfg.get('cache_dir', 'data/raw/enroute_weather_cache')),
                    max_flights=max_flights,
                    max_points_per_flight=int(enroute_cfg.get('max_points_per_flight', 24)),
                    request_sleep_seconds=float(enroute_cfg.get('request_sleep_seconds', 0.05)),
                    round_latlon_decimals=int(enroute_cfg.get('round_latlon_decimals', 2)),
                    round_time=str(enroute_cfg.get('round_time', '1h')),
                    timeout_seconds=int(enroute_cfg.get('timeout_seconds', 30)),
                    force_refresh=bool(enroute_cfg.get('force_refresh', False)),
                ),
            )
        except Exception as exc:
            logger.warning("En-route weather enrichment failed: %s", exc)

    traj_path = Path(paths['processed_data_dir']) / paths['trajectories_file']
    if traj_path.exists():
        try:
            from src.route_deviation import add_route_deviation_features
            df_weather_enriched = add_route_deviation_features(df_weather_enriched, traj_path)
        except Exception as exc:
            logger.warning("Route deviation enrichment failed: %s", exc)

    validator.validate(
        df_weather_enriched,
        _stage_schema(
            ML_DATASET_SCHEMA,
            df_weather_enriched,
            required_cols=['flight_key', 'scheduled_dep', 'scheduled_arr', 'origin', 'destination', 'label_source'],
        ),
        'weather_enriched_dataset',
    )
    
    out = Path('data/processed/ml_dataset_weather.parquet')
    df_weather_enriched.to_parquet(out, index=False)
    
    _log_run_metadata('weather', start, 'SUCCESS', len(df_weather_enriched))


def stage_label(config: dict, validator: DataValidator, force: bool = False):
    """Stage: Generate disruption labels."""
    if _should_skip('label', force):
        return
    
    start = datetime.now()
    
    weather_path = Path('data/processed/ml_dataset_weather.parquet')
    merge_path = Path('data/processed/ml_dataset_merged.parquet')
    
    input_path = weather_path if weather_path.exists() else merge_path
    if not input_path.exists():
        logger.error("No input dataset for labeling. Run --stage weather first.")
        return
    
    df = pd.read_parquet(input_path)
    
    labeler = LabelGenerator(delay_threshold_minutes=15)
    df_labeled = labeler.generate_labels(df)
    # Note: Eurocontrol rows already stripped in stage_weather (Task 3.1)
    df_final = labeler.standardize_dataset(df_labeled)
    
    # Inner join filtering
    if 'scheduled_dep' in df_final.columns:
        df_final = df_final.dropna(subset=['scheduled_dep']).copy()

    from src.feature_enrichment import add_temporal_features, add_airport_features, add_congestion_features
    df_final = add_temporal_features(df_final)
    df_final = add_airport_features(df_final)
    df_final = add_congestion_features(df_final)
    
    validator.validate(
        df_final,
        _stage_schema(
            ML_DATASET_SCHEMA,
            df_final,
            required_cols=['flight_key', 'scheduled_dep', 'scheduled_arr', 'origin', 'destination', 'label', 'delay_minutes'],
        ),
        'ml_dataset',
    )
    
    out = Path('data/processed/ml_dataset_labeled.parquet')
    df_final.to_parquet(out, index=False)
    
    _log_run_metadata('label', start, 'SUCCESS', len(df_final))


def stage_validate(config: dict, force: bool = False):
    """Stage: Quality gates on labeled dataset."""
    if _should_skip('validate', force):
        return
    
    start = datetime.now()
    
    label_path = Path('data/processed/ml_dataset_labeled.parquet')
    if not label_path.exists():
        logger.error("Labeled dataset not found. Run --stage label first.")
        return
    
    df = pd.read_parquet(label_path)
    
    # T15: Run quality gates
    qg = QualityGateRunner()
    report = qg.run_all(df, feature_cols=ML_FEATURE_COLUMNS)
    
    # T15: Add quality score columns
    df = qg.compute_feature_quality_score(df, feature_cols=ML_FEATURE_COLUMNS)
    
    # T16: Enforce column order
    final_cols = []
    for col in ML_DATASET_COLUMN_ORDER:
        if col in df.columns:
            final_cols.append(col)
        else:
            df[col] = pd.NA
            final_cols.append(col)

    # Optional, expensive feature families should be preserved when present but
    # not created as all-null columns when their enrichment step is disabled.
    optional_prefixes = ('enroute_',)
    optional_cols = [
        col for col in df.columns
        if col not in final_cols and col.startswith(optional_prefixes)
    ]
    final_cols.extend(optional_cols)
    
    # Drop any extra columns
    extra_cols = [c for c in df.columns if c not in ML_DATASET_COLUMN_ORDER]
    extra_cols = [c for c in extra_cols if c not in optional_cols]
    if extra_cols:
        logger.info(f"Dropping extra columns: {extra_cols}")
    
    df_final = df[final_cols].copy()
    
    # Save final ML dataset
    paths = config['paths']
    out_file = Path(paths['processed_data_dir']) / paths['ml_dataset_file']
    ensure_dir(out_file.parent)
    df_final.to_parquet(out_file, index=False)
    logger.info(f"Final ML dataset saved to {out_file} ({len(df_final)} rows, {len(final_cols)} columns)")
    logger.info(f"Final dataset columns: {final_cols}")
    
    # Validation summary
    if 'label' in df_final.columns:
        dist = df_final['label'].value_counts()
        logger.info("Label Distribution:")
        for label, count in dist.items():
            logger.info(f"  {label}: {count} ({count/len(df_final)*100:.1f}%)")
    
    if 'region' in df_final.columns:
        reg_dist = df_final['region'].value_counts()
        logger.info("Region Distribution:")
        for reg, count in reg_dist.items():
            logger.info(f"  {reg}: {count}")

    merged_path = Path('data/processed/ml_dataset_merged.parquet')
    if merged_path.exists():
        df_merged = pd.read_parquet(merged_path)
        _write_matching_regression_report(df_merged, df_final)
    
    _log_run_metadata('validate', start, 'SUCCESS', len(df_final), {
        'quality_gates_passed': report.get('overall_passed', False),
        'training_eligible': report.get('feature_quality', {}).get('training_eligible_count', 0)
    })


def stage_train(config: dict, force: bool = False):
    """Stage: Train ML models."""
    if _should_skip('train', force):
        return
    
    start = datetime.now()
    paths = config['paths']
    
    ml_path = Path(paths['processed_data_dir']) / paths['ml_dataset_file']
    if not ml_path.exists():
        logger.error(f"ML dataset not found at {ml_path}. Run --stage validate first.")
        return
    
    df = pd.read_parquet(ml_path)
    
    try:
        from src.model_training import ModelTrainer
        
        trainer = ModelTrainer(config)
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        logger.info(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        # Train all models
        results = trainer.train_all(X_train, X_test, y_train, y_test)
        
        # Generate comparison report and visuals
        trainer.generate_comparison_report(results, X_test, y_test)
        
        # T21: SHAP explanations for best model
        try:
            trainer.explain(results, X_test, y_test)
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
        
        _log_run_metadata('train', start, 'SUCCESS', len(df), {
            'n_models': len(results),
            'model_names': list(results.keys())
        })
    except ImportError as e:
        logger.error(f"ML dependencies not installed: {e}. Install via: pip install scikit-learn xgboost lightgbm optuna shap")
        _log_run_metadata('train', start, 'FAILED', 0, {'error': str(e)})


def stage_drift(config: dict, force: bool = False):
    """Stage: Drift detection on ML dataset (T31)."""
    if _should_skip('drift', force):
        return
    
    start = datetime.now()
    paths = config['paths']
    
    ml_path = Path(paths['processed_data_dir']) / paths['ml_dataset_file']
    if not ml_path.exists():
        logger.error(f"ML dataset not found at {ml_path}. Run --stage validate first.")
        return
    
    df = pd.read_parquet(ml_path)
    
    try:
        from src.drift_detector import DriftDetector
        
        detector = DriftDetector()
        drift_df = detector.monthly_drift_report(df, feature_cols=ML_FEATURE_COLUMNS)
        
        if drift_df is not None and not drift_df.empty:
            stability = detector.stability_ranking(drift_df)
            
            # Cross with feature importance if available
            importance_path = Path('models/feature_importance.json')
            if importance_path.exists():
                import json
                with open(importance_path) as f:
                    importance = pd.DataFrame(json.load(f))
                unstable = detector.flag_high_importance_unstable(drift_df, importance)
                
                # Warn about high-importance unstable features
                if unstable is not None and not unstable.empty:
                    for _, row in unstable.iterrows():
                        logger.warning(
                            f"DRIFT WARNING: Feature '{row['feature']}' is drifting "
                            f"(PSI={row.get('avg_psi', 'N/A'):.3f}) AND is in top-10 importance"
                        )
            
            # Console summary
            alerts = drift_df[drift_df['drift_alert'] == True] if 'drift_alert' in drift_df.columns else pd.DataFrame()
            if not alerts.empty:
                logger.warning(f"Drift detected in {alerts['feature'].nunique()} features across {alerts['month'].nunique()} months")
            else:
                logger.info("No significant drift detected.")
        
        _log_run_metadata('drift', start, 'SUCCESS', len(df))
    except ImportError as e:
        logger.error(f"Drift detection dependencies not installed: {e}")
        _log_run_metadata('drift', start, 'FAILED', 0, {'error': str(e)})


def main():
    """Main pipeline orchestrator with --stage CLI (T8)."""
    parser = argparse.ArgumentParser(description="Flight Disruption Prediction Pipeline")
    parser.add_argument('--stage', type=str, default='all',
                        choices=['ingest', 'features', 'merge', 'weather', 'label', 
                                 'validate', 'train', 'drift', 'all'],
                        help='Pipeline stage to run')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run without executing')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if output exists (T9)')
    
    args = parser.parse_args()
    
    # Setup
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "configs" / "config.yaml"
    env_path = base_dir / ".env"
    
    # Pre-flight checks
    validate_environment(env_path)
    
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    
    # T4: DataValidator
    validation_mode = config.get('validation_mode', 'warn_only')
    validator = DataValidator(mode=validation_mode)
    
    # Stage dispatch
    stages_order = ['ingest', 'features', 'merge', 'weather', 'label', 'validate', 'train', 'drift']
    
    if args.stage == 'all':
        stages_to_run = stages_order
    else:
        stages_to_run = [args.stage]
    
    if args.dry_run:
        for s in stages_to_run:
            print(f"Would run: {s}")
        return
    
    logger.info(f"Starting pipeline — stages: {stages_to_run}")
    
    stage_functions = {
        'ingest': lambda: stage_ingest(config, args.force),
        'features': lambda: stage_features(config, validator, args.force),
        'merge': lambda: stage_merge(config, validator, args.force),
        'weather': lambda: stage_weather(config, validator, args.force),
        'label': lambda: stage_label(config, validator, args.force),
        'validate': lambda: stage_validate(config, args.force),
        'train': lambda: stage_train(config, args.force),
        'drift': lambda: stage_drift(config, args.force),
    }
    
    for stage_name in stages_to_run:
        logger.info(f"{'='*60}")
        logger.info(f"Running stage: {stage_name}")
        logger.info(f"{'='*60}")
        try:
            stage_functions[stage_name]()
        except Exception as e:
            logger.error(f"Stage '{stage_name}' failed: {e}")
            _log_run_metadata(stage_name, datetime.now(), 'FAILED', 0, {'error': str(e)})
            if args.stage != 'all':
                raise
    
    logger.info("Pipeline Complete!")


if __name__ == "__main__":
    main()
