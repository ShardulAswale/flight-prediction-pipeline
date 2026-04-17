from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import main as pipeline_main
from src.data_validator import DataValidator
from src.model_training import ModelTrainer


def _build_test_config() -> dict:
    return {
        "seed": 42,
        "validation_mode": "warn_only",
        "trajectory": {
            "max_gap_minutes": 15,
            "batch_size": 10_000,
        },
        "ingestion": {
            "adsb_combined_file": "data/processed/adsb_combined.parquet",
            "bts_combined_file": "data/processed/bts_combined.parquet",
            "euro_combined_file": "data/processed/eurocontrol_combined.parquet",
        },
        "merge": {
            "tolerance_hours": {
                "US": 3,
                "default": 2,
            }
        },
        "paths": {
            "raw_data_dir": "data/raw",
            "processed_data_dir": "data/processed",
            "trajectories_file": "trajectories.parquet",
            "features_file": "trajectory_features.parquet",
            "ml_dataset_file": "ml_dataset.parquet",
            "metar_data_file": "data/raw/metar.parquet",
        },
        "training": {
            "use_gpu": False,
            "correlation_threshold": 0.95,
        },
    }


def _make_synthetic_adsb_and_bts():
    adsb_rows = []
    bts_rows = []
    airports = [("KATL", "KJFK"), ("KJFK", "KATL")]
    base_day = pd.Timestamp("2022-05-01 08:00:00", tz="UTC")

    for flight_idx in range(30):
        dep_time = base_day + pd.Timedelta(days=flight_idx // 3, hours=flight_idx % 3)
        arr_time = dep_time + pd.Timedelta(minutes=55)
        icao24 = f"abc{flight_idx:03d}"
        callsign = f"AAL{100 + flight_idx}"
        origin, destination = airports[flight_idx % len(airports)]
        delay = 0 if flight_idx % 4 else 35

        for point_idx in range(12):
            ts = dep_time + pd.Timedelta(minutes=5 * point_idx)
            lat_base = 33.64 if origin == "KATL" else 40.64
            lon_base = -84.43 if origin == "KATL" else -73.78
            lat_step = 0.35 if destination == "KJFK" else -0.35
            lon_step = 0.85 if destination == "KJFK" else -0.85
            phase_altitudes = [100, 400, 1200, 3000, 5500, 8500, 10500, 10500, 8500, 5000, 1200, 120]
            adsb_rows.append(
                {
                    "icao24": icao24,
                    "callsign": callsign,
                    "timestamp": int(ts.timestamp()),
                    "latitude": lat_base + point_idx * 0.6 + (0.02 * np.sin(point_idx)),
                    "longitude": lon_base + point_idx * lon_step * 0.2,
                    "geo_altitude": float(phase_altitudes[point_idx]),
                    "baro_altitude": float(phase_altitudes[point_idx]),
                    "velocity": float(160 + point_idx * 8),
                    "true_track": float(90 if destination == "KJFK" else 270),
                    "vertical_rate": float(0 if point_idx in {6, 7} else 3.5),
                    "on_ground": bool(point_idx in {0, 11}),
                    "sample_date": ts.strftime("%Y-%m-%d"),
                }
            )

        bts_rows.append(
            {
                "flight_key": f"{callsign}_{dep_time.date()}",
                "callsign": callsign,
                "scheduled_dep_utc": dep_time,
                "scheduled_arr_utc": arr_time,
                "service_day_utc": dep_time.normalize(),
                "actual_dep_utc": dep_time + pd.Timedelta(minutes=max(delay, 0)),
                "actual_arr_utc": arr_time + pd.Timedelta(minutes=max(delay, 0)),
                "origin": origin,
                "destination": destination,
                "cancelled": 0,
                "Cancelled": 0,
                "DepDelay": delay,
                "region": "US",
                "source_dataset": "bts",
            }
        )

    return pd.DataFrame(adsb_rows), pd.DataFrame(bts_rows)


def _make_synthetic_weather(bts_df: pd.DataFrame) -> pd.DataFrame:
    icao_to_iata = {"KATL": "ATL", "KJFK": "JFK"}
    rows = []
    for airport in sorted(set(bts_df["origin"]).union(set(bts_df["destination"]))):
        iata = icao_to_iata[airport]
        for hour in range(0, 24 * 12):
            ts = pd.Timestamp("2022-05-01 00:00:00", tz="UTC") + pd.Timedelta(hours=hour)
            rows.append(
                {
                    "airport_code": iata,
                    "timestamp": ts,
                    "wind_speed": 10.0 + (hour % 5),
                    "visibility": 10.0,
                    "temperature": 18.0 + (hour % 7),
                    "precipitation": 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_pipeline_e2e(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for rel in ["data/raw", "data/processed", "logs", "models", "outputs"]:
        Path(rel).mkdir(parents=True, exist_ok=True)

    config = _build_test_config()
    adsb_df, bts_df = _make_synthetic_adsb_and_bts()
    weather_df = _make_synthetic_weather(bts_df)

    adsb_df.to_parquet("data/processed/adsb_combined.parquet", index=False)
    bts_df.to_parquet("data/processed/bts_combined.parquet", index=False)
    weather_df.to_parquet("data/raw/metar.parquet", index=False)

    validator = DataValidator(mode="warn_only", log_dir="logs")

    pipeline_main.stage_features(config, validator, force=True)
    pipeline_main.stage_merge(config, validator, force=True)
    pipeline_main.stage_weather(config, validator, force=True)
    pipeline_main.stage_label(config, validator, force=True)
    pipeline_main.stage_validate(config, force=True)

    ml_path = Path("data/processed/ml_dataset.parquet")
    assert ml_path.exists(), "Final ML dataset was not created."

    df_ml = pd.read_parquet(ml_path)
    assert not df_ml.empty
    assert "label" in df_ml.columns
    assert "weather_severity" in df_ml.columns
    assert "dep_hour" in df_ml.columns
    assert "origin_flight_count" in df_ml.columns
    assert df_ml["weather_severity"].notna().any()
    assert df_ml["dep_hour"].notna().any()
    assert df_ml["origin_flight_count"].notna().any()

    trainer = ModelTrainer(config)
    X_train, X_test, y_train, y_test = trainer.prepare_data(df_ml)
    model = trainer.train_logistic_regression(X_train, y_train)
    assert model is not None
    assert Path("models/logistic_regression.pkl").exists()
