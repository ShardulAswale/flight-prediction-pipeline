"""
T23: Shared pytest fixtures for all test modules.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone


@pytest.fixture
def sample_adsb_df():
    """20-row ADS-B state vectors DataFrame."""
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        'icao24': [f'abc{i:03d}' for i in range(n)],
        'callsign': [f'DL{100+i}' for i in range(n)],
        'timestamp': [1704067200 + i * 300 for i in range(n)],
        'latitude': np.random.uniform(40, 55, n),
        'longitude': np.random.uniform(-5, 10, n),
        'baro_altitude': np.random.uniform(5000, 12000, n),
        'velocity': np.random.uniform(150, 300, n),
        'true_track': np.random.uniform(0, 360, n),
        'on_ground': [False] * n,
    })


@pytest.fixture
def sample_schedule_df():
    """10-row canonical schedule DataFrame."""
    n = 10
    base_time = pd.Timestamp('2025-06-15 08:00:00', tz='UTC')
    return pd.DataFrame({
        'flight_key': [f'DL{100+i}_20250615' for i in range(n)],
        'callsign': [f'DL{100+i}' for i in range(n)],
        'scheduled_dep': [base_time + pd.Timedelta(hours=i) for i in range(n)],
        'scheduled_arr': [base_time + pd.Timedelta(hours=i+2) for i in range(n)],
        'origin': ['KJFK', 'KLAX', 'KORD', 'KATL', 'KSFO'] * 2,
        'destination': ['KLAX', 'KORD', 'KATL', 'KSFO', 'KJFK'] * 2,
        'cancelled': [0] * n,
        'source_dataset': ['bts'] * 5 + ['eurocontrol'] * 5,
    })


@pytest.fixture
def sample_features_df():
    """10-row flight features DataFrame."""
    np.random.seed(42)
    n = 10
    return pd.DataFrame({
        'trajectory_id': [f'abc{i:03d}_1' for i in range(n)],
        'icao24': [f'abc{i:03d}' for i in range(n)],
        'callsign': [f'DL{100+i}' for i in range(n)],
        'timestamp': [1704067200 + i * 3600 for i in range(n)],
        'flight_duration': np.random.uniform(3600, 36000, n),
        'trajectory_length': np.random.uniform(100, 5000, n),
        'mean_altitude': np.random.uniform(5000, 12000, n),
        'altitude_variance': np.random.uniform(0, 1000, n),
        'mean_speed': np.random.uniform(150, 300, n),
        'max_speed': np.random.uniform(250, 350, n),
        'speed_std': np.random.uniform(5, 30, n),
        'vertical_rate_std': np.random.uniform(0, 5, n),
        'heading_variability': np.random.uniform(0, 100, n),
        'holding_pattern_count': np.random.randint(0, 3, n),
        'altitude_change_count': np.random.randint(0, 10, n),
        'unstable_descent_flag': [False] * n,
    })


@pytest.fixture
def sample_weather_df():
    """5-row weather DataFrame."""
    base_time = pd.Timestamp('2025-06-15 08:00:00', tz='UTC')
    return pd.DataFrame({
        'airport_code': ['KJFK', 'KLAX', 'KORD', 'KATL', 'KSFO'],
        'timestamp': [base_time + pd.Timedelta(hours=i) for i in range(5)],
        'wind_speed': [10.0, 15.0, 25.0, 5.0, 12.0],
        'visibility': [10.0, 8.0, 3.0, 15.0, 6.0],
        'temperature': [72.0, 80.0, 65.0, 85.0, 60.0],
        'precipitation': [0.0, 0.0, 0.3, 0.0, 0.1],
    })


@pytest.fixture
def sample_ml_dataset(sample_features_df, sample_schedule_df):
    """Combined ML-ready dataset for quality gate testing."""
    df = sample_features_df.copy()
    df['flight_key'] = [f'DL{100+i}_20250615' for i in range(len(df))]
    df['region'] = ['US'] * 5 + ['EU'] * 5
    df['source_dataset'] = ['bts'] * 5 + ['eurocontrol'] * 5
    df['scheduled_dep'] = pd.Timestamp('2025-06-15 08:00:00', tz='UTC')
    df['scheduled_arr'] = pd.Timestamp('2025-06-15 10:00:00', tz='UTC')
    df['origin'] = 'KJFK'
    df['destination'] = 'KLAX'
    df['cancelled'] = 0
    df['wind_speed'] = 10.0
    df['visibility'] = 8.0
    df['temperature'] = 72.0
    df['precipitation'] = 0.0
    df['weather_severity'] = 0.0
    df['weather_confidence'] = 'High'
    df['match_quality'] = 'Phase A (Exact)'
    df['delay_minutes'] = [5, 20, 0, -3, 30, 10, 45, 0, 15, 60]
    df['label'] = ['Normal', 'Late', 'Normal', 'Normal', 'Late',
                   'Normal', 'Late', 'Normal', 'Normal', 'Late']
    return df
