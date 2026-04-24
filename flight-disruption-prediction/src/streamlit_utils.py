"""FE1: Shared Streamlit utility functions."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st


def load_config():
    import yaml
    config_path = Path('configs/config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


@st.cache_data(ttl=300)
def load_dataset(name: str = 'ml_dataset') -> pd.DataFrame:
    config = load_config()
    paths = config.get('paths', {})
    data_dir = paths.get('processed_data_dir', 'data/processed')
    
    file_map = {
        'ml_dataset': paths.get('ml_dataset_file', 'ml_dataset.parquet'),
        'features': paths.get('features_file', 'trajectory_features.parquet'),
        'trajectories': paths.get('trajectories_file', 'trajectories.parquet'),
        'merged': 'ml_dataset_merged.parquet',
        'labeled': 'ml_dataset_labeled.parquet',
    }
    
    filename = file_map.get(name, f'{name}.parquet')
    filepath = Path(data_dir) / filename
    
    if filepath.exists():
        return pd.read_parquet(filepath)
    return pd.DataFrame()


@st.cache_data(ttl=300)
def get_parquet_metadata(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        return {"exists": False, "rows": 0, "columns": []}
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        return {
            "exists": True,
            "rows": int(parquet_file.metadata.num_rows),
            "columns": parquet_file.schema.names,
        }
    except Exception:
        return {"exists": True, "rows": 0, "columns": []}


@st.cache_data(ttl=300)
def read_parquet_limited(
    path: str | Path,
    columns: list[str] | None = None,
    limit: int = 100_000,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        selected_columns = columns or dataset.schema.names
        selected_columns = [col for col in selected_columns if col in dataset.schema.names]
        if not selected_columns:
            selected_columns = dataset.schema.names
        table = dataset.head(limit, columns=selected_columns)
        return table.to_pandas()
    except Exception:
        try:
            return pd.read_parquet(path, columns=columns).head(limit)
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=300)
def read_parquet_filtered(
    path: str | Path,
    columns: list[str] | None = None,
    filters: dict | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    filters = filters or {}
    try:
        import pyarrow.dataset as ds

        dataset = ds.dataset(path, format="parquet")
        selected_columns = columns or dataset.schema.names
        selected_columns = [col for col in selected_columns if col in dataset.schema.names]
        if not selected_columns:
            selected_columns = dataset.schema.names

        expression = None
        for key, value in filters.items():
            if key not in dataset.schema.names:
                continue
            term = ds.field(key) == value
            expression = term if expression is None else expression & term

        table = dataset.head(limit, columns=selected_columns, filter=expression)
        return table.to_pandas()
    except Exception:
        try:
            df = pd.read_parquet(path, columns=columns)
            for key, value in filters.items():
                if key in df.columns:
                    df = df[df[key] == value]
            return df.head(limit)
        except Exception:
            return pd.DataFrame()


def load_model(name: str):
    import joblib
    path = Path('models') / f'{name}.pkl'
    if path.exists():
        return joblib.load(path)
    return None


def load_json(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def get_run_metadata() -> list:
    log_dir = Path('logs')
    if not log_dir.exists():
        return []
    
    runs = []
    for f in sorted(log_dir.glob('run_*.json'), reverse=True):
        with open(f) as fp:
            runs.append(json.load(fp))
    return runs


def style_page(title: str, icon: str = "✈️"):
    st.set_page_config(page_title=f"{title} — Flight Pipeline", page_icon=icon, layout="wide")
    st.markdown(f"# {icon} {title}")
