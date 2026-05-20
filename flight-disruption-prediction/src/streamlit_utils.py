"""FE1: Shared Streamlit utility functions."""
import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


WALKTHROUGH_STEPS = [
    {
        "key": "overview",
        "number": 1,
        "title": "Study Overview",
        "chapter": "Chapters 1-2",
        "page": "pages/0_Study_Overview.py",
        "purpose": "Define the research problem, aim, final design choice, and dissertation roadmap.",
    },
    {
        "key": "pipeline",
        "number": 2,
        "title": "Data and Pipeline",
        "chapter": "Chapter 3 / Section 4.2",
        "page": "pages/1_Pipeline_Overview.py",
        "purpose": "Show the data sources, processing stages, retained outputs, and final schedule-aware pipeline.",
    },
    {
        "key": "dataset",
        "number": 3,
        "title": "Dataset Construction",
        "chapter": "Chapter 4.3-4.6",
        "page": "pages/2_Data_Explorer.py",
        "purpose": "Inspect the retained datasets and the reduction from raw data to the final labelled ML table.",
    },
    {
        "key": "trajectory",
        "number": 4,
        "title": "ADS-B Trajectory Evidence",
        "chapter": "Chapter 3.4 / Section 4.5",
        "page": "pages/3_Trajectory_Map.py",
        "purpose": "Demonstrate what ADS-B adds beyond schedule data: geometry, altitude, speed, and trace quality.",
    },
    {
        "key": "features",
        "number": 5,
        "title": "Feature Engineering",
        "chapter": "Chapter 3.4 / Sections 4.5-4.8",
        "page": "pages/4_Feature_Analysis.py",
        "purpose": "Explain the engineered feature groups, importance rankings, and interpretability evidence.",
    },
    {
        "key": "models",
        "number": 6,
        "title": "Model Results",
        "chapter": "Sections 4.9-4.14",
        "page": "pages/5_Model_Performance.py",
        "purpose": "Compare Logistic Regression, Random Forest, and XGBoost using the final binary target.",
    },
    {
        "key": "quality",
        "number": 7,
        "title": "Quality and Limitations",
        "chapter": "Sections 4.4, 4.12 and Chapter 5",
        "page": "pages/6_Data_Quality.py",
        "purpose": "Surface missingness, class imbalance, and the main limitations that constrain interpretation.",
    },
    {
        "key": "prediction",
        "number": 8,
        "title": "Prediction Demo",
        "chapter": "Practical Demonstration",
        "page": "pages/7_Predictions_Explorer.py",
        "purpose": "Finish with the live probability interface and route-level prediction demonstration.",
    },
]


PAGE_URL_PATHS = {
    "pages/0_Study_Overview.py": "",
    "pages/1_Pipeline_Overview.py": "pipeline_overview",
    "pages/2_Data_Explorer.py": "data_explorer",
    "pages/3_Trajectory_Map.py": "trajectory_map",
    "pages/4_Feature_Analysis.py": "feature_analysis",
    "pages/5_Model_Performance.py": "model_performance",
    "pages/6_Data_Quality.py": "data_quality",
    "pages/7_Predictions_Explorer.py": "predictions_explorer",
    "pages/8_Pipeline_Runbook.py": "pipeline_runbook",
}


def render_walkthrough_header(step_key: str):
    steps = WALKTHROUGH_STEPS
    step = next(item for item in steps if item["key"] == step_key)
    idx = steps.index(step)
    st.caption(f"Step {step['number']} of {len(steps)} | {step['chapter']}")
    st.progress(step["number"] / len(steps))
    st.info(step["purpose"])

    st.markdown(
        """
        <style>
            .walkthrough-nav {
                display: flex;
                justify-content: space-between;
                align-items: stretch;
                gap: 1rem;
                margin: 0.7rem 0 1.1rem 0;
            }
            .walkthrough-nav-item {
                flex: 1 1 0;
                min-width: 0;
            }
            .walkthrough-nav-item.next {
                text-align: right;
            }
            .walkthrough-nav-link {
                display: inline-flex;
                flex-direction: column;
                max-width: min(100%, 360px);
                padding: 0.55rem 0.8rem;
                border-radius: 12px;
                border: 1px solid rgba(120, 144, 168, 0.36);
                background: rgba(240, 246, 250, 0.78);
                color: inherit;
                text-decoration: none;
                line-height: 1.25;
                white-space: normal;
            }
            .walkthrough-nav-link:hover {
                border-color: rgba(50, 101, 138, 0.65);
                background: rgba(229, 240, 247, 0.95);
                text-decoration: none;
            }
            .walkthrough-nav-direction {
                font-size: 0.75rem;
                font-weight: 700;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                opacity: 0.72;
            }
            .walkthrough-nav-title {
                margin-top: 0.12rem;
                font-size: 0.93rem;
                font-weight: 700;
            }
            @media (max-width: 760px) {
                .walkthrough-nav {
                    flex-direction: column;
                }
                .walkthrough-nav-item.next {
                    text-align: left;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    previous_html = ""
    next_html = ""
    if idx > 0:
        prev_step = steps[idx - 1]
        previous_html = _walkthrough_nav_link(prev_step, "Previous", "prev")
    if idx < len(steps) - 1:
        next_step = steps[idx + 1]
        next_html = _walkthrough_nav_link(next_step, "Next", "next")

    st.markdown(
        f"""
        <div class="walkthrough-nav">
            <div class="walkthrough-nav-item prev">{previous_html}</div>
            <div class="walkthrough-nav-item next">{next_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _walkthrough_nav_link(step: dict, direction: str, css_class: str) -> str:
    href = _page_href(step["page"])
    direction_text = escape(direction)
    title = escape(step["title"])
    return (
        f'<a class="walkthrough-nav-link {css_class}" href="{href}">'
        f'<span class="walkthrough-nav-direction">{direction_text}</span>'
        f'<span class="walkthrough-nav-title">{title}</span>'
        "</a>"
    )


def _page_href(page: str) -> str:
    url_path = PAGE_URL_PATHS.get(page)
    if url_path is None:
        page_name = Path(page).stem
        if page_name.startswith("0_"):
            url_path = ""
        else:
            url_path = page_name.split("_", 1)[-1].lower()
    return "/" if not url_path else f"/{url_path}"


def _safe_page_link(container, page: str, label: str):
    try:
        container.page_link(page, label=label)
    except KeyError:
        # Streamlit's isolated AppTest mode does not register multipage routes.
        container.caption(label)


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


def style_page(title: str, icon: str = "âœˆï¸"):
    st.set_page_config(page_title=f"{title} â€” Flight Pipeline", page_icon=icon, layout="wide")
    st.markdown(f"# {icon} {title}")
