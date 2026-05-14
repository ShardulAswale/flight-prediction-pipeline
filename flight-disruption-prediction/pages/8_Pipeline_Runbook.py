"""Interactive pipeline runbook page.

Shows each pipeline stage as a clickable flow step with the command, purpose,
main inputs, and outputs needed to run or explain the project.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import get_parquet_metadata, load_json


st.set_page_config(page_title="Pipeline Runbook", page_icon="Flow", layout="wide")


@dataclass(frozen=True)
class Stage:
    key: str
    title: str
    command: str
    purpose: str
    inputs: list[str]
    outputs: list[str]
    notes: list[str]
    status_paths: list[str]


STAGES: list[Stage] = [
    Stage(
        key="plan",
        title="1. Plan Ingestion",
        command="python scripts/run_master_ingestion.py --start-date 2022-01-01 --end-date 2022-06-30",
        purpose="Create the monthly ingestion plan without downloading data.",
        inputs=["configs/config.yaml"],
        outputs=[
            "outputs/ingestion_control/monthly_ingestion_plan.csv",
            "outputs/ingestion_control/monthly_ingestion_status.csv",
        ],
        notes=[
            "Use this first when checking the date window and source coverage.",
            "This is a dry-run planning step; it should be fast.",
        ],
        status_paths=["outputs/ingestion_control/monthly_ingestion_plan.csv"],
    ),
    Stage(
        key="download",
        title="2. Download Sources",
        command="python scripts/run_master_ingestion.py --start-date 2022-01-01 --end-date 2022-06-30 --execute",
        purpose="Download and normalize the source datasets used by the pipeline.",
        inputs=["OpenSky sample archive", "BTS On-Time Performance data", "METAR/IEM weather source"],
        outputs=[
            "data/processed/adsb_combined.parquet",
            "data/processed/bts_combined.parquet",
            "data/raw/metar.parquet",
        ],
        notes=[
            "This is the heavy data acquisition stage.",
            "Eurocontrol data may exist locally, but the final supervised model uses BTS labels only.",
        ],
        status_paths=[
            "data/processed/adsb_combined.parquet",
            "data/processed/bts_combined.parquet",
            "data/raw/metar.parquet",
        ],
    ),
    Stage(
        key="features",
        title="3. Extract ADS-B Features",
        command="python main.py --stage features --force",
        purpose="Run the schedule-aware ADS-B feature extraction path.",
        inputs=["data/processed/adsb_combined.parquet", "data/processed/bts_combined.parquet"],
        outputs=[
            "data/processed/trajectory_features.parquet",
            "data/processed/trajectory_sketches.parquet",
        ],
        notes=[
            "BTS schedule windows are loaded first.",
            "ADS-B pings are filtered by scheduled callsign and time window.",
            "The pipeline avoids building a full point-level trajectory table.",
        ],
        status_paths=["data/processed/trajectory_features.parquet"],
    ),
    Stage(
        key="merge",
        title="4. Merge With Schedule Labels",
        command="python main.py --stage merge --force",
        purpose="Match ADS-B-derived feature rows to BTS schedule records.",
        inputs=["data/processed/trajectory_features.parquet", "data/processed/bts_combined.parquet"],
        outputs=["data/processed/ml_dataset_merged.parquet", "logs/matching_regression_report.json"],
        notes=[
            "Matching uses callsign, airport, and scheduled departure timing.",
            "The final labelled dataset is BTS-based because BTS contains verified delay and cancellation fields.",
        ],
        status_paths=["data/processed/ml_dataset_merged.parquet"],
    ),
    Stage(
        key="weather",
        title="5. Add METAR Weather",
        command="python main.py --stage weather --force",
        purpose="Add nearest airport METAR weather observations to the merged flight table.",
        inputs=["data/processed/ml_dataset_merged.parquet", "data/raw/metar.parquet"],
        outputs=["data/processed/ml_dataset_weather.parquet", "logs/weather_coverage.json"],
        notes=[
            "Weather is merged by nearest airport and nearest timestamp within tolerance.",
            "Weather coverage is partial and should be reported as such.",
        ],
        status_paths=["data/processed/ml_dataset_weather.parquet"],
    ),
    Stage(
        key="label",
        title="6. Generate Labels",
        command="python main.py --stage label --force",
        purpose="Create Normal, Late, Cancelled, and binary Normal vs Disrupted target labels.",
        inputs=["data/processed/ml_dataset_weather.parquet"],
        outputs=["data/processed/ml_dataset_labeled.parquet", "logs/label_distribution.json"],
        notes=[
            "Normal: arrival delay below 15 minutes and not cancelled.",
            "Late: arrival delay at or above 15 minutes.",
            "Cancelled: BTS cancellation flag set.",
            "Binary Disrupted merges Late and Cancelled.",
        ],
        status_paths=["data/processed/ml_dataset_labeled.parquet"],
    ),
    Stage(
        key="validate",
        title="7. Validate Dataset",
        command="python main.py --stage validate --force",
        purpose="Run quality gates and write the final modelling dataset.",
        inputs=["data/processed/ml_dataset_labeled.parquet"],
        outputs=["data/processed/ml_dataset.parquet", "logs/quality_gates_report.json"],
        notes=[
            "Checks missingness, duplicates, feature quality, and label balance.",
            "The final model should use this validated dataset.",
        ],
        status_paths=["data/processed/ml_dataset.parquet"],
    ),
    Stage(
        key="train",
        title="8. Train Models",
        command="python main.py --stage train --force",
        purpose="Train Logistic Regression, Random Forest, and XGBoost and save evaluation outputs.",
        inputs=["data/processed/ml_dataset.parquet"],
        outputs=[
            "models/model_comparison.json",
            "models/xgboost.pkl",
            "outputs/roc_curves.png",
            "outputs/shap_summary.png",
        ],
        notes=[
            "The current final target is binary: Normal vs Disrupted.",
            "Use the saved model comparison JSON as the source of final metrics.",
            "Generated probabilities are raw model probabilities, not calibrated risk estimates.",
        ],
        status_paths=["models/model_comparison.json", "outputs/shap_summary.png"],
    ),
    Stage(
        key="reports",
        title="9. Generate Reports",
        command="python scripts/generate_pipeline_report.py",
        purpose="Generate saved charts and tables for the dissertation and dashboard.",
        inputs=["data/processed/ml_dataset.parquet", "models/model_comparison.json"],
        outputs=["outputs/pipeline_report/", "outputs/notebook_reports/"],
        notes=[
            "Run this after training if report figures or CSV tables need refreshing.",
            "Use report outputs for dissertation figures rather than recomputing manually.",
        ],
        status_paths=["outputs/pipeline_report/model_ranking.csv"],
    ),
    Stage(
        key="tests",
        title="10. Run Tests",
        command="python -m pytest tests -q --basetemp .pytest_tmp_run",
        purpose="Run the automated checks for pipeline behaviours and data-processing logic.",
        inputs=["tests/"],
        outputs=["Terminal test result", ".pytest_tmp_run/"],
        notes=[
            "Use this after code changes.",
            "The temporary test output folder avoids interfering with real pipeline artifacts.",
        ],
        status_paths=["tests"],
    ),
]


def path_status(path: str) -> tuple[bool, str]:
    p = Path(path)
    if p.suffix == ".parquet":
        meta = get_parquet_metadata(path)
        if meta.get("exists"):
            return True, f"{meta.get('rows', 0):,} rows"
        return False, "missing"
    if p.exists():
        if p.is_dir():
            return True, "directory exists"
        return True, f"{p.stat().st_size:,} bytes"
    return False, "missing"


def stage_ready(stage: Stage) -> bool:
    return all(path_status(path)[0] for path in stage.status_paths)


def show_stage(stage: Stage) -> None:
    ready = stage_ready(stage)
    st.markdown(f"## {stage.title}")
    st.caption("Ready" if ready else "Not complete yet")

    st.markdown("### Command")
    st.code(stage.command, language="powershell")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Purpose")
        st.write(stage.purpose)

        st.markdown("### Inputs")
        for item in stage.inputs:
            st.markdown(f"- `{item}`")

    with c2:
        st.markdown("### Outputs")
        for item in stage.outputs:
            st.markdown(f"- `{item}`")

        st.markdown("### Notes")
        for note in stage.notes:
            st.markdown(f"- {note}")

    st.markdown("### Artifact Status")
    rows = []
    for path in stage.status_paths:
        exists, detail = path_status(path)
        rows.append({"artifact": path, "status": "ready" if exists else "missing", "detail": detail})
    st.dataframe(rows, width="stretch", hide_index=True)

    if stage.key == "merge":
        report = load_json("logs/matching_regression_report.json")
        if report:
            st.markdown("### Current Merge Summary")
            st.json(
                {
                    "merged_rows_total": report.get("merged_rows_total"),
                    "merged_source_counts": report.get("merged_source_counts"),
                    "bts_retention_percentage": report.get("merge_candidate_metrics", {})
                    .get("bts", {})
                    .get("retention_percentage"),
                }
            )

    if stage.key == "label":
        labels = load_json("logs/label_distribution.json")
        if labels:
            st.markdown("### Current Label Distribution")
            st.json(labels)

    if stage.key == "weather":
        weather = load_json("logs/weather_coverage.json")
        if weather:
            st.markdown("### Current Weather Coverage")
            st.json(weather)

    if stage.key == "train":
        comparison = load_json("models/model_comparison.json")
        if comparison:
            st.markdown("### Current Model Summary")
            st.json(
                {
                    "best_model": comparison.get("best_model"),
                    "ranking": comparison.get("ranking", [])[:3],
                }
            )


st.markdown("# Pipeline Runbook")
st.write(
    "Click a stage in the flow to see the exact command, what it does, and which artifacts it should produce."
)

if "selected_stage" not in st.session_state:
    st.session_state.selected_stage = STAGES[0].key

st.markdown("## Interactive Flow")
flow_cols = st.columns(5)
for idx, stage in enumerate(STAGES):
    with flow_cols[idx % 5]:
        ready = stage_ready(stage)
        label = f"{stage.title}\n{'Ready' if ready else 'Missing'}"
        if st.button(label, key=f"flow_{stage.key}", use_container_width=True):
            st.session_state.selected_stage = stage.key

st.caption("Recommended order: plan -> download -> features -> merge -> weather -> label -> validate -> train -> reports -> tests")
st.divider()

selected = next(stage for stage in STAGES if stage.key == st.session_state.selected_stage)
show_stage(selected)

