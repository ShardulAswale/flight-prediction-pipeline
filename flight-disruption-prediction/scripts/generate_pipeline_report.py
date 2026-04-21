"""Generate saved visual/report artifacts for the whole pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_reporting import (
    configure_pretty_logging,
    output_dir,
    save_bar_chart,
    save_dataframe,
    save_dataset_inventory,
    save_missingness_chart,
    save_pipeline_flow_diagram,
    save_table_image,
)
from src.trajectory_preview import generate_trajectory_preview
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create saved tables/charts for pipeline review.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "config.yaml"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs" / "pipeline_report"))
    return parser.parse_args()


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def main() -> None:
    args = parse_args()
    configure_pretty_logging(PROJECT_ROOT / "logs", "pipeline_report")
    config = load_config(args.config)
    paths = config["paths"]
    processed = PROJECT_ROOT / paths.get("processed_data_dir", "data/processed")
    out = output_dir(root=args.out_dir)

    dataset_paths = {
        "adsb_combined": processed / "adsb_combined.parquet",
        "bts_combined": processed / "bts_combined.parquet",
        "eurocontrol_combined": processed / "eurocontrol_combined.parquet",
        "trajectory_features": processed / paths.get("features_file", "trajectory_features.parquet"),
        "trajectory_sketches": processed / paths.get("trajectory_sketches_file", "trajectory_sketches.parquet"),
        "ml_dataset_merged": processed / "ml_dataset_merged.parquet",
        "ml_dataset_weather": processed / "ml_dataset_weather.parquet",
        "ml_dataset_labeled": processed / "ml_dataset_labeled.parquet",
        "ml_dataset": processed / paths.get("ml_dataset_file", "ml_dataset.parquet"),
    }
    inventory = save_dataset_inventory(dataset_paths, out)
    preview_result = generate_trajectory_preview(
        sketches_path=dataset_paths["trajectory_sketches"],
        features_path=dataset_paths["trajectory_features"],
        out_dir=out / "trajectory_preview",
    )
    save_dataframe(pd.DataFrame([preview_result]), "trajectory_preview_status", out)

    save_pipeline_flow_diagram(
        [
            "Monthly Ingestion",
            "Schedule Windows",
            "ADS-B Filter",
            "Downsample",
            "Trajectory Sketches",
            "Feature Vectors",
            "En-route Weather",
            "Merge Labels",
            "Weather",
            "Train",
            "Explain",
        ],
        "pipeline_data_flow",
        out,
        title="Flight Disruption Prediction Data Flow",
    )

    features = _safe_read(dataset_paths["trajectory_features"])
    if features is not None and not features.empty:
        if "trajectory_quality_status" in features.columns:
            quality = features["trajectory_quality_status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
            save_dataframe(quality, "trajectory_quality_distribution", out)
            save_table_image(quality, "trajectory_quality_distribution", out, title="Trajectory Quality Distribution")
            save_bar_chart(quality, "status", "count", "trajectory_quality_distribution", out, title="Trajectory Quality Status Counts", color="#2D6A8E")

        numeric_summary_cols = [
            c for c in ["num_points", "raw_window_points", "flight_duration", "trajectory_length", "route_coverage_fraction", "trajectory_quality_score"]
            if c in features.columns
        ]
        if numeric_summary_cols:
            summary = features[numeric_summary_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T.reset_index(names="feature")
            save_dataframe(summary, "trajectory_feature_summary", out)
            save_table_image(summary, "trajectory_feature_summary", out, title="Trajectory Feature Summary", max_rows=30)

    ml = _safe_read(dataset_paths["ml_dataset"])
    if ml is not None and not ml.empty:
        if "label" in ml.columns:
            labels = ml["label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
            save_dataframe(labels, "label_distribution", out)
            save_bar_chart(labels, "label", "count", "label_distribution", out, title="Final Label Distribution", color="#785EF0")

        if "scheduled_dep" in ml.columns:
            months = (
                pd.to_datetime(ml["scheduled_dep"], utc=True, errors="coerce")
                .dt.to_period("M")
                .astype(str)
                .value_counts()
                .sort_index()
                .rename_axis("month")
                .reset_index(name="flights")
            )
            save_dataframe(months, "monthly_final_flight_counts", out)
            if not months.empty:
                save_bar_chart(months, "month", "flights", "monthly_final_flight_counts", out, title="Final Matched Flights by Month", color="#3A7D44")

        coverage_cols = [
            "wind_speed",
            "visibility",
            "temperature",
            "precipitation",
            "weather_severity",
            "wind_speed_dest",
            "visibility_dest",
            "temperature_dest",
            "precipitation_dest",
            "weather_severity_dest",
            "enroute_weather_coverage_ratio",
            "enroute_temperature_mean",
            "enroute_wind_speed_mean",
            "enroute_precipitation_max",
            "enroute_weather_severity_max",
        ]
        coverage = pd.DataFrame(
            [
                {"column": c, "non_null_pct": round((1 - ml[c].isna().mean()) * 100, 2)}
                for c in coverage_cols
                if c in ml.columns
            ]
        )
        if not coverage.empty:
            save_dataframe(coverage, "weather_coverage", out)
            save_bar_chart(coverage, "column", "non_null_pct", "weather_coverage", out, title="Weather Feature Coverage", ylabel="Non-null %", color="#DD6B20")

        save_missingness_chart(ml, "ml_dataset_missingness", out, top_n=35)

    comparison_path = PROJECT_ROOT / "logs" / "model_comparison.json"
    if comparison_path.exists():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        ranking = pd.DataFrame(comparison.get("ranking", []))
        if not ranking.empty:
            keep_cols = [c for c in ["model", "accuracy", "weighted_f1", "macro_f1", "cohens_kappa"] if c in ranking.columns]
            ranking_view = ranking[keep_cols]
            save_dataframe(ranking_view, "model_ranking", out)
            save_table_image(ranking_view, "model_ranking", out, title="Model Ranking")
            save_bar_chart(ranking_view, "model", "macro_f1", "model_macro_f1", out, title="Model Macro-F1", color="#005F73")

    importance_path = PROJECT_ROOT / "models" / "feature_importance.json"
    if importance_path.exists():
        importance = json.loads(importance_path.read_text(encoding="utf-8"))
        importance_df = (
            pd.DataFrame([{"feature": k, "importance": v} for k, v in importance.items()])
            .sort_values("importance", ascending=False)
            .head(30)
        )
        save_dataframe(importance_df, "top_feature_importance", out)
        save_bar_chart(
            importance_df.sort_values("importance"),
            "feature",
            "importance",
            "top_feature_importance",
            out,
            title="Top Feature Importances",
            horizontal=True,
            color="#2A9D8F",
        )

    print(f"Pipeline report saved to {out}")
    print(inventory.to_string(index=False))


if __name__ == "__main__":
    main()
