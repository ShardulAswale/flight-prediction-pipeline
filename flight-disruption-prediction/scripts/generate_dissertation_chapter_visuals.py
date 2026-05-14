"""Generate dissertation chapter visuals that are not produced by the pipeline.

The figures/tables here are explanatory dissertation assets. They do not change
pipeline data, model artifacts, or notebook outputs.
"""

from __future__ import annotations

import shutil
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs" / "chapter_visuals"
IMPORTANT_DIR = PROJECT_ROOT / "important outputs" / "chapter_visuals"


def wrap_text(value: object, width: int = 28) -> str:
    text = "" if pd.isna(value) else str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def save_table_image(
    df: pd.DataFrame,
    path: Path,
    *,
    title: str,
    column_widths: list[float] | None = None,
    font_size: int = 8,
    wrap_width: int = 28,
) -> None:
    preview = df.copy()
    for col in preview.columns:
        preview[col] = preview[col].map(lambda value: wrap_text(value, wrap_width))

    line_counts = [
        max(str(value).count("\n") + 1 for value in row)
        for _, row in preview.iterrows()
    ]
    fig_h = max(4.0, 1.2 + 0.34 * sum(line_counts) + 0.35 * len(preview))
    fig_w = 15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=15, weight="bold", pad=14)

    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        colWidths=column_widths,
        bbox=[0.0, 0.0, 1.0, 0.94],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1, 1.45)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        if row_idx == 0:
            cell.set_facecolor("#17324D")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#F8FAFC" if row_idx % 2 == 0 else "white")
        if col_idx == 0 and row_idx > 0:
            cell.set_text_props(weight="bold")

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_markdown_table(df: pd.DataFrame, path: Path) -> None:
    """Write a simple GitHub-style markdown table without optional dependencies."""
    text_df = df.fillna("").astype(str)
    headers = list(text_df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text_df.iterrows():
        values = [str(row[col]).replace("\n", " ").replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_box(ax, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        facecolor=color,
        edgecolor="#17324D",
        linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
        wrap=True,
    )


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="#17324D",
        )
    )


def figure_study_framework() -> None:
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 1.16)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Study Framework Overview", fontsize=18, weight="bold", pad=20)

    source_y = [0.72, 0.50, 0.28]
    sources = [
        ("OpenSky ADS-B\nstate vectors", "#DBEAFE"),
        ("BTS schedules\nand outcomes", "#DCFCE7"),
        ("Airport METAR\nweather", "#FEF3C7"),
    ]
    for y, (label, color) in zip(source_y, sources):
        draw_box(ax, (0.04, y), 0.18, 0.13, label, color)
        draw_arrow(ax, (0.22, y + 0.065), (0.33, 0.50))

    draw_box(
        ax,
        (0.33, 0.36),
        0.20,
        0.28,
        "Pipeline\ningestion -> cleaning\nschedule-aware ADS-B filtering\nweather merge",
        "#E0F2FE",
    )
    draw_arrow(ax, (0.53, 0.50), (0.61, 0.50))

    draw_box(
        ax,
        (0.61, 0.36),
        0.16,
        0.28,
        "Feature vectors\ntrajectory\nweather\ntime\nroute\ncongestion",
        "#FCE7F3",
    )
    draw_arrow(ax, (0.77, 0.50), (0.83, 0.68))
    draw_arrow(ax, (0.77, 0.50), (0.83, 0.50))
    draw_arrow(ax, (0.77, 0.50), (0.83, 0.32))

    models = [
        ("Logistic\nregression", 0.68),
        ("Random\nforest", 0.50),
        ("XGBoost", 0.32),
    ]
    for label, y in models:
        draw_box(ax, (0.83, y - 0.055), 0.12, 0.11, label, "#EDE9FE")
        draw_arrow(ax, (0.95, y), (0.985, y))

    ax.text(
        1.03,
        0.50,
        "Outputs\nraw probabilities\nbinary disruption risk\nmodel comparison\nSHAP explanations",
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F1F5F9", edgecolor="#17324D"),
    )

    for suffix in ["png", "svg"]:
        fig.savefig(OUT_DIR / f"figure_1_1_study_framework_overview.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def figure_adsb_vs_schedule() -> None:
    sketch_path = PROJECT_ROOT / "outputs" / "trajectory_previews" / "NKS993_202205301115_NK99320220530_sketch_points.csv"
    sketch = pd.read_csv(sketch_path)
    first = sketch.iloc[0]

    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.25, 1.25], height_ratios=[1, 1], wspace=0.35, hspace=0.38)
    ax_schedule = fig.add_subplot(gs[:, 0])
    ax_route = fig.add_subplot(gs[0, 1:])
    ax_alt = fig.add_subplot(gs[1, 1:])

    ax_schedule.axis("off")
    ax_schedule.set_title("Schedule data", fontsize=15, weight="bold", pad=12)
    schedule_rows = [
        ("flight_key", first.get("flight_key")),
        ("callsign", first.get("callsign")),
        ("origin", first.get("origin")),
        ("destination", first.get("destination")),
        ("scheduled_dep", first.get("scheduled_dep")),
        ("scheduled_arr", first.get("scheduled_arr")),
    ]
    y = 0.88
    for key, value in schedule_rows:
        ax_schedule.text(0.05, y, key, fontsize=10, weight="bold", color="#17324D")
        ax_schedule.text(0.05, y - 0.06, str(value), fontsize=10)
        y -= 0.14
    ax_schedule.text(
        0.05,
        0.03,
        "Schedule data gives planned context.\nADS-B adds actual route, speed,\naltitude, gaps and holding behaviour.",
        fontsize=10,
        wrap=True,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#DCFCE7", edgecolor="#4D7C0F"),
    )

    route = sketch.dropna(subset=["longitude", "latitude"]).copy()
    ax_route.plot(route["longitude"], route["latitude"], color="#2563EB", linewidth=2)
    ax_route.scatter(route["longitude"].iloc[0], route["latitude"].iloc[0], color="#16A34A", s=70, label="start")
    ax_route.scatter(route["longitude"].iloc[-1], route["latitude"].iloc[-1], color="#DC2626", s=70, label="end")
    ax_route.set_title("ADS-B adds route geometry", fontsize=15, weight="bold")
    ax_route.set_xlabel("longitude")
    ax_route.set_ylabel("latitude")
    ax_route.grid(True, alpha=0.25)
    ax_route.legend(loc="best")

    alt = sketch.dropna(subset=["elapsed_minutes", "altitude"]).copy()
    ax_alt.plot(alt["elapsed_minutes"], alt["altitude"], color="#EA580C", linewidth=2)
    ax_alt.set_title("ADS-B adds altitude profile and trajectory shape", fontsize=15, weight="bold")
    ax_alt.set_xlabel("elapsed minutes")
    ax_alt.set_ylabel("altitude")
    ax_alt.grid(True, alpha=0.25)
    ax_alt.text(
        0.01,
        0.92,
        "Example uses retained sketch points from NKS993 / 2022-05-30.",
        transform=ax_alt.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF7ED", edgecolor="#EA580C"),
    )

    fig.suptitle("Schedule Data vs ADS-B Movement Data", fontsize=18, weight="bold", y=1.02)
    for suffix in ["png", "svg"]:
        fig.savefig(OUT_DIR / f"figure_1_2_adsb_vs_schedule_illustration.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def reviewed_studies_table() -> pd.DataFrame:
    rows = [
        {
            "study": "Rebollo and Balakrishnan",
            "year": 2014,
            "data_source": "US operational delay/network data",
            "model_type": "Random forest classification and regression",
            "target_variable": "Route-level departure delay threshold and delay value",
            "adsb_used": "No",
            "calibration_assessed": "No",
            "source_url": "https://www.mit.edu/~hamsa/pubs/RebolloBalakrishnanTRC2014.pdf",
        },
        {
            "study": "Patgiri, Hussain and Nongmeikapam",
            "year": 2020,
            "data_source": "Airline delay data plus weather variables",
            "model_type": "LR, Naive Bayes, KNN, decision tree, random forest",
            "target_variable": "Binary delay threshold",
            "adsb_used": "No",
            "calibration_assessed": "No",
            "source_url": "https://arxiv.org/abs/2002.10254",
        },
        {
            "study": "Zoutendijk and Mitici",
            "year": 2021,
            "data_source": "Airport schedules plus weather",
            "model_type": "Mixture density network and random forest regression",
            "target_variable": "Probabilistic delay distribution",
            "adsb_used": "No",
            "calibration_assessed": "Yes, probabilistic scoring via CRPS",
            "source_url": "https://www.mdpi.com/2226-4310/8/6/152",
        },
        {
            "study": "Kaewunruen, Sresakoolchai and Xiang",
            "year": 2021,
            "data_source": "Birmingham Airport operations and weather",
            "model_type": "RF, ANN, SVM and LR",
            "target_variable": "Flight punctuality / delay",
            "adsb_used": "No",
            "calibration_assessed": "No",
            "source_url": "https://www.mdpi.com/2225-1154/9/8/127",
        },
        {
            "study": "Shao et al.",
            "year": 2022,
            "data_source": "LAX airport situational awareness, trajectory, weather and schedules",
            "model_type": "Hand-crafted ML features and TrajCNN",
            "target_variable": "Departure delay regression",
            "adsb_used": "Trajectory/sensor movement data",
            "calibration_assessed": "No",
            "source_url": "https://doi.org/10.1016/j.neucom.2021.04.136",
        },
        {
            "study": "Sahfienya and Regan",
            "year": 2021,
            "data_source": "ADS-B trajectories at ATL",
            "model_type": "CNN-GRU and 3D-CNN with MC dropout",
            "target_variable": "4D trajectory prediction, not delay classification",
            "adsb_used": "Yes",
            "calibration_assessed": "Uncertainty considered, not delay calibration",
            "source_url": "https://arxiv.org/abs/2110.07774",
        },
        {
            "study": "Kilic and Sallan",
            "year": 2023,
            "data_source": "Public US flight and weather data",
            "model_type": "LR, RF, GBM and feed-forward neural network",
            "target_variable": "Arrival delay prediction",
            "adsb_used": "No",
            "calibration_assessed": "No",
            "source_url": "https://www.mdpi.com/2226-4310/10/4/342",
        },
        {
            "study": "Chaudhuri, Zhang and Zhang",
            "year": 2024,
            "data_source": "Weather, flight information and ADS-B real-time trajectory data",
            "model_type": "LSTM-attention classifier",
            "target_variable": "Binary and three-class arrival delay classification",
            "adsb_used": "Yes",
            "calibration_assessed": "No",
            "source_url": "https://sesar.eu/sites/default/files/documents/sid/2024/papers/SIDs_2024_paper_006%20final.pdf",
        },
        {
            "study": "Kowalczuk, Wszolek and Okuniewska",
            "year": 2025,
            "data_source": "ADS-B, METAR and TAF streams",
            "model_type": "Random forests and linear regression in distributed system",
            "target_variable": "Flight delay prediction",
            "adsb_used": "Yes",
            "calibration_assessed": "No",
            "source_url": "https://doi.org/10.1016/j.ifacol.2025.07.034",
        },
    ]
    return pd.DataFrame(rows)


def table_reviewed_studies() -> pd.DataFrame:
    df = reviewed_studies_table()
    df.to_csv(OUT_DIR / "table_2_1_reviewed_studies.csv", index=False)
    save_markdown_table(df, OUT_DIR / "table_2_1_reviewed_studies.md")
    image_df = df.drop(columns=["source_url"]).copy()
    image_df["study"] = image_df["study"].replace(
        {
            "Patgiri, Hussain and Nongmeikapam": "Patgiri et al.",
            "Kaewunruen, Sresakoolchai and Xiang": "Kaewunruen et al.",
            "Chaudhuri, Zhang and Zhang": "Chaudhuri et al.",
            "Kowalczuk, Wszolek and Okuniewska": "Kowalczuk et al.",
        }
    )
    image_df = image_df.rename(
        columns={
            "study": "Study",
            "year": "Year",
            "data_source": "Data source",
            "model_type": "Model type",
            "target_variable": "Target variable",
            "adsb_used": "ADS-B used",
            "calibration_assessed": "Calibration assessed",
        }
    )
    save_table_image(
        image_df,
        OUT_DIR / "table_2_1_reviewed_studies.png",
        title="Table 2.1 Summary of Reviewed Studies",
        column_widths=[0.15, 0.05, 0.21, 0.20, 0.17, 0.08, 0.14],
        font_size=7,
        wrap_width=26,
    )
    return df


def figure_research_gap(studies: pd.DataFrame) -> None:
    gap_rows = [
        ("Rebollo and Balakrishnan", [0, 0, 0, 0]),
        ("Patgiri et al.", [0, 0, 0, 0]),
        ("Zoutendijk and Mitici", [0, 0, 0, 1]),
        ("Kaewunruen et al.", [0, 0, 0, 0]),
        ("Shao et al.", [1, 0, 0, 0]),
        ("Sahfienya and Regan", [1, 0, 0, 0.5]),
        ("Kilic and Sallan", [0, 0, 0, 0]),
        ("Chaudhuri et al.", [1, 1, 0, 0]),
        ("Kowalczuk et al.", [1, 0, 0, 0]),
        ("This project", [1, 0.5, 1, 0]),
    ]
    columns = ["ADS-B /\ntrajectory", "multi-class /\nsubtype target", "imbalance\nhandling", "calibration /\nprobabilistic eval"]
    matrix = np.array([row[1] for row in gap_rows], dtype=float)
    labels = [row[0] for row in gap_rows]

    fig, ax = plt.subplots(figsize=(12, 7))
    cmap = ListedColormap(["#FEE2E2", "#FEF3C7", "#DCFCE7"])
    code_matrix = np.where(matrix == 1, 2, np.where(matrix == 0.5, 1, 0))
    ax.imshow(code_matrix, cmap=cmap, aspect="auto", vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(columns)), labels=columns)
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=9)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = "Yes" if matrix[i, j] == 1 else "Partial" if matrix[i, j] == 0.5 else "No"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, weight="bold", color="#111827")

    ax.set_title("Figure 2.1 Research Gap Matrix", fontsize=16, weight="bold", pad=16)
    ax.set_xlabel("Research gap dimensions")
    ax.set_ylabel("Reviewed study")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(OUT_DIR / f"figure_2_1_research_gap_matrix.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    gap_df = pd.DataFrame(matrix, index=labels, columns=[c.replace("\n", " ") for c in columns])
    gap_df.to_csv(OUT_DIR / "figure_2_1_research_gap_matrix.csv")


def table_feature_engineering_summary() -> pd.DataFrame:
    feature_list = pd.read_json(PROJECT_ROOT / "models" / "feature_list.json", typ="series").tolist()

    groups = {
        "Trajectory behaviour": {
            "features": [
                "flight_duration", "trajectory_length", "mean_speed", "max_speed", "speed_std",
                "mean_altitude", "altitude_variance", "vertical_rate_std", "heading_variability",
                "holding_pattern_count", "altitude_change_count", "unstable_descent_flag",
            ],
            "source": "OpenSky ADS-B pings inside BTS schedule windows",
            "purpose": "Capture movement, speed, altitude, heading and shape behaviour.",
        },
        "Trajectory quality": {
            "features": [
                "trajectory_quality_score", "dep_anchor_confidence", "arr_anchor_confidence",
                "middle_gap_count", "gap_fraction_of_flight", "max_inter_ping_seconds",
                "ping_interval_cv", "is_full_flight",
            ],
            "source": "ADS-B coverage diagnostics and schedule-window matching",
            "purpose": "Represent completeness, timing gaps and confidence in observed traces.",
        },
        "Temporal": {
            "features": ["dep_hour", "dep_day_of_week", "dep_month", "is_peak_hour"],
            "source": "BTS scheduled departure timestamp",
            "purpose": "Capture time-of-day, day-of-week and seasonal effects.",
        },
        "Airport and route context": {
            "features": ["origin_flight_count", "dest_flight_count", "route_gc_distance_km"],
            "source": "BTS route schedule and airport pair metadata",
            "purpose": "Represent route length and broad airport activity context.",
        },
        "Congestion": {
            "features": ["origin_flights_1hr", "dest_flights_1hr"],
            "source": "BTS schedule counts around departure/arrival windows",
            "purpose": "Approximate short-term demand pressure at origin and destination.",
        },
        "Weather": {
            "features": [
                "wind_speed", "visibility", "temperature", "precipitation", "weather_severity",
                "wind_speed_dest", "visibility_dest", "temperature_dest", "precipitation_dest",
                "weather_severity_dest",
            ],
            "source": "Airport METAR weather merged by nearest station/time",
            "purpose": "Capture origin and destination operating conditions.",
        },
    }

    rows = []
    used = set(feature_list)
    for group, meta in groups.items():
        present = [feature for feature in meta["features"] if feature in used]
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(present),
                "features": ", ".join(present),
                "data_source": meta["source"],
                "purpose": meta["purpose"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "table_3_1_feature_engineering_summary.csv", index=False)
    save_markdown_table(df, OUT_DIR / "table_3_1_feature_engineering_summary.md")
    image_df = df.rename(
        columns={
            "feature_group": "Feature group",
            "feature_count": "Count",
            "features": "Current features",
            "data_source": "Data source",
            "purpose": "Purpose",
        }
    )
    save_table_image(
        image_df,
        OUT_DIR / "table_3_1_feature_engineering_summary.png",
        title="Table 3.1 Feature Engineering Summary",
        column_widths=[0.16, 0.06, 0.36, 0.21, 0.21],
        font_size=7,
        wrap_width=42,
    )
    return df


def table_train_test_split_summary() -> pd.DataFrame:
    cols = ["scheduled_dep", "label", "label_binary", "training_eligible", "disruption_subtype"]
    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "ml_dataset.parquet", columns=cols)
    if "training_eligible" in df.columns:
        df = df[df["training_eligible"] == True].copy()
    df = df[df["label"] != "Unverified"].copy()

    if "label_binary" in df.columns:
        target = df["label_binary"].copy()
        fallback = df["label"].where(df["label"].eq("Normal"), "Disrupted")
        target = target.where(target.isin(["Normal", "Disrupted"]), fallback)
    else:
        target = df["label"].where(df["label"].eq("Normal"), "Disrupted")

    dep_dt = pd.to_datetime(df["scheduled_dep"], utc=True, errors="coerce")
    dep_dates = dep_dt.dt.normalize()
    unique_days = sorted(dep_dates.dropna().unique())
    test_start = pd.Timestamp(unique_days[-7])
    if test_start.tzinfo is None:
        test_start = test_start.tz_localize("UTC")

    masks = {
        "Train": dep_dates < test_start,
        "Test": dep_dates >= test_start,
        "Total": pd.Series(True, index=df.index),
    }

    rows = []
    for split, mask in masks.items():
        sub = df[mask].copy()
        sub_target = target[mask]
        original_counts = sub["label"].value_counts()
        binary_counts = sub_target.value_counts()
        rows.append(
            {
                "split": split,
                "date_range": f"{dep_dt[mask].min():%Y-%m-%d} to {dep_dt[mask].max():%Y-%m-%d}",
                "records": int(len(sub)),
                "normal": int(binary_counts.get("Normal", 0)),
                "disrupted": int(binary_counts.get("Disrupted", 0)),
                "late_original": int(original_counts.get("Late", 0)),
                "cancelled_original": int(original_counts.get("Cancelled", 0)),
                "disrupted_pct": round(float(binary_counts.get("Disrupted", 0) / max(len(sub), 1) * 100), 2),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "table_3_2_train_test_split_summary.csv", index=False)
    save_markdown_table(summary, OUT_DIR / "table_3_2_train_test_split_summary.md")
    image_summary = summary.rename(
        columns={
            "split": "Split",
            "date_range": "Date range",
            "records": "Records",
            "normal": "Normal",
            "disrupted": "Disrupted",
            "late_original": "Late",
            "cancelled_original": "Cancelled",
            "disrupted_pct": "Disrupted %",
        }
    )
    save_table_image(
        image_summary,
        OUT_DIR / "table_3_2_train_test_split_summary.png",
        title="Table 3.2 Train-Test Split Summary",
        column_widths=[0.08, 0.22, 0.10, 0.10, 0.10, 0.10, 0.12, 0.12],
        font_size=8,
        wrap_width=22,
    )
    return summary


def write_readme() -> None:
    readme = """# Dissertation Chapter Visuals

Generated assets that were missing from the normal pipeline reports.

## Chapter 1

- `figure_1_1_study_framework_overview.png/.svg`
- `figure_1_2_adsb_vs_schedule_illustration.png/.svg`

## Chapter 2

- `table_2_1_reviewed_studies.csv/.md/.png`
- `figure_2_1_research_gap_matrix.png/.svg/.csv`

## Chapter 3

- `table_3_1_feature_engineering_summary.csv/.md/.png`
- `table_3_2_train_test_split_summary.csv/.md/.png`

Notes:

- Existing Chapter 3 pipeline figures were not regenerated here.
- Table 3.2 is based on the same temporal split logic used by `src/model_training.py`: last seven observed scheduled-departure days are held out as the test window.
- Figure 1.2 uses the existing NKS993 trajectory preview sketch points from `outputs/trajectory_previews`.
- The research-gap figure is a dissertation synthesis visual; it should be checked against the exact studies cited in the final literature review before submission.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def copy_to_important_outputs() -> None:
    IMPORTANT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.iterdir():
        if path.is_file():
            shutil.copy2(path, IMPORTANT_DIR / path.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    figure_study_framework()
    figure_adsb_vs_schedule()
    studies = table_reviewed_studies()
    figure_research_gap(studies)
    table_feature_engineering_summary()
    table_train_test_split_summary()
    write_readme()
    copy_to_important_outputs()
    print(f"Saved chapter visuals to {OUT_DIR}")
    print(f"Copied chapter visuals to {IMPORTANT_DIR}")


if __name__ == "__main__":
    main()
