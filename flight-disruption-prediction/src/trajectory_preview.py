"""Route map and altitude-profile preview generation."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir
from src.visualisation import TrajectoryVisualiser

logger = logging.getLogger(__name__)


def generate_trajectory_preview(
    *,
    sketches_path: str | Path,
    features_path: str | Path | None = None,
    out_dir: str | Path = "outputs/trajectory_previews",
    trajectory_id: str | None = None,
) -> dict:
    """Save a Folium route map and matplotlib altitude chart.

    Returns a small status dictionary suitable for notebook/report display.
    """

    sketches_path = Path(sketches_path)
    features_path = Path(features_path) if features_path else None
    out = Path(out_dir)
    ensure_dir(out)

    if not sketches_path.exists():
        return {
            "status": "missing_sketches",
            "message": f"Trajectory sketches not found at {sketches_path}. Run `python main.py --stage features --force` first.",
            "sketches_path": str(sketches_path),
        }

    try:
        sketches = pd.read_parquet(sketches_path)
    except Exception as exc:
        logger.warning("Could not read trajectory sketches from %s: %s", sketches_path, exc)
        return {
            "status": "invalid_sketches",
            "message": (
                f"Trajectory sketches exist but could not be read as Parquet: {sketches_path}. "
                "Regenerate them with `python main.py --stage features --force`."
            ),
            "sketches_path": str(sketches_path),
            "error": str(exc),
        }
    if sketches.empty:
        return {
            "status": "empty_sketches",
            "message": f"Trajectory sketches are empty: {sketches_path}",
            "sketches_path": str(sketches_path),
        }

    try:
        features = pd.read_parquet(features_path) if features_path and features_path.exists() else pd.DataFrame()
    except Exception as exc:
        logger.warning("Could not read trajectory features from %s: %s", features_path, exc)
        features = pd.DataFrame()
    selected_id = trajectory_id or select_preview_trajectory(sketches, features)
    if not selected_id:
        return {
            "status": "no_trajectory",
            "message": "No trajectory_id available for preview.",
            "sketches_path": str(sketches_path),
        }

    traj = sketches[sketches["trajectory_id"].astype(str) == str(selected_id)].copy()
    if traj.empty:
        return {
            "status": "not_found",
            "trajectory_id": selected_id,
            "message": f"trajectory_id not found in sketches: {selected_id}",
            "sketches_path": str(sketches_path),
        }

    traj = traj.sort_values(["point_idx", "timestamp"], kind="stable").reset_index(drop=True)
    safe_id = _safe_filename(str(selected_id))
    map_path = out / f"{safe_id}_route_map.html"
    altitude_path = out / f"{safe_id}_altitude_profile.png"
    points_path = out / f"{safe_id}_sketch_points.csv"

    m = TrajectoryVisualiser.generate_interactive_map_with_projections(
        traj,
        trajectory_ids=[str(selected_id)],
        gap_threshold_seconds=20 * 60,
    )
    m.save(str(map_path))
    save_altitude_profile(traj, altitude_path, title=f"Altitude Profile: {selected_id}")
    traj.to_csv(points_path, index=False)

    return {
        "status": "success",
        "trajectory_id": str(selected_id),
        "point_count": int(len(traj)),
        "map_html": str(map_path),
        "altitude_png": str(altitude_path),
        "points_csv": str(points_path),
    }


def select_preview_trajectory(sketches: pd.DataFrame, features: pd.DataFrame | None = None) -> str | None:
    """Choose a good preview trajectory using quality metadata when available."""

    if features is not None and not features.empty and "trajectory_id" in features.columns:
        df = features.copy()
        if "trajectory_quality_score" not in df.columns:
            df["trajectory_quality_score"] = 0.0
        if "is_full_flight" not in df.columns:
            df["is_full_flight"] = False
        df["_sketch_points"] = df["trajectory_id"].astype(str).map(
            sketches["trajectory_id"].astype(str).value_counts()
        ).fillna(0)
        df = df[df["_sketch_points"] >= 2].copy()
        if not df.empty:
            df = df.sort_values(
                ["is_full_flight", "trajectory_quality_score", "_sketch_points"],
                ascending=[False, False, False],
            )
            return str(df.iloc[0]["trajectory_id"])

    if "trajectory_id" not in sketches.columns:
        return None
    counts = sketches["trajectory_id"].astype(str).value_counts()
    if counts.empty:
        return None
    return str(counts.index[0])


def save_altitude_profile(df: pd.DataFrame, path: str | Path, *, title: str) -> Path:
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot = df.copy()
    if "elapsed_minutes" not in plot.columns:
        ts = pd.to_numeric(plot.get("timestamp"), errors="coerce")
        plot["elapsed_minutes"] = (ts - ts.min()) / 60.0
    if "altitude" not in plot.columns:
        plot["altitude"] = pd.NA
    plot["altitude"] = pd.to_numeric(plot["altitude"], errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(plot["elapsed_minutes"], plot["altitude"], marker="o", linewidth=2.0, markersize=4, color="#1F6F8B")
    ax.set_title(title, fontsize=15, weight="bold")
    ax.set_xlabel("Elapsed minutes")
    ax.set_ylabel("Altitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:140]
