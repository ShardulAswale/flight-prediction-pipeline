"""Generate QC summaries and plots for converted OpenSky Parquet data."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


logger = logging.getLogger(__name__)


def load_date_parquet(date: str, parquet_root: str = "data/parquet/opensky_state_vectors") -> pd.DataFrame:
    """Load all parquet partitions for a date."""
    date_dir = Path(parquet_root) / date
    if not date_dir.exists():
        raise FileNotFoundError(f"Parquet date directory not found: {date_dir}")

    files = sorted(date_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {date_dir}")

    frames = [pd.read_parquet(path) for path in files]
    df = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d parquet rows from %d files", len(df), len(files))
    return df


def write_qc_summary(df: pd.DataFrame, date: str, out_dir: str = "outputs/qc") -> Path:
    """Write compact QC summary CSV including missingness metrics."""
    summary_rows = [
        {"metric": "rows", "value": int(len(df))},
        {
            "metric": "unique_icao24",
            "value": int(df["icao24"].nunique(dropna=True)) if "icao24" in df.columns else 0,
        },
        {
            "metric": "unique_callsign",
            "value": int(df["callsign"].nunique(dropna=True)) if "callsign" in df.columns else 0,
        },
    ]

    for column in df.columns:
        missing_pct = float(df[column].isna().mean() * 100.0)
        summary_rows.append({"metric": f"missing_pct__{column}", "value": round(missing_pct, 4)})

    out_path = Path(out_dir) / f"qc_summary_{date}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    logger.info("Wrote QC summary: %s", out_path)
    return out_path


def _save_hist(df: pd.DataFrame, column: str, out_path: Path, bins: int = 50) -> None:
    """Save histogram for a numeric column."""
    plt.figure(figsize=(8, 5))
    values = pd.to_numeric(df[column], errors="coerce") if column in df.columns else pd.Series(dtype=float)
    values = values.dropna()
    if values.empty:
        plt.text(0.5, 0.5, f"No data for {column}", ha="center", va="center")
    else:
        plt.hist(values, bins=bins)
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Wrote plot: %s", out_path)


def _save_density_scatter(df: pd.DataFrame, out_path: Path, sample_size: int = 100_000) -> None:
    """Save a simple sampled lat/lon density scatter plot."""
    plt.figure(figsize=(10, 6))

    if "lat" not in df.columns or "lon" not in df.columns:
        plt.text(0.5, 0.5, "Missing lat/lon columns", ha="center", va="center")
    else:
        coords = df[["lon", "lat"]].dropna()
        if len(coords) > sample_size:
            coords = coords.sample(sample_size, random_state=42)
        if coords.empty:
            plt.text(0.5, 0.5, "No valid lat/lon rows", ha="center", va="center")
        else:
            plt.hexbin(coords["lon"], coords["lat"], gridsize=200, mincnt=1)
            plt.colorbar(label="Density")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")

    plt.title("OpenSky State-Vector Spatial Density")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Wrote plot: %s", out_path)


def generate_qc_outputs(date: str, parquet_root: str = "data/parquet/opensky_state_vectors", out_dir: str = "outputs/qc") -> None:
    """Generate QC CSV and image artefacts for a date."""
    df = load_date_parquet(date=date, parquet_root=parquet_root)
    write_qc_summary(df=df, date=date, out_dir=out_dir)

    _save_hist(df, "velocity", Path(out_dir) / f"qc_hist_velocity_{date}.png")
    _save_hist(df, "baroaltitude", Path(out_dir) / f"qc_hist_altitude_{date}.png")
    _save_density_scatter(df, Path(out_dir) / f"qc_map_density_{date}.png")


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for QC utility."""
    parser = argparse.ArgumentParser(description="Run QC checks/plots on OpenSky parquet data.")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD")
    parser.add_argument("--parquet-root", default="data/parquet/opensky_state_vectors")
    parser.add_argument("--out-dir", default="outputs/qc")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    generate_qc_outputs(date=args.date, parquet_root=args.parquet_root, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
