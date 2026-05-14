"""Reporting helpers for notebooks and pipeline monitoring.

The project produces a lot of intermediate data. These helpers keep notebooks
thin by centralising common output behaviour:

- pretty run logs with timings
- CSV/JSON/PNG outputs in ``outputs/``
- quick dataset inventory charts
- simple flow diagrams for explaining the data path
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def output_dir(*parts: str, root: str | Path = "outputs") -> Path:
    path = Path(root).joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _memory_mb() -> float | None:
    try:
        import psutil

        process = psutil.Process()
        return round(process.memory_info().rss / 1024**2, 2)
    except Exception:
        return None


@dataclass
class PerformanceLog:
    """Collect and persist start/end timing records for a workflow."""

    name: str
    out_dir: Path = field(default_factory=lambda: output_dir("run_logs"))
    records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self.name.replace(" ", "_").lower()
        self.csv_path = self.out_dir / f"{safe_name}_performance.csv"
        self.json_path = self.out_dir / f"{safe_name}_performance.json"

    @contextmanager
    def step(self, step_name: str, **metadata) -> Iterator[None]:
        start = datetime.now()
        start_perf = time.perf_counter()
        start_mem = _memory_mb()
        logger.info("START %-28s %s", step_name, metadata or "")
        status = "SUCCESS"
        error = None
        try:
            yield
        except Exception as exc:
            status = "FAILED"
            error = repr(exc)
            raise
        finally:
            end = datetime.now()
            duration = time.perf_counter() - start_perf
            record = {
                "workflow": self.name,
                "step": step_name,
                "status": status,
                "start_time": start.isoformat(timespec="seconds"),
                "end_time": end.isoformat(timespec="seconds"),
                "duration_seconds": round(duration, 3),
                "start_memory_mb": start_mem,
                "end_memory_mb": _memory_mb(),
                "error": error,
                **metadata,
            }
            self.records.append(record)
            self.save()
            logger.info(
                "END   %-28s status=%s duration=%.2fs",
                step_name,
                status,
                duration,
            )

    def save(self) -> None:
        if not self.records:
            return
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)
        self.json_path.write_text(json.dumps(self.records, indent=2, default=str))


def save_dataframe(
    df: pd.DataFrame,
    stem: str,
    out_dir: str | Path,
    *,
    index: bool = False,
    save_parquet: bool = False,
) -> dict[str, Path]:
    """Save a dataframe as CSV and optionally parquet."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"csv": out / f"{stem}.csv"}
    df.to_csv(paths["csv"], index=index)
    if save_parquet:
        paths["parquet"] = out / f"{stem}.parquet"
        df.to_parquet(paths["parquet"], index=index)
    return paths


def save_table_image(
    df: pd.DataFrame,
    stem: str,
    out_dir: str | Path,
    *,
    title: str | None = None,
    max_rows: int = 20,
    max_cols: int = 8,
) -> Path:
    """Render a compact dataframe preview as a PNG table."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    preview = df.head(max_rows).iloc[:, :max_cols].copy()
    preview = preview.astype(str)

    fig_h = max(2.5, 0.45 * (len(preview) + 2))
    fig_w = max(8, 1.4 * max(1, len(preview.columns)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, weight="bold", pad=12)
    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#27374D")
        elif row % 2 == 0:
            cell.set_facecolor("#F4F7FB")
    path = out / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _format_bar_value(value: float, context: str = "") -> str:
    """Format compact labels for chart bars."""
    if pd.isna(value):
        return ""

    value = float(value)
    context_l = context.lower()
    if "%" in context_l or "percent" in context_l or "pct" in context_l:
        return f"{value:.1f}%"
    if "size" in context_l and "mb" in context_l:
        return _format_size(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 10_000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    if 0 < abs(value) < 1:
        return f"{value:.3f}"
    if value.is_integer():
        return f"{value:,.0f}"
    return f"{value:.2f}"


def _annotate_bar_values(
    ax,
    *,
    horizontal: bool = False,
    formatter: Callable[[float], str] | None = None,
) -> None:
    """Add readable numeric labels to matplotlib bar charts."""
    formatter = formatter or (lambda value: _format_bar_value(value))
    for patch in ax.patches:
        if horizontal:
            value = patch.get_width()
            if pd.isna(value):
                continue
            ax.annotate(
                formatter(float(value)),
                xy=(value, patch.get_y() + patch.get_height() / 2),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=8,
            )
        else:
            value = patch.get_height()
            if pd.isna(value):
                continue
            ax.annotate(
                formatter(float(value)),
                xy=(patch.get_x() + patch.get_width() / 2, value),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )


def save_bar_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    stem: str,
    out_dir: str | Path,
    *,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    horizontal: bool = False,
    color: str = "#2D6A8E",
    value_labels: bool = True,
    value_formatter: Callable[[float], str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    values = pd.to_numeric(df[y], errors="coerce")
    if horizontal:
        ax.barh(df[x].astype(str), values, color=color)
        ax.set_xlabel(ylabel or y)
        ax.set_ylabel(xlabel or x)
        ax.margins(x=0.18)
    else:
        ax.bar(df[x].astype(str), values, color=color)
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel(ylabel or y)
        ax.tick_params(axis="x", rotation=35)
        ax.margins(y=0.15)
    ax.set_title(title, fontsize=15, weight="bold")
    ax.grid(axis="y" if not horizontal else "x", alpha=0.25)
    if value_labels:
        context = " ".join(part for part in [y, ylabel or ""] if part)
        formatter = value_formatter or (lambda value: _format_bar_value(value, context))
        _annotate_bar_values(ax, horizontal=horizontal, formatter=formatter)
    fig.tight_layout()
    path = out / f"{stem}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _format_size(size_mb: float) -> str:
    if pd.isna(size_mb):
        return ""
    if size_mb >= 1024:
        return f"{size_mb / 1024:.2f} GB"
    return f"{size_mb:.2f} MB"


def save_dataset_size_charts(inventory: pd.DataFrame, out_dir: str | Path) -> dict[str, Path]:
    """Save readable dataset-size charts for sources with very different scales."""
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chart_df = inventory[inventory["exists"]].copy()
    chart_df["size_mb"] = pd.to_numeric(chart_df["size_mb"], errors="coerce").fillna(0)
    chart_df = chart_df.sort_values("size_mb", ascending=False)

    paths: dict[str, Path] = {}
    if chart_df.empty:
        return paths

    size_table = chart_df[["dataset", "size_mb", "rows", "columns"]].copy()
    size_table["size"] = size_table["size_mb"].map(_format_size)
    size_table = size_table[["dataset", "size", "size_mb", "rows", "columns"]]
    save_dataframe(size_table, "dataset_sizes_table", out)
    paths["table"] = save_table_image(
        size_table,
        "dataset_sizes_table",
        out,
        title="Dataset Sizes",
        max_rows=30,
        max_cols=5,
    )

    processed_df = chart_df[~chart_df["dataset"].astype(str).eq("adsb_combined")].copy()

    fig, axes = plt.subplots(1, 2, figsize=(17, 6), gridspec_kw={"width_ratios": [1.1, 1]})
    ax_all, ax_processed = axes
    ax_all.bar(chart_df["dataset"].astype(str), chart_df["size_mb"], color="#3A7D44")
    ax_all.set_yscale("log")
    ax_all.set_xlabel("Dataset")
    ax_all.set_ylabel("Size (MB, log scale)")
    ax_all.set_title("All datasets, log scale", fontsize=13, weight="bold")
    ax_all.tick_params(axis="x", rotation=35)
    ax_all.grid(axis="y", alpha=0.25)
    ax_all.margins(y=0.18)
    _annotate_bar_values(ax_all, formatter=_format_size)

    if processed_df.empty:
        ax_processed.axis("off")
        ax_processed.text(0.5, 0.5, "No processed datasets found", ha="center", va="center")
    else:
        ax_processed.bar(processed_df["dataset"].astype(str), processed_df["size_mb"], color="#2D6A8E")
        ax_processed.set_xlabel("Dataset")
        ax_processed.set_ylabel("Size (MB)")
        ax_processed.set_title("Processed outputs, ADS-B excluded", fontsize=13, weight="bold")
        ax_processed.tick_params(axis="x", rotation=35)
        ax_processed.grid(axis="y", alpha=0.25)
        ax_processed.margins(y=0.18)
        _annotate_bar_values(ax_processed, formatter=_format_size)

    fig.suptitle("Dataset Sizes: Full Scale and Processed Outputs", fontsize=16, weight="bold", y=1.04)
    fig.tight_layout()
    paths["combined"] = out / "dataset_sizes.png"
    fig.savefig(paths["combined"], dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(chart_df["dataset"].astype(str), chart_df["size_mb"], color="#3A7D44")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Size (MB, log scale)")
    ax.set_title("Dataset Size by Source (Log Scale)", fontsize=15, weight="bold")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    for idx, (_, row) in enumerate(chart_df.iterrows()):
        ax.text(idx, max(float(row["size_mb"]), 1.0), _format_size(float(row["size_mb"])), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    paths["log"] = out / "dataset_sizes_log.png"
    fig.savefig(paths["log"], dpi=160, bbox_inches="tight")
    plt.close(fig)

    if not processed_df.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(processed_df["dataset"].astype(str), processed_df["size_mb"], color="#2D6A8E")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Size (MB)")
        ax.set_title("Processed Dataset Sizes (ADS-B Excluded)", fontsize=15, weight="bold")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        for idx, (_, row) in enumerate(processed_df.iterrows()):
            ax.text(idx, float(row["size_mb"]), _format_size(float(row["size_mb"])), ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        paths["processed_only"] = out / "dataset_sizes_processed_only.png"
        fig.savefig(paths["processed_only"], dpi=160, bbox_inches="tight")
        plt.close(fig)

    return paths


def save_missingness_chart(df: pd.DataFrame, stem: str, out_dir: str | Path, *, top_n: int = 30) -> Path:
    missing = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    missing.columns = ["column", "missing_pct"]
    save_dataframe(missing, f"{stem}_missingness", out_dir)
    return save_bar_chart(
        missing.sort_values("missing_pct"),
        "column",
        "missing_pct",
        stem,
        out_dir,
        title="Top Missingness by Column",
        xlabel="Column",
        ylabel="Missing %",
        horizontal=True,
        color="#B85042",
    )


def dataset_inventory(paths: Mapping[str, str | Path]) -> pd.DataFrame:
    """Return a quick inventory with rows, columns, size, and time range when possible."""
    import pyarrow.parquet as pq

    rows = []
    for name, raw_path in paths.items():
        path = Path(raw_path)
        record = {
            "dataset": name,
            "path": str(path),
            "exists": path.exists(),
            "size_mb": round(path.stat().st_size / 1024**2, 2) if path.exists() else np.nan,
            "rows": np.nan,
            "columns": np.nan,
            "min_time": None,
            "max_time": None,
        }
        if path.exists() and path.suffix.lower() == ".parquet":
            try:
                pf = pq.ParquetFile(path)
                record["rows"] = int(pf.metadata.num_rows)
                record["columns"] = len(pf.schema.names)
                time_col = next(
                    (c for c in ["scheduled_dep", "start_time_utc", "timestamp", "sample_date"] if c in pf.schema.names),
                    None,
                )
                if time_col:
                    if pf.metadata.num_row_groups > 1 and time_col in pf.schema.names:
                        idx = pf.schema.names.index(time_col)
                        mins, maxs = [], []
                        for i in range(pf.metadata.num_row_groups):
                            stats = pf.metadata.row_group(i).column(idx).statistics
                            if stats and stats.has_min_max:
                                mins.append(stats.min)
                                maxs.append(stats.max)
                        if mins:
                            record["min_time"] = str(min(mins))
                            record["max_time"] = str(max(maxs))
                    else:
                        s = pd.read_parquet(path, columns=[time_col])[time_col]
                        if time_col != "sample_date":
                            s = pd.to_datetime(s, utc=True, errors="coerce")
                        record["min_time"] = str(s.min())
                        record["max_time"] = str(s.max())
            except Exception as exc:
                record["error"] = repr(exc)
        rows.append(record)
    return pd.DataFrame(rows)


def save_dataset_inventory(paths: Mapping[str, str | Path], out_dir: str | Path) -> pd.DataFrame:
    inv = dataset_inventory(paths)
    save_dataframe(inv, "dataset_inventory", out_dir)
    save_table_image(inv, "dataset_inventory", out_dir, title="Dataset Inventory")
    if inv["exists"].any():
        save_dataset_size_charts(inv, out_dir)
    return inv


def save_pipeline_flow_diagram(
    steps: Sequence[str],
    stem: str,
    out_dir: str | Path,
    *,
    title: str = "Pipeline Data Flow",
) -> Path:
    """Create a simple left-to-right flow diagram as a PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = len(steps)
    fig_w = max(12, n * 2.1)
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, weight="bold", pad=18)
    for i, step in enumerate(steps):
        box = FancyBboxPatch(
            (i + 0.08, 0.36),
            0.82,
            0.28,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            facecolor="#EAF2F8",
            edgecolor="#2D6A8E",
            linewidth=1.5,
        )
        ax.add_patch(box)
        ax.text(i + 0.49, 0.5, step, ha="center", va="center", fontsize=9, weight="bold", wrap=True)
        if i < n - 1:
            arrow = FancyArrowPatch(
                (i + 0.9, 0.5),
                (i + 1.08, 0.5),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#2D6A8E",
            )
            ax.add_patch(arrow)
    path = out / f"{stem}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def configure_pretty_logging(log_dir: str | Path = "logs", name: str = "pipeline") -> Path:
    """Configure console + file logging for notebook/script runs."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    file_path = log_path / f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    return file_path
