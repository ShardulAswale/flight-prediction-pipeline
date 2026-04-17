import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.schemas import ML_DATASET_COLUMN_ORDER

logger = logging.getLogger(__name__)


class LabelGenerator:
    """Generates ground-truth labels for flight disruption prediction."""

    # T13: Columns that must be stripped before ML training to prevent target leakage
    LEAKY_COLUMNS = ["actual_dep", "actual_arr", "delay_minutes"]

    def __init__(self, delay_threshold_minutes: int = 15, log_dir: str = "logs"):
        self.delay_threshold = delay_threshold_minutes
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _compute_delay(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prefer BTS DepDelay when available; otherwise fall back to actual-vs-scheduled
        departure or arrival timestamps.
        """
        out_df = df.copy()

        if "DepDelay" in out_df.columns:
            out_df["delay_minutes"] = pd.to_numeric(out_df["DepDelay"], errors="coerce")
            return out_df

        has_dep = "actual_dep" in out_df.columns and "scheduled_dep" in out_df.columns
        has_arr = "actual_arr" in out_df.columns and "scheduled_arr" in out_df.columns

        if not has_dep and not has_arr:
            logger.warning("Neither departure nor arrival time pairs found. Cannot compute delay.")
            out_df["delay_minutes"] = np.nan
            return out_df

        if has_dep:
            actual_dep = pd.to_datetime(out_df["actual_dep"], utc=True, errors="coerce")
            scheduled_dep = pd.to_datetime(out_df["scheduled_dep"], utc=True, errors="coerce")
            out_df["delay_minutes"] = (actual_dep - scheduled_dep).dt.total_seconds().div(60)
            return out_df

        actual_arr = pd.to_datetime(out_df["actual_arr"], utc=True, errors="coerce")
        scheduled_arr = pd.to_datetime(out_df["scheduled_arr"], utc=True, errors="coerce")
        out_df["delay_minutes"] = (actual_arr - scheduled_arr).dt.total_seconds().div(60)
        return out_df

    @staticmethod
    def _get_cancelled_mask(df: pd.DataFrame) -> pd.Series:
        if "Cancelled" in df.columns:
            cancelled = pd.to_numeric(df["Cancelled"], errors="coerce").fillna(0)
        elif "cancelled" in df.columns:
            cancelled = pd.to_numeric(df["cancelled"], errors="coerce").fillna(0)
        else:
            cancelled = pd.Series(0, index=df.index, dtype="int64")

        return cancelled.eq(1)

    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate labels:
        - Cancelled == 1 -> Cancelled
        - delay_minutes > 15 -> Late
        - otherwise -> Normal
        """
        if df.empty:
            return df

        df = self._compute_delay(df)
        df["delay_minutes"] = pd.to_numeric(df.get("delay_minutes"), errors="coerce")
        df["label"] = "Normal"

        if "label_source" not in df.columns:
            df["label_source"] = "bts_verified"

        unverified_mask = df["label_source"].astype("string").eq("unverified_euro")
        df.loc[unverified_mask, "label"] = "Unverified"
        df.loc[unverified_mask, "delay_minutes"] = np.nan

        late_mask = (df["delay_minutes"] > self.delay_threshold) & ~unverified_mask
        df.loc[late_mask, "label"] = "Late"

        cancelled_mask = self._get_cancelled_mask(df)
        df.loc[cancelled_mask & ~unverified_mask, "label"] = "Cancelled"

        if "cancelled" not in df.columns:
            df["cancelled"] = cancelled_mask.astype("int64")
        df["label"] = df["label"].fillna("Normal")

        try:
            dist = df["label"].value_counts()
            total = len(df)
            distribution = {
                str(label_value): {
                    "count": int(count),
                    "pct": round(count / total * 100, 2) if total > 0 else 0,
                }
                for label_value, count in dist.items()
            }

            report_path = self.log_dir / "label_distribution.json"
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(distribution, handle, indent=4)

            logger.info("Label distribution saved to %s", report_path)
            logger.info("Label distribution: %s", dist.to_dict())
            if not df["delay_minutes"].dropna().empty:
                logger.info(
                    "Delay minutes range: min=%.2f max=%.2f",
                    df["delay_minutes"].min(),
                    df["delay_minutes"].max(),
                )
        except Exception as exc:
            logger.warning("Failed to save label distribution: %s", exc)

        return df

    def strip_leaky_columns(self, df: pd.DataFrame, keep_target: bool = True) -> pd.DataFrame:
        """Remove columns that would cause target leakage in ML training."""
        cols_to_drop = ["actual_dep", "actual_arr"]
        if not keep_target:
            cols_to_drop.append("delay_minutes")

        dropped = [column for column in cols_to_drop if column in df.columns]
        if dropped:
            logger.info("Target leakage guard: dropping columns %s", dropped)
            df = df.drop(columns=dropped)

        return df

    def standardize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure final dataset has standard columns."""
        df = self.strip_leaky_columns(df, keep_target=True)
        existing_cols = set(df.columns)
        final_cols = []
        for column in ML_DATASET_COLUMN_ORDER:
            if column in existing_cols:
                final_cols.append(column)
        return df[final_cols].copy()
