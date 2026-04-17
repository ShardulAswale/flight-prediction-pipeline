import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MergeQA:
    """Quality Assurance reporting for dataset integration."""

    def __init__(self, log_dir: str = "logs/merge_qa"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        adsb_start_len: int,
        merged_df: pd.DataFrame,
        candidate_len: int,
        filtered_len: int,
        duplicates_dropped: int,
        source_name: str,
    ) -> dict:
        retention_pct = len(merged_df) / adsb_start_len if adsb_start_len > 0 else 0
        sample_size = min(10, len(merged_df))
        audit_sample = merged_df.sample(sample_size).to_dict(orient="records") if sample_size > 0 else []

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "source_dataset": source_name,
            "metrics": {
                "adsb_input_rows": adsb_start_len,
                "candidate_rows": candidate_len,
                "time_filtered_rows": filtered_len,
                "final_merged_rows": len(merged_df),
                "retention_percentage": retention_pct,
                "duplicates_resolved": duplicates_dropped,
            },
            "audit_sample": audit_sample,
        }

        report_file = self.log_dir / f"merge_qa_report_{source_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=4, default=str)

        logger.info(
            "Merge QA (%s): %s final rows from %s ADS-B features. Report saved to %s",
            source_name,
            len(merged_df),
            adsb_start_len,
            report_file,
        )
        return report


class DataMerger:
    """
    Hierarchical multi-anchor matcher between ADS-B features and schedule data.

    BTS matching uses:
    - harmonized callsigns
    - local service-day window (day-1/day/day+1)
    - minimum of departure and arrival anchor differences

    Eurocontrol matching uses:
    - exact service-day alignment
    - minimum of departure and arrival anchor differences
    - rows remain available for coverage but are marked unverified for labeling
    """

    INVALID_CALLSIGNS = {"", "0", "00000000", "NONE", "NAN", "NULL", "UNKNOWN"}
    BTS_AIRLINE_PREFIX_MAP = {
        "AA": "AAL",
        "AS": "ASA",
        "B6": "JBU",
        "DL": "DAL",
        "EV": "ASH",
        "F9": "FFT",
        "G4": "AAY",
        "HA": "HAL",
        "MQ": "ENY",
        "NK": "NKS",
        "OH": "JIA",
        "OO": "SKW",
        "QX": "QXE",
        "UA": "UAL",
        "WN": "SWA",
        "YV": "ASH",
        "YX": "RPA",
        "9E": "EDV",
    }

    def __init__(self, tolerance_hours: Union[int, dict] = 2):
        if isinstance(tolerance_hours, dict):
            self.tolerance_map = {key: pd.Timedelta(hours=value) for key, value in tolerance_hours.items()}
        else:
            self.tolerance_map = {"default": pd.Timedelta(hours=tolerance_hours)}
        self.qa = MergeQA()

    def _get_source_name(self, schedule_df: pd.DataFrame) -> str:
        if "source_dataset" in schedule_df.columns and not schedule_df["source_dataset"].dropna().empty:
            return str(schedule_df["source_dataset"].dropna().iloc[0]).strip().lower()
        return "unknown"

    def _get_tolerance(self, schedule_df: pd.DataFrame) -> pd.Timedelta:
        if "region" in schedule_df.columns and not schedule_df["region"].dropna().empty:
            region = schedule_df["region"].dropna().iloc[0]
            if region in self.tolerance_map:
                return self.tolerance_map[region]
        return self.tolerance_map.get("default", pd.Timedelta(hours=2))

    @staticmethod
    def _ensure_utc_ns(series: pd.Series, unit: str | None = None) -> pd.Series:
        if unit:
            normalized = pd.to_datetime(series, unit=unit, utc=True, errors="coerce")
        else:
            normalized = pd.to_datetime(series, utc=True, errors="coerce")
        return normalized.astype("datetime64[ns, UTC]")

    @staticmethod
    def _normalize_day_key(series: pd.Series) -> pd.Series:
        utc_series = pd.to_datetime(series, utc=True, errors="coerce")
        return utc_series.dt.tz_localize(None).dt.normalize()

    @staticmethod
    def _coerce_day_key(series: pd.Series) -> pd.Series:
        coerced = pd.to_datetime(series, errors="coerce")
        if pd.api.types.is_datetime64tz_dtype(coerced):
            coerced = coerced.dt.tz_localize(None)
        return coerced.dt.normalize()

    @classmethod
    def _clean_callsign(cls, series: pd.Series, source_name: str) -> pd.Series:
        cleaned = (
            series.astype("string")
            .str.upper()
            .str.strip()
            .str.replace(r"\s+", "", regex=True)
        )

        if source_name == "bts":
            for iata_prefix, icao_prefix in cls.BTS_AIRLINE_PREFIX_MAP.items():
                cleaned = cleaned.str.replace(
                    rf"^{iata_prefix}(\d+.*)$",
                    rf"{icao_prefix}\1",
                    regex=True,
                )

        return cleaned.where(~cleaned.isin(cls.INVALID_CALLSIGNS), pd.NA)

    def _prepare_adsb(self, adsb_df: pd.DataFrame) -> pd.DataFrame:
        adsb = adsb_df.copy()
        if "callsign" not in adsb.columns:
            adsb["callsign"] = pd.NA

        if "start_time_utc" in adsb.columns:
            adsb["dep_anchor_utc"] = self._ensure_utc_ns(adsb["start_time_utc"])
        elif "timestamp" in adsb.columns and pd.api.types.is_numeric_dtype(adsb["timestamp"]):
            adsb["dep_anchor_utc"] = self._ensure_utc_ns(adsb["timestamp"], unit="s")
        else:
            adsb["dep_anchor_utc"] = self._ensure_utc_ns(adsb["timestamp"])

        if "end_time_utc" in adsb.columns:
            adsb["arr_anchor_utc"] = self._ensure_utc_ns(adsb["end_time_utc"])
        elif {"timestamp", "flight_duration"}.issubset(adsb.columns):
            arr_seconds = pd.to_numeric(adsb["timestamp"], errors="coerce") + pd.to_numeric(adsb["flight_duration"], errors="coerce")
            adsb["arr_anchor_utc"] = self._ensure_utc_ns(arr_seconds, unit="s")
        else:
            adsb["arr_anchor_utc"] = pd.NaT

        if "dep_anchor_confidence" not in adsb.columns:
            adsb["dep_anchor_confidence"] = np.where(adsb.get("takeoff_detected", False), 1.0, 0.35)
        if "arr_anchor_confidence" not in adsb.columns:
            adsb["arr_anchor_confidence"] = np.where(adsb.get("landing_detected", False), 1.0, 0.35)

        adsb["callsign_clean"] = self._clean_callsign(adsb["callsign"], source_name="adsb")
        adsb["dep_service_day"] = self._normalize_day_key(adsb["dep_anchor_utc"])
        adsb["arr_service_day"] = self._normalize_day_key(adsb["arr_anchor_utc"])
        adsb["adsb_callsign"] = adsb["callsign"]
        adsb["adsb_dt"] = adsb["dep_anchor_utc"]

        return adsb.dropna(subset=["callsign_clean", "dep_anchor_utc", "dep_service_day"]).copy()

    def _prepare_schedule(self, schedule_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        sched = schedule_df.copy()
        source_name = self._get_source_name(sched)

        def pick_series(*candidates: str, default_value=pd.NaT) -> pd.Series:
            for candidate in candidates:
                if candidate in sched.columns:
                    return sched[candidate]
            return pd.Series(default_value, index=sched.index)

        if source_name == "bts":
            sched["scheduled_dep_ref"] = self._ensure_utc_ns(pick_series("scheduled_dep_utc", "scheduled_dep"))
            sched["scheduled_arr_ref"] = self._ensure_utc_ns(pick_series("scheduled_arr_utc", "scheduled_arr"))
            sched["actual_dep"] = self._ensure_utc_ns(pick_series("actual_dep_utc", "actual_dep"))
            sched["actual_arr"] = self._ensure_utc_ns(pick_series("actual_arr_utc", "actual_arr"))
            service_day_source = pick_series("service_day_local", "service_day_utc", default_value=pd.NaT)
            if service_day_source.isna().all():
                service_day_source = sched["scheduled_dep_ref"]
            sched["service_day_key"] = self._coerce_day_key(service_day_source)
            sched["label_source"] = "bts_verified"
        else:
            sched["scheduled_dep_ref"] = self._ensure_utc_ns(pick_series("scheduled_dep"))
            sched["scheduled_arr_ref"] = self._ensure_utc_ns(pick_series("scheduled_arr"))
            if "actual_dep" in sched.columns:
                sched["actual_dep"] = self._ensure_utc_ns(sched["actual_dep"])
            else:
                sched["actual_dep"] = pd.NaT
            if "actual_arr" in sched.columns:
                sched["actual_arr"] = self._ensure_utc_ns(sched["actual_arr"])
            else:
                sched["actual_arr"] = pd.NaT
            service_day_source = pick_series("service_day_utc", "service_day_local", default_value=pd.NaT)
            if service_day_source.isna().all():
                service_day_source = sched["scheduled_dep_ref"]
            sched["service_day_key"] = self._coerce_day_key(service_day_source)
            sched["label_source"] = "unverified_euro"

        if "callsign" not in sched.columns:
            sched["callsign"] = pd.NA
        sched["callsign_clean"] = self._clean_callsign(sched["callsign"], source_name=source_name)
        sched["cancelled"] = pd.to_numeric(sched.get("cancelled", 0), errors="coerce").fillna(0).astype("int64")
        if "Cancelled" not in sched.columns:
            sched["Cancelled"] = sched["cancelled"]
        else:
            sched["Cancelled"] = pd.to_numeric(sched["Cancelled"], errors="coerce").fillna(0).astype("int64")
        if "DepDelay" in sched.columns:
            sched["DepDelay"] = pd.to_numeric(sched["DepDelay"], errors="coerce")
        else:
            sched["DepDelay"] = np.nan

        sched["scheduled_dep"] = sched["scheduled_dep_ref"]
        sched["scheduled_arr"] = sched["scheduled_arr_ref"]
        return sched.dropna(subset=["callsign_clean", "service_day_key", "scheduled_dep_ref"]).copy(), source_name

    def _expand_schedule_service_days(self, sched: pd.DataFrame, source_name: str) -> pd.DataFrame:
        offsets = [-1, 0, 1] if source_name == "bts" else [0]
        frames = []
        for offset in offsets:
            frame = sched.copy()
            frame["candidate_service_day"] = frame["service_day_key"] + pd.to_timedelta(offset, unit="D")
            frame["candidate_day_offset"] = offset
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else sched.copy()

    def _merge_candidates(self, adsb: pd.DataFrame, sched_expanded: pd.DataFrame, anchor_name: str) -> pd.DataFrame:
        day_col = "dep_service_day" if anchor_name == "departure" else "arr_service_day"
        if day_col not in adsb.columns:
            return pd.DataFrame()
        working_adsb = adsb.dropna(subset=[day_col]).copy()
        if working_adsb.empty:
            return pd.DataFrame()

        merged = pd.merge(
            working_adsb,
            sched_expanded,
            left_on=["callsign_clean", day_col],
            right_on=["callsign_clean", "candidate_service_day"],
            how="inner",
            suffixes=("", "_sched"),
        )
        merged["candidate_anchor"] = anchor_name
        return merged

    def _score_matches(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates

        dep_score = (
            (candidates["dep_anchor_utc"] - candidates["scheduled_dep_ref"])
            .abs()
            .dt.total_seconds()
            .div(60)
        )
        arr_score = (
            (candidates["arr_anchor_utc"] - candidates["scheduled_arr_ref"])
            .abs()
            .dt.total_seconds()
            .div(60)
        )

        candidates["dep_match_minutes"] = dep_score
        candidates["arr_match_minutes"] = arr_score
        candidates["match_score_minutes"] = pd.concat([dep_score, arr_score], axis=1).min(axis=1, skipna=True)
        candidates["match_anchor"] = np.where(
            arr_score.isna() | (dep_score <= arr_score),
            "departure",
            "arrival",
        )
        candidates["time_diff_minutes"] = candidates["match_score_minutes"]

        chosen_confidence = np.where(
            candidates["match_anchor"] == "departure",
            pd.to_numeric(candidates.get("dep_anchor_confidence", 0.35), errors="coerce"),
            pd.to_numeric(candidates.get("arr_anchor_confidence", 0.35), errors="coerce"),
        )
        candidates["match_confidence"] = np.clip(chosen_confidence / (1.0 + candidates["match_score_minutes"].fillna(9999) / 60.0), 0.0, 1.0)
        return candidates

    @staticmethod
    def _schedule_match_key(df: pd.DataFrame) -> pd.Series:
        if "flight_key" in df.columns:
            key = df["flight_key"].astype("string")
        else:
            key = df["callsign_clean"].astype("string")
        dep_key = pd.to_datetime(df["scheduled_dep_ref"], utc=True, errors="coerce").astype("string")
        return key + "|" + dep_key

    def _deduplicate_matches(self, candidates: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if candidates.empty:
            return candidates, 0

        ranked = candidates.sort_values(
            ["match_score_minutes", "candidate_day_offset", "trajectory_id", "scheduled_dep_ref"],
            kind="stable",
        ).copy()
        ranked["schedule_match_key"] = self._schedule_match_key(ranked)
        before = len(ranked)
        ranked = ranked.drop_duplicates(subset=["trajectory_id"], keep="first")
        ranked = ranked.drop_duplicates(subset=["schedule_match_key"], keep="first")
        duplicates_dropped = before - len(ranked)
        ranked = ranked.drop(columns=["schedule_match_key"])
        return ranked, duplicates_dropped

    def merge_datasets(self, adsb_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
        if adsb_df.empty or schedule_df.empty:
            logger.warning("One of the datasets to merge is empty.")
            return pd.DataFrame()

        adsb_start_len = len(adsb_df)
        adsb = self._prepare_adsb(adsb_df)
        sched, source_name = self._prepare_schedule(schedule_df)
        tolerance = self._get_tolerance(sched)
        max_time_diff_minutes = tolerance.total_seconds() / 60.0

        if adsb.empty or sched.empty:
            logger.warning("Prepared ADS-B or schedule dataframe is empty after cleaning.")
            self.qa.generate_report(adsb_start_len, pd.DataFrame(), 0, 0, 0, source_name)
            return pd.DataFrame()

        sched_expanded = self._expand_schedule_service_days(sched, source_name)
        dep_candidates = self._merge_candidates(adsb, sched_expanded, "departure")
        arr_candidates = self._merge_candidates(adsb, sched_expanded, "arrival")
        candidates = pd.concat([dep_candidates, arr_candidates], ignore_index=True)
        candidate_len = len(candidates)

        if candidates.empty:
            logger.warning("No multi-anchor candidates found for %s.", source_name)
            self.qa.generate_report(adsb_start_len, pd.DataFrame(), 0, 0, 0, source_name)
            return pd.DataFrame()

        candidates = self._score_matches(candidates)
        filtered = candidates[candidates["match_score_minutes"] < max_time_diff_minutes].copy()
        filtered_len = len(filtered)

        if filtered.empty:
            logger.warning(
                "No %s candidates survived the %.0f minute time filter.",
                source_name,
                max_time_diff_minutes,
            )
            self.qa.generate_report(adsb_start_len, pd.DataFrame(), candidate_len, 0, 0, source_name)
            return pd.DataFrame()

        matched, duplicates_dropped = self._deduplicate_matches(filtered)
        matched["match_quality"] = f"multi_anchor_service_day<{int(max_time_diff_minutes)}m"
        matched["ambiguity_flag"] = matched.duplicated(subset=["callsign_clean", "service_day_key"], keep=False)
        matched["source_priority"] = 0 if source_name == "bts" else 1

        logger.info(
            "Merge summary (%s): candidates=%s, within_tolerance=%s, final=%s, min_score=%.2f, max_score=%.2f",
            source_name,
            candidate_len,
            filtered_len,
            len(matched),
            matched["match_score_minutes"].min(),
            matched["match_score_minutes"].max(),
        )

        self.qa.generate_report(
            adsb_start_len,
            matched,
            candidate_len,
            filtered_len,
            duplicates_dropped,
            source_name,
        )
        return matched
