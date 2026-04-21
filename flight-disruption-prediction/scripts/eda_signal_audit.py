from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    numeric: list[str]
    categorical: list[str]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", min_frequency=25, sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore")


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected file: {path}")
    return pd.read_parquet(path)


def _safe_pct(series: pd.Series) -> pd.Series:
    return (series.value_counts(dropna=False, normalize=True) * 100).round(3)


def _add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dep = pd.to_datetime(df.get("scheduled_dep"), errors="coerce", utc=True)
    arr = pd.to_datetime(df.get("scheduled_arr"), errors="coerce", utc=True)

    df["eda_dep_hour"] = dep.dt.hour
    df["eda_dep_day_of_week"] = dep.dt.dayofweek
    df["eda_dep_month"] = dep.dt.month
    df["eda_is_weekend"] = dep.dt.dayofweek.isin([5, 6]).astype("float")
    df["eda_is_peak_hour"] = dep.dt.hour.isin([6, 7, 8, 9, 16, 17, 18, 19, 20]).astype("float")
    df["eda_scheduled_duration_min"] = (arr - dep).dt.total_seconds() / 60.0

    origin = df.get("origin", pd.Series(index=df.index, dtype="object")).astype("string").str.upper()
    dest = df.get("destination", pd.Series(index=df.index, dtype="object")).astype("string").str.upper()
    df["eda_route"] = origin.fillna("UNKNOWN") + "_" + dest.fillna("UNKNOWN")
    df["eda_origin_freq"] = origin.map(origin.value_counts(dropna=True)).astype("float")
    df["eda_dest_freq"] = dest.map(dest.value_counts(dropna=True)).astype("float")
    df["eda_route_freq"] = df["eda_route"].map(df["eda_route"].value_counts(dropna=True)).astype("float")

    if "scheduled_dep" in df.columns and "origin" in df.columns:
        df["eda_origin_flights_1hr"] = _window_counts(df, "origin", dep, minutes=60)
    else:
        df["eda_origin_flights_1hr"] = np.nan

    if "scheduled_arr" in df.columns and "destination" in df.columns:
        df["eda_dest_flights_1hr"] = _window_counts(df, "destination", arr, minutes=60)
    else:
        df["eda_dest_flights_1hr"] = np.nan

    return df


def _window_counts(df: pd.DataFrame, group_col: str, times: pd.Series, minutes: int) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float")
    valid = df[group_col].notna() & times.notna()
    window_ns = np.int64(minutes * 60 * 1_000_000_000)

    for _, group_idx in df.index[valid].to_series().groupby(df.loc[valid, group_col].astype(str)):
        idx = group_idx.to_numpy()
        t = times.loc[idx].astype("int64").to_numpy()
        order = np.argsort(t)
        sorted_t = t[order]
        left = np.searchsorted(sorted_t, sorted_t - window_ns, side="left")
        right = np.searchsorted(sorted_t, sorted_t + window_ns, side="right")
        counts = right - left - 1
        result.loc[idx[order]] = counts.astype("float")

    return result


def _existing(cols: Iterable[str], df: pd.DataFrame, require_non_null: bool = True) -> list[str]:
    values = []
    for col in cols:
        if col not in df.columns:
            continue
        if require_non_null and df[col].notna().sum() == 0:
            continue
        values.append(col)
    return values


def _make_groups(df: pd.DataFrame) -> list[FeatureGroup]:
    trajectory_cols = [
        "flight_duration",
        "trajectory_length",
        "num_points",
        "mean_speed",
        "max_speed",
        "speed_std",
        "mean_altitude",
        "altitude_variance",
        "vertical_rate_std",
        "heading_variability",
        "holding_pattern_count",
        "altitude_change_count",
        "unstable_descent_flag",
        "dep_anchor_confidence",
        "arr_anchor_confidence",
        "takeoff_detected",
        "landing_detected",
        "trajectory_quality_score",
        "is_full_flight",
        "route_coverage_fraction",
    ]
    weather_cols = [
        "wind_speed",
        "visibility",
        "temperature",
        "precipitation",
        "wind_speed_dest",
        "visibility_dest",
        "temperature_dest",
        "precipitation_dest",
    ]
    weather_categorical_cols = ["weather_confidence", "weather_severity_dest"]
    temporal_cols = [
        "eda_dep_hour",
        "eda_dep_day_of_week",
        "eda_dep_month",
        "eda_is_weekend",
        "eda_is_peak_hour",
        "eda_scheduled_duration_min",
    ]
    congestion_cols = [
        "eda_origin_freq",
        "eda_dest_freq",
        "eda_route_freq",
        "eda_origin_flights_1hr",
        "eda_dest_flights_1hr",
    ]
    route_deviation_cols = [
        "route_gc_distance_km",
        "lateral_deviation_mean_km",
        "lateral_deviation_max_km",
        "lateral_deviation_std_km",
        "route_stretch_ratio",
        "approach_deviation_km",
        "altitude_deviation_mean_m",
        "altitude_deviation_max_m",
    ]

    groups = [
        FeatureGroup("trajectory_existing", _existing(trajectory_cols, df), []),
        FeatureGroup("weather_existing", _existing(weather_cols, df), _existing(weather_categorical_cols, df)),
        FeatureGroup("temporal_derived", _existing(temporal_cols, df), []),
        FeatureGroup("airport_route_categorical", _existing(congestion_cols, df), _existing(["origin", "destination", "eda_route"], df)),
        FeatureGroup("congestion_derived", _existing(congestion_cols, df), []),
        FeatureGroup("route_deviation_existing", _existing(route_deviation_cols, df), []),
    ]

    combined_numeric = sorted({col for group in groups for col in group.numeric})
    combined_categorical = sorted({col for group in groups for col in group.categorical})
    groups.append(FeatureGroup("combined_available", combined_numeric, combined_categorical))
    return groups


def _split_train_test(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    dep = pd.to_datetime(df["scheduled_dep"], errors="coerce", utc=True)
    if dep.notna().sum() > 0:
        cutoff = dep.quantile(0.8)
        train_mask = (dep < cutoff).to_numpy()
        test_mask = (dep >= cutoff).to_numpy()
        if train_mask.sum() >= 1000 and test_mask.sum() >= 1000:
            return train_mask, test_mask, f"time holdout, cutoff >= {cutoff}"

    rng = np.random.default_rng(42)
    order = rng.permutation(len(df))
    split = int(len(df) * 0.8)
    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[order[:split]] = True
    return train_mask, ~train_mask, "random holdout fallback"


def _fit_group_model(df: pd.DataFrame, group: FeatureGroup, y: pd.Series, train_mask: np.ndarray, test_mask: np.ndarray) -> dict[str, object] | None:
    numeric = [col for col in group.numeric if col in df.columns and df[col].notna().sum() > 0]
    categorical = [col for col in group.categorical if col in df.columns and df[col].notna().sum() > 0]
    if not numeric and not categorical:
        return None

    transformers = []
    if numeric:
        transformers.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _one_hot_encoder()),
                    ]
                ),
                categorical,
            )
        )

    model = Pipeline(
        steps=[
            ("prep", ColumnTransformer(transformers=transformers, remainder="drop")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=90,
                    max_depth=14,
                    min_samples_leaf=20,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    X_train = df.loc[train_mask, numeric + categorical]
    X_test = df.loc[test_mask, numeric + categorical]
    y_train = y.loc[train_mask]
    y_test = y.loc[test_mask]

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    precision, recall, f1_disrupted, _ = precision_recall_fscore_support(
        y_test, pred, labels=[1], average="binary", zero_division=0
    )

    result = {
        "group": group.name,
        "numeric_features": len(numeric),
        "categorical_features": len(categorical),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "positive_rate_test": float(y_test.mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "precision_disrupted": float(precision),
        "recall_disrupted": float(recall),
        "f1_disrupted": float(f1_disrupted),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "model": model,
        "used_numeric": numeric,
        "used_categorical": categorical,
    }
    return result


def _aggregate_rf_importance(model: Pipeline, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    importances = clf.feature_importances_
    names: list[str] = []

    if numeric:
        names.extend(numeric)
    if categorical:
        cat_pipe = prep.named_transformers_["cat"]
        onehot = cat_pipe.named_steps["onehot"]
        try:
            encoded_names = onehot.get_feature_names_out(categorical)
        except Exception:
            encoded_names = onehot.get_feature_names(categorical)
        names.extend(encoded_names.tolist())

    rows = []
    for name, importance in zip(names, importances):
        base = name
        for cat in categorical:
            if name.startswith(cat + "_"):
                base = cat
                break
        rows.append((base, float(importance)))

    out = pd.DataFrame(rows, columns=["feature", "importance"])
    return out.groupby("feature", as_index=False)["importance"].sum().sort_values("importance", ascending=False)


def _univariate_numeric_auc(df: pd.DataFrame, y: pd.Series, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        if x.notna().sum() < 100 or x.nunique(dropna=True) <= 1:
            continue
        filled = x.fillna(x.median())
        try:
            auc = roc_auc_score(y, filled)
        except ValueError:
            continue
        directional_auc = max(auc, 1 - auc)
        corr = pd.Series(filled).corr(y, method="spearman")
        rows.append(
            {
                "feature": col,
                "non_null_pct": round(float(x.notna().mean() * 100), 3),
                "directional_auc": round(float(directional_auc), 4),
                "spearman_abs": round(float(abs(corr)) if pd.notna(corr) else 0.0, 4),
                "higher_means_more_disrupted": bool(auc >= 0.5),
            }
        )
    return pd.DataFrame(rows).sort_values(["directional_auc", "spearman_abs"], ascending=False)


def _rate_table(df: pd.DataFrame, group_col: str, y: pd.Series, min_count: int) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()
    tmp = pd.DataFrame({group_col: df[group_col], "disrupted": y})
    out = (
        tmp.dropna(subset=[group_col])
        .groupby(group_col)
        .agg(flights=("disrupted", "size"), disrupted_rate=("disrupted", "mean"))
        .reset_index()
    )
    out = out[out["flights"] >= min_count].copy()
    out["disrupted_rate"] = (out["disrupted_rate"] * 100).round(2)
    return out.sort_values(["disrupted_rate", "flights"], ascending=[False, False])


def _markdown_table(df: pd.DataFrame, *, index: bool = False) -> str:
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```text\n" + df.to_string(index=index) + "\n```"


def _write_markdown(
    out_path: Path,
    *,
    root: Path,
    source_path: Path,
    df: pd.DataFrame,
    split_desc: str,
    group_results: pd.DataFrame,
    top_importance: pd.DataFrame,
    top_auc: pd.DataFrame,
    nulls: pd.DataFrame,
    origin_rates: pd.DataFrame,
    dest_rates: pd.DataFrame,
    route_rates: pd.DataFrame,
) -> None:
    label_counts = df["label"].value_counts(dropna=False)
    disrupted_rate = (df["eda_disrupted"].mean() * 100).round(3)
    dep = pd.to_datetime(df["scheduled_dep"], errors="coerce", utc=True)

    lines = [
        "# EDA Signal Audit",
        "",
        f"- Project root: `{root}`",
        f"- Source dataset: `{source_path}`",
        f"- Rows: `{len(df):,}`",
        f"- Columns: `{df.shape[1]:,}`",
        f"- Scheduled departure range: `{dep.min()}` to `{dep.max()}`",
        f"- Split: `{split_desc}`",
        f"- Binary target: `Late or Cancelled` vs `Normal`",
        f"- Disrupted rate: `{disrupted_rate}%`",
        "",
        "## Label Distribution",
        "",
        _markdown_table(label_counts.to_frame("rows"), index=True),
        "",
        "## Model Signal By Feature Group",
        "",
        _markdown_table(group_results.drop(columns=["model"], errors="ignore"), index=False),
        "",
        "## Top Combined Model Feature Importance",
        "",
        _markdown_table(top_importance.head(25), index=False),
        "",
        "## Top Individual Numeric Signals",
        "",
        _markdown_table(top_auc.head(25), index=False),
        "",
        "## Highest Missingness Columns",
        "",
        _markdown_table(nulls.head(30), index=False),
        "",
        "## Highest Disruption Rate Origins",
        "",
        _markdown_table(origin_rates.head(20), index=False),
        "",
        "## Highest Disruption Rate Destinations",
        "",
        _markdown_table(dest_rates.head(20), index=False),
        "",
        "## Highest Disruption Rate Routes",
        "",
        _markdown_table(route_rates.head(20), index=False),
        "",
        "## Read This Carefully",
        "",
        "- This report audits the current final ML dataset, not the unfinished 6-month trajectory rebuild.",
        "- Columns that are 100% null cannot help the model until the upstream pipeline is rerun successfully.",
        "- Airport/route rate tables are descriptive only. If used as model features, they need leakage-safe encoding.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(root: Path) -> dict[str, object]:
    processed = root / "data" / "processed"
    out_dir = root / "outputs" / "eda_signal_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = processed / "ml_dataset.parquet"
    df = _read_parquet(source_path)
    df = df[df["label"].isin(["Normal", "Late", "Cancelled"])].copy()
    df = _add_derived_features(df)
    df["eda_disrupted"] = df["label"].isin(["Late", "Cancelled"]).astype(int)
    y = df["eda_disrupted"]

    train_mask, test_mask, split_desc = _split_train_test(df)
    groups = _make_groups(df)

    results = []
    models: dict[str, dict[str, object]] = {}
    for group in groups:
        print(f"Fitting group: {group.name}")
        result = _fit_group_model(df, group, y, train_mask, test_mask)
        if result is None:
            continue
        models[group.name] = result
        results.append({key: value for key, value in result.items() if key != "model"})

    group_results = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    group_results.to_csv(out_dir / "feature_group_model_results.csv", index=False)

    best = models.get("combined_available")
    if best is not None:
        top_importance = _aggregate_rf_importance(
            best["model"], best["used_numeric"], best["used_categorical"]
        )
    else:
        top_importance = pd.DataFrame(columns=["feature", "importance"])
    top_importance.to_csv(out_dir / "combined_feature_importance.csv", index=False)

    all_numeric = sorted({col for group in groups for col in group.numeric})
    top_auc = _univariate_numeric_auc(df, y, all_numeric)
    top_auc.to_csv(out_dir / "numeric_univariate_auc.csv", index=False)

    nulls = (
        (df.isna().mean() * 100)
        .round(3)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "column", 0: "null_pct"})
    )
    nulls.to_csv(out_dir / "missingness.csv", index=False)

    origin_rates = _rate_table(df, "origin", y, min_count=100)
    dest_rates = _rate_table(df, "destination", y, min_count=100)
    route_rates = _rate_table(df, "eda_route", y, min_count=50)
    origin_rates.to_csv(out_dir / "origin_disruption_rates.csv", index=False)
    dest_rates.to_csv(out_dir / "destination_disruption_rates.csv", index=False)
    route_rates.to_csv(out_dir / "route_disruption_rates.csv", index=False)

    _write_markdown(
        out_dir / "eda_signal_audit.md",
        root=root,
        source_path=source_path,
        df=df,
        split_desc=split_desc,
        group_results=group_results,
        top_importance=top_importance,
        top_auc=top_auc,
        nulls=nulls,
        origin_rates=origin_rates,
        dest_rates=dest_rates,
        route_rates=route_rates,
    )

    summary = {
        "rows": int(len(df)),
        "disrupted_rate": float(y.mean()),
        "split": split_desc,
        "best_group": None if group_results.empty else group_results.iloc[0]["group"],
        "best_roc_auc": None if group_results.empty else float(group_results.iloc[0]["roc_auc"]),
        "report": str(out_dir / "eda_signal_audit.md"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a signal audit on the current ML dataset.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    args = parser.parse_args()

    summary = run(args.root.resolve())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
