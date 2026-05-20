"""FE5: Feature Analysis - correlation, importance, SHAP, mutual information."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.streamlit_utils import render_walkthrough_header


STATE_ORDER = ["Normal", "Late", "Cancelled"]
STATE_COL_CANDIDATES = ["disruption_subtype", "label_original", "label"]
DEFAULT_DISTRIBUTION_FEATURES = ["mean_speed", "dep_hour", "holding_pattern_count"]
DEVIATION_FEATURES = [
    "lateral_deviation_mean_km",
    "lateral_deviation_max_km",
    "lateral_deviation_std_km",
    "route_stretch_ratio",
    "approach_deviation_km",
    "altitude_deviation_mean_m",
    "altitude_deviation_max_m",
]
DISCRETE_FEATURE_META = {
    "holding_pattern_count": {
        "label": "Holding pattern count",
        "order": [0, 1],
    },
    "dep_anchor_confidence": {
        "label": "Departure anchor confidence",
        "order": [0.35, 1.0],
    },
    "arr_anchor_confidence": {
        "label": "Arrival anchor confidence",
        "order": [0.35, 1.0],
    },
    "middle_gap_count": {
        "label": "Middle gap count",
        "order": None,
    },
    "dep_hour": {
        "label": "Departure hour",
        "order": list(range(24)),
    },
    "dep_day_of_week": {
        "label": "Departure day of week",
        "order": list(range(7)),
        "ticktext": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    },
    "dep_month": {
        "label": "Departure month",
        "order": list(range(1, 13)),
        "ticktext": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    },
    "is_peak_hour": {
        "label": "Peak hour",
        "order": [0, 1],
        "ticktext": ["No", "Yes"],
    },
    "weather_severity": {
        "label": "Origin weather severity",
        "order": [0, 1, 2, 3],
    },
    "weather_severity_dest": {
        "label": "Destination weather severity",
        "order": [0, 1, 2, 3],
    },
    "unstable_descent_flag": {
        "label": "Unstable descent flag",
        "order": [False, True],
        "ticktext": ["False", "True"],
    },
    "is_full_flight": {
        "label": "Full flight trace",
        "order": [False, True],
        "ticktext": ["False", "True"],
    },
}


@st.cache_data(show_spinner=False)
def load_ml_dataset(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_model_features(path: str) -> list[str]:
    feature_path = Path(path)
    if not feature_path.exists():
        return []
    with open(feature_path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def compute_correlation_matrix(path: str, numeric_columns: tuple[str, ...]) -> pd.DataFrame:
    work = pd.read_parquet(path, columns=list(numeric_columns))
    return work.corr(method="pearson")


@st.cache_data(show_spinner=False)
def compute_mutual_information(path: str, numeric_columns: tuple[str, ...]) -> pd.DataFrame:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    work = pd.read_parquet(path, columns=[*numeric_columns, "label"])
    X = work[list(numeric_columns)].fillna(0)
    y = LabelEncoder().fit_transform(work["label"].astype(str))
    mi = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({"feature": list(numeric_columns), "mi_score": mi}).sort_values("mi_score", ascending=False)


def state_column(df: pd.DataFrame) -> str | None:
    return next((col for col in STATE_COL_CANDIDATES if col in df.columns), None)


def feature_label(feature: str) -> str:
    return DISCRETE_FEATURE_META.get(feature, {}).get("label", feature.replace("_", " ").title())


def build_state_count_figure(df: pd.DataFrame, state_col: str) -> go.Figure:
    state_counts = (
        df[state_col]
        .astype(str)
        .value_counts()
        .reindex(STATE_ORDER, fill_value=0)
        .rename_axis("state")
        .reset_index(name="count")
    )
    max_count = int(state_counts["count"].max())
    cancelled_count = int(state_counts.loc[state_counts["state"] == "Cancelled", "count"].iloc[0])
    low_ceiling = max(cancelled_count * 2, 1000)
    high_floor = max(int(max_count * 0.15), low_ceiling + 1)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.72, 0.28],
    )
    colors = {"Normal": "#2f6b9a", "Late": "#d08b2d", "Cancelled": "#b8534d"}
    fig.add_trace(
        go.Bar(
            x=state_counts["state"],
            y=state_counts["count"],
            marker_color=[colors[state] for state in state_counts["state"]],
            text=state_counts["count"],
            texttemplate="%{text:,}",
            textposition="outside",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=state_counts["state"],
            y=state_counts["count"],
            marker_color=[colors[state] for state in state_counts["state"]],
            text=state_counts["count"],
            texttemplate="%{text:,}",
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(range=[high_floor, max_count * 1.08], title_text="Flights", row=1, col=1)
    fig.update_yaxes(range=[0, low_ceiling], title_text="Flights", row=2, col=1)
    fig.update_xaxes(categoryorder="array", categoryarray=STATE_ORDER, title_text="Outcome state", row=2, col=1)
    fig.add_annotation(
        text="Axis break",
        xref="paper",
        yref="paper",
        x=1.005,
        y=0.29,
        showarrow=False,
        font=dict(size=11, color="#667085"),
    )
    fig.update_layout(
        title="Outcome State Distribution",
        height=520,
        margin=dict(t=65, r=70, b=50, l=70),
        bargap=0.32,
    )
    return fig


def build_continuous_distribution(df: pd.DataFrame, feature: str, state_col: str) -> go.Figure:
    plot_df = df[[feature, state_col]].dropna().copy()
    fig = px.histogram(
        plot_df,
        x=feature,
        color=state_col,
        histnorm="percent",
        barmode="overlay",
        opacity=0.58,
        category_orders={state_col: STATE_ORDER},
        title=f"{feature_label(feature)} by Outcome State",
    )
    fig.update_layout(
        xaxis_title=feature_label(feature),
        yaxis_title="Within-state percentage",
        legend_title_text="Outcome state",
    )
    return fig


def build_discrete_distribution(df: pd.DataFrame, feature: str, state_col: str) -> go.Figure:
    meta = DISCRETE_FEATURE_META[feature]
    plot_df = df[[feature, state_col]].dropna().copy()
    if meta.get("order") is None:
        category_order = sorted(plot_df[feature].dropna().unique().tolist())
    else:
        category_order = meta["order"]

    counts = (
        plot_df.groupby([state_col, feature], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = counts.groupby(state_col, observed=True)["count"].transform("sum")
    counts["percentage"] = counts["count"] / totals * 100

    fig = px.bar(
        counts,
        x=feature,
        y="percentage",
        color=state_col,
        barmode="group",
        category_orders={
            state_col: STATE_ORDER,
            feature: category_order,
        },
        title=f"{feature_label(feature)} by Outcome State",
    )
    fig.update_layout(
        xaxis_title=feature_label(feature),
        yaxis_title="Within-state percentage",
        legend_title_text="Outcome state",
    )
    if meta.get("ticktext"):
        fig.update_xaxes(tickmode="array", tickvals=meta["order"], ticktext=meta["ticktext"])
    return fig


def build_feature_distribution(df: pd.DataFrame, feature: str, state_col: str) -> go.Figure:
    if feature in DISCRETE_FEATURE_META:
        return build_discrete_distribution(df, feature, state_col)
    return build_continuous_distribution(df, feature, state_col)


st.set_page_config(page_title="Feature Engineering", page_icon="Feature", layout="wide")
st.markdown("# Step 5. Feature Engineering")
render_walkthrough_header("features")

ml_path = Path("data/processed/ml_dataset.parquet")
if not ml_path.exists():
    st.warning("ML dataset not available.")
    st.stop()

df = load_ml_dataset(str(ml_path))
if df.empty:
    st.warning("ML dataset is empty.")
    st.stop()

model_features = [feature for feature in load_model_features("models/feature_list.json") if feature in df.columns]
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
outcome_col = state_column(df)
tabs = st.tabs(["Correlation", "Feature Importance", "Category Importance", "SHAP", "Mutual Information", "Route Deviation", "Distributions"])

with tabs[0]:
    st.subheader("Correlation Matrix (Pearson)")
    if numeric_cols:
        corr = compute_correlation_matrix(str(ml_path), tuple(numeric_cols))
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No numeric columns available.")

with tabs[1]:
    st.subheader("Feature Importance")
    imp_path = Path("models/feature_importance.json")
    if imp_path.exists():
        with open(imp_path, encoding="utf-8") as handle:
            importance = json.load(handle)
        imp_df = pd.DataFrame({"feature": list(importance.keys()), "importance": list(importance.values())}).sort_values(
            "importance", ascending=False
        )
        top = imp_df.head(10)
        fig = px.bar(top.sort_values("importance"), x="importance", y="feature", orientation="h", title="Top 10 Features")
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Feature importance file not found yet.")

with tabs[2]:
    st.subheader("Feature Importance by Category")
    category_img = Path("outputs/feature_importance_by_category.png")
    if category_img.exists():
        st.image(str(category_img), width="stretch")
    else:
        st.info("Grouped category importance image not available yet.")

with tabs[3]:
    st.subheader("SHAP Summary")
    shap_img = Path("outputs/shap_summary.png")
    if shap_img.exists():
        st.image(str(shap_img), width="stretch")
    else:
        st.info("SHAP summary image not available.")

with tabs[4]:
    st.subheader("Mutual Information vs Label")
    if "label" in df.columns and numeric_cols:
        try:
            mi_df = compute_mutual_information(str(ml_path), tuple(numeric_cols))
            fig = px.bar(mi_df.head(20).sort_values("mi_score"), x="mi_score", y="feature", orientation="h", title="Top MI Features")
            st.plotly_chart(fig, width="stretch")
        except Exception as exc:
            st.info(f"MI computation unavailable: {exc}")
    else:
        st.info("Need label + numeric features for MI.")


@st.fragment
def render_route_deviation_panel():
    st.subheader("Route Deviation Features")
    deviation_cols = [feature for feature in DEVIATION_FEATURES if feature in df.columns]
    if not deviation_cols:
        st.info("No route deviation columns available in the ML dataset.")
        return

    coverage = pd.DataFrame(
        {
            "feature": deviation_cols,
            "coverage_pct": [round(float(df[feature].notna().mean() * 100), 2) for feature in deviation_cols],
        }
    )
    fig_cov = px.bar(coverage, x="feature", y="coverage_pct", title="Route Deviation Coverage %")
    st.plotly_chart(fig_cov, width="stretch")

    selected_dev = st.selectbox("Deviation feature", deviation_cols, key="route_dev_feature")
    plot_cols = [selected_dev] + ([outcome_col] if outcome_col else [])
    plot_df = df[plot_cols].copy()
    if outcome_col is not None:
        fig = px.histogram(
            plot_df,
            x=selected_dev,
            color=outcome_col,
            histnorm="percent",
            barmode="overlay",
            opacity=0.65,
            category_orders={outcome_col: STATE_ORDER},
        )
    else:
        fig = px.histogram(plot_df, x=selected_dev)
    st.plotly_chart(fig, width="stretch")


@st.fragment
def render_distributions_panel():
    st.subheader("Feature Distributions")
    if outcome_col is None:
        st.info("Need an outcome-state column to compare distributions.")
        return

    st.plotly_chart(build_state_count_figure(df, outcome_col), width="stretch")
    if not model_features:
        st.info("No trained model feature list is available.")
        return

    default_features = [feature for feature in DEFAULT_DISTRIBUTION_FEATURES if feature in model_features]
    selected_features = st.multiselect(
        "Features",
        model_features,
        default=default_features,
    )
    for feature in selected_features:
        st.plotly_chart(build_feature_distribution(df, feature, outcome_col), width="stretch")


with tabs[5]:
    render_route_deviation_panel()

with tabs[6]:
    render_distributions_panel()
