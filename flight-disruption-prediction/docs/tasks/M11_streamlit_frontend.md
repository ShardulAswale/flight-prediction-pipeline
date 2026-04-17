# Milestone 11 — Streamlit Frontend

> **Status**: `[~]` In Progress (code complete, data/model-dependent widgets pending)
> **Priority**: 🔵 MEDIUM  
> **Depends on**: M8 (T20 for trained models), M9 (T21 for SHAP)  
> **Note**: Each page is an independent file in `pages/` — can be developed/tested in isolation.

---

## `[x]` TASK-FE1: Convert app.py to multi-page Streamlit app

**Files**: `app.py`, `pages/` [NEW directory], `src/streamlit_utils.py` [NEW]  
**What**: Refactor the existing single-file `app.py` into a multi-page Streamlit app using the `pages/` directory convention.

**Steps**:
1. Rename existing `app.py` content into a "Home" page
2. Create `pages/` directory
3. Create `src/streamlit_utils.py` with shared helpers: `load_dataset()`, `load_model()`, `get_run_metadata()`, `style_page()`
4. Add global page config: dark theme, wide layout, sidebar navigation
5. Home page shows: project title, pipeline status summary, last run timestamp

**Acceptance Criteria**:
- [ ] `streamlit run app.py` shows sidebar with navigation to all pages
- [ ] Home page loads without errors and displays pipeline status
- [ ] Each page file in `pages/` loads independently without import errors
- [ ] Consistent styling across all pages (dark theme, common header)

---

## `[x]` TASK-FE2: Pipeline Overview page

**Files**: `pages/1_Pipeline_Overview.py` [NEW]  
**What**: Dashboard showing pipeline stage status, row counts, last run time, and data source volume chart.

**Steps**:
1. Read `logs/run_*.json` files to show stage execution history
2. Stage status table: Stage | Status | Rows In | Rows Out | Duration
3. Plotly bar chart: data volume by source (OpenSky, BTS, Eurocontrol)
4. Merge funnel: ADS-B input → Phase A → Phase B → Final (from `merge_qa_report_*.json`)

**Acceptance Criteria**:
- [ ] Page loads and shows table of stage run history
- [ ] Bar chart renders with correct data source volumes
- [ ] Merge funnel visualizes retention at each phase
- [ ] Handles missing log files gracefully ("No data yet")

---

## `[x]` TASK-FE3: Data Explorer page

**Files**: `pages/2_Data_Explorer.py` [NEW]  
**What**: Interactive data browser for raw ADS-B, schedule, and merged datasets.

**Steps**:
1. Dataset selector: Raw ADS-B | BTS Schedules | Eurocontrol Schedules | Merged | ML Dataset
2. Filter widgets: date range picker, airport multiselect, callsign text search
3. Show filtered DataFrame with `st.dataframe` (paginated, 100 rows)
4. Summary statistics panel: row count, null %, column types
5. Download filtered data as CSV button

**Acceptance Criteria**:
- [ ] All 5 dataset options load and display correctly
- [ ] Filters update displayed data in real-time
- [ ] CSV download works for filtered data
- [ ] Handles large datasets without timeout (sampling for >1M rows)

---

## `[x]` TASK-FE4: Trajectory Map page

**Files**: `pages/3_Trajectory_Map.py` [NEW]  
**What**: Interactive Folium map showing flight trajectories with altitude profile chart.

**Steps**:
1. Trajectory selector: dropdown of trajectory IDs from `trajectories.parquet`
2. Folium map: plot selected trajectory polyline with color-coded altitude
3. Altitude profile: matplotlib line chart (distance vs altitude) below map
4. Speed profile: optional toggle to show speed along trajectory
5. Trajectory metadata panel: icao24, callsign, duration, distance, max altitude

**Acceptance Criteria**:
- [ ] Map renders with at least one trajectory visible
- [ ] Altitude profile updates when different trajectory is selected
- [ ] Map includes departure/arrival markers (if matched to schedule)
- [ ] Page loads in <5 seconds for a single trajectory

---

## `[x]` TASK-FE5: Feature Analysis page

**Files**: `pages/4_Feature_Analysis.py` [NEW]  
**What**: Interactive feature analysis dashboard with correlation matrix, importance ranking, and SHAP.

**Steps**:
1. Correlation matrix heatmap (Pearson): interactive plotly, highlight >0.8
2. Feature importance bar chart: load from `outputs/`
3. SHAP summary plot: embed `outputs/shap_summary.png`
4. Mutual information ranking: bar chart of MI scores vs label
5. Feature selector: checkboxes to compare distributions (histogram overlay)

**Acceptance Criteria**:
- [ ] Correlation matrix renders and is interactive (hover shows values)
- [ ] Feature importance chart shows top 10 features
- [ ] SHAP plot displays correctly
- [ ] Feature histograms update when different features selected

---

## `[x]` TASK-FE6: Model Performance page

**Files**: `pages/5_Model_Performance.py` [NEW]  
**What**: Model comparison dashboard with confusion matrices, ROC curves, and ranking table.

**Steps**:
1. Comparison table: Model | F1 | Precision | Recall | Accuracy (from `logs/model_comparison.json`)
2. Confusion matrix heatmaps: one per model in 2×2 grid
3. ROC curves: embed `outputs/roc_curves.png` or render with plotly
4. Precision-Recall curves: one per class per model
5. Class distribution pie chart: actual vs predicted

**Acceptance Criteria**:
- [ ] Comparison table loads and highlights best model
- [ ] Confusion matrices show correct predictions per class
- [ ] ROC curves display for all models with AUC values
- [ ] All visuals handle the 3-class problem correctly

---

## `[x]` TASK-FE7: Predictions Explorer page

**Files**: `pages/6_Predictions_Explorer.py` [NEW]  
**What**: Interactive prediction tool — input flight features, get predicted label with SHAP explanation.

**Steps**:
1. Input form: sliders for key features (flight_duration, mean_speed, mean_altitude, wind_speed, visibility)
2. "Predict" button: load best model, run prediction, show label + confidence %
3. SHAP force plot: generate per-prediction explanation, display as HTML
4. Similar flights panel: show 5 nearest neighbours from training set
5. Batch prediction: upload CSV → download CSV with predictions

**Acceptance Criteria**:
- [ ] Single prediction returns label and confidence percentage
- [ ] SHAP force plot renders inline showing feature contributions
- [ ] Batch prediction: upload 10-row CSV → download CSV with predictions
- [ ] Input validation: sliders have reasonable min/max bounds

---

## `[x]` TASK-FE8: Data Quality Dashboard page

**Files**: `pages/7_Data_Quality.py` [NEW]  
**What**: Dashboard showing data quality metrics: missingness, drift alerts, quality gates, weather coverage.

**Steps**:
1. Missingness heatmap: null % per column per month
2. Quality gate results: load `logs/quality_gates_report.json`, show pass/fail badges
3. Drift alerts: load `logs/drift_report.json`, show PSI trend line chart per feature
4. Weather coverage: load `logs/weather_coverage.json`, show % flights with weather
5. Data freshness: timestamps of latest ingestion and model training

**Acceptance Criteria**:
- [ ] Missingness heatmap renders with clear column labels
- [ ] Quality gates show green/red status badges
- [ ] Drift chart shows PSI values with alert threshold line at 0.2
- [ ] All components handle missing log files gracefully
