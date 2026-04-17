# Milestone 6 — Weather & Labeling Hardening

> **Status**: `[x]` Complete
> **Priority**: 🟡 MEDIUM
> **Depends on**: M5 (merge produces actual data)

---

## `[x]` TASK-12: Handle NaT/missing times in label generation

**Files**: `src/labeling.py`
**What**: `_compute_delay()` can crash if `actual_dep` or `scheduled_dep` are NaT. Ensure it returns `NaN` for delay_minutes when times are missing, and log how many rows have missing times.

**Steps**:
1. In `_compute_delay()`, after computing `delay_minutes`, log: `"{n} flights have NaT departure times, delay set to NaN"`
2. Ensure `np.nan` (not error) when subtraction produces NaT
3. Add explicit NaT check: if both time pairs are NaT for a row, set `delay_minutes = np.nan`

**Acceptance Criteria**:
- [ ] `LabelGenerator().generate_labels(df_with_nat_times)` does not raise any exception
- [ ] Rows with NaT actual_dep get `delay_minutes = NaN` and `label = 'Normal'` (not crash)
- [ ] Log message reports count of NaT rows

---

## `[x]` TASK-13: Add target leakage guard

**Files**: `src/labeling.py`
**What**: After labels are generated, verify that `actual_dep` and `actual_arr` columns are NOT included in the feature set that goes to ML training. Add a `strip_leaky_columns()` method that removes them and logs what was removed.

**Steps**:
1. Add `LEAKY_COLUMNS = ['actual_dep', 'actual_arr', 'delay_minutes']` class attribute
2. Add `strip_leaky_columns(self, df, keep_target=True)` method that drops leaky cols (but keeps `delay_minutes` if `keep_target=True`)
3. In `standardize_dataset()`, call `strip_leaky_columns()` and log removed columns

**Acceptance Criteria**:
- [ ] `standardize_dataset()` output never contains `actual_dep` or `actual_arr`
- [ ] `delay_minutes` is retained (it's the target, not a feature)
- [ ] Log message lists each dropped column

---

## `[x]` TASK-14: Emit class distribution and weather null diagnostics

**Files**: `src/labeling.py`, `src/weather.py`
**What**: After labeling, write a JSON report with label distribution counts and percentages. After weather integration, write a JSON report with % flights missing weather data per column.

**Steps**:
1. In `LabelGenerator.generate_labels()`, after assigning labels: compute `value_counts()`, save to `logs/label_distribution.json`
2. In `WeatherIntegrator.add_weather_features()`, at end: compute null % for `wind_speed, visibility, temperature, precipitation, weather_severity`, save to `logs/weather_coverage.json`

**Acceptance Criteria**:
- [ ] `logs/label_distribution.json` exists after pipeline run, contains `{"Normal": {"count": N, "pct": P}, ...}`
- [ ] `logs/weather_coverage.json` exists, contains null % per weather column
- [ ] Both files are valid JSON parseable by `json.load()`
