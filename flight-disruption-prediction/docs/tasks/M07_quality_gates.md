# Milestone 7 — Quality Gates & Final Dataset

> **Status**: `[x]` Complete
> **Priority**: 🟡 MEDIUM
> **Depends on**: M2 (T3 for schemas), M6 (labeled data)

---

## `[x]` TASK-15: Create quality_gates.py

**Files**: `src/quality_gates.py` [NEW]
**What**: Create a quality gate runner that checks: (1) missingness ≤ threshold per column, (2) no duplicate `flight_key`s, (3) label balance (each class ≥5%), (4) feature quality score (1 − null fraction). Adds `training_eligible` and `feature_quality_score` columns.

**Steps**:
1. Create `src/quality_gates.py` with class `QualityGateRunner`
2. Methods: `check_missingness(df, max_null_pct=0.3)`, `check_duplicates(df, key_col='flight_key')`, `check_label_balance(df, min_pct=0.05)`, `compute_feature_quality_score(df, feature_cols)`, `run_all(df) -> dict`
3. `run_all()` returns a JSON-serializable report and saves to `logs/quality_gates_report.json`
4. Add `training_eligible` column: `True` if `feature_quality_score > 0.7`
5. Add `feature_quality_score` column: `1.0 - (null_count / total_feature_cols)` per row

**Acceptance Criteria**:
- [ ] `from src.quality_gates import QualityGateRunner` succeeds
- [ ] `QualityGateRunner().run_all(df)` returns dict with keys: `missingness_passed`, `duplicates_passed`, `label_balance_passed`, `overall_passed`
- [ ] Output df has `training_eligible` (bool) and `feature_quality_score` (float 0-1) columns
- [ ] `logs/quality_gates_report.json` is created with gate results

---

## `[x]` TASK-16: Enforce strict column order in final Parquet exporter

**Files**: `main.py`, `src/schemas.py`
**What**: Before saving the final `ml_dataset.parquet`, reorder columns to match the exact order defined in `ML_DATASET_SCHEMA`. Drop any extra columns, add missing ones as NaN, and log the final column manifest.

**Steps**:
1. In `src/schemas.py`, add `ML_DATASET_COLUMN_ORDER: list[str]` matching the Stage 5 table order
2. In `main.py` step 10, before `to_parquet()`: reorder df to `ML_DATASET_COLUMN_ORDER`, add missing cols as NaN, drop extras
3. Log: `"Final dataset columns: [...]"` and `"Dropped extra columns: [...]"`

**Acceptance Criteria**:
- [ ] `pd.read_parquet('ml_dataset.parquet').columns.tolist()` matches `ML_DATASET_COLUMN_ORDER` exactly
- [ ] No KeyError if a column is missing from the pipeline output (gets NaN)
- [ ] Extra columns produced by intermediate stages are not in the final file
