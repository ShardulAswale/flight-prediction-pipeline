# Milestone 10 — CI/CD & Testing

> **Status**: `[x]` Complete
> **Priority**: 🟢 MEDIUM (T23 can start early — no dependencies)

---

## `[x]` TASK-23: Set up pytest infrastructure

**Files**: `pytest.ini` [NEW], `tests/__init__.py` [NEW], `tests/conftest.py` [NEW]  
**What**: Create test directory, pytest config, and shared fixtures.

**Steps**:
1. Create `pytest.ini` with `testpaths = tests`, `python_files = test_*.py`
2. Create `tests/__init__.py` (empty)
3. Create `tests/conftest.py` with fixtures: `sample_adsb_df()` (20 rows), `sample_schedule_df()` (10 rows), `sample_features_df()` (10 rows), `sample_weather_df()` (5 rows)

**Acceptance Criteria**:
- [ ] `pytest --collect-only` discovers conftest, reports 0 errors
- [ ] Each fixture returns a valid DataFrame matching its stage schema
- [ ] `pytest -v` runs with exit code 0

---

## `[x]` TASK-24: Unit tests for DataValidator

**Files**: `tests/test_data_validator.py` [NEW]  
**What**: Test `DataValidator.validate()` in both modes. Cover: missing columns, dtype mismatch, null threshold, UTC enforcement.

**Tests**: `test_validate_passes_valid_data`, `test_validate_fails_missing_column`, `test_validate_warns_missing_column`, `test_validate_fails_null_threshold`, `test_validate_fails_non_utc_datetime`, `test_validate_passes_utc_datetime`

**Acceptance Criteria**:
- [ ] `pytest tests/test_data_validator.py -v` → 6 tests pass
- [ ] 100% branch coverage of `DataValidator.validate()`
- [ ] No test takes >1 second

---

## `[x]` TASK-25: Unit tests for ScheduleNormalizer

**Files**: `tests/test_normalization.py` [NEW]  
**What**: Test `normalize_bts()` and `normalize_eurocontrol()`. Cover: column mapping, IATA→ICAO, UTC timestamps, flight_key.

**Tests**: `test_normalize_bts_output_columns`, `test_normalize_bts_callsign_format`, `test_normalize_bts_iata_to_icao`, `test_normalize_bts_unknown_airport_fallback`, `test_normalize_eurocontrol_output_columns`, `test_normalize_eurocontrol_utc_times`

**Acceptance Criteria**:
- [ ] `pytest tests/test_normalization.py -v` → 6 tests pass
- [ ] Both normalizers tested with 10+ row DataFrames
- [ ] Edge case: empty DataFrame does not crash

---

## `[x]` TASK-26: Unit tests for LabelGenerator

**Files**: `tests/test_labeling.py` [NEW]  
**What**: Test `generate_labels()` and `standardize_dataset()`. Cover: normal/late/cancelled, NaT handling, threshold edge cases.

**Tests**: `test_label_normal` (5min→Normal), `test_label_late` (20min→Late), `test_label_cancelled` (cancelled=1→Cancelled), `test_label_nat_times`, `test_label_threshold_boundary` (15min exactly→Normal), `test_standardize_drops_extra_columns`

**Acceptance Criteria**:
- [ ] `pytest tests/test_labeling.py -v` → 6 tests pass
- [ ] Boundary test verifies `>` (not `>=`) for threshold

---

## `[x]` TASK-27: Unit tests for QualityGateRunner

**Files**: `tests/test_quality_gates.py` [NEW]  
**What**: Test each gate method independently and combined `run_all()`.

**Tests**: `test_missingness_passes`, `test_missingness_fails`, `test_duplicates_passes`, `test_duplicates_fails`, `test_label_balance_passes`, `test_label_balance_fails`, `test_feature_quality_score`, `test_run_all_returns_report`

**Acceptance Criteria**:
- [ ] `pytest tests/test_quality_gates.py -v` → 8 tests pass
- [ ] Each gate testable in isolation
- [ ] `run_all()` report is JSON-serializable

---

## `[x]` TASK-28: Create GitHub Actions CI workflow

**Files**: `.github/workflows/ci.yml` [NEW]  
**What**: CI pipeline on push/PR: lint (ruff), type-check (mypy), unit tests (pytest).

**Steps**:
1. Create `.github/workflows/ci.yml`
2. Jobs: `lint`, `typecheck`, `test` — Python 3.11, cached pip
3. Fail on any job failure

**Acceptance Criteria**:
- [ ] Valid GitHub Actions YAML syntax
- [ ] All three jobs defined: `lint`, `typecheck`, `test`
- [ ] Triggers on `push` and `pull_request` to `main`

---

## `[x]` TASK-29: Pin all dependencies in requirements.txt

**Files**: `requirements.txt`  
**What**: Pin all existing and new dependencies to specific versions.

**Steps**:
1. `pip freeze` to capture current versions
2. Pin all deps with `==`, add: `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap`, `ruff`, `mypy`, `pytest`, `pytest-cov`
3. Separate with comments: `# Core`, `# ML`, `# Testing`, `# Dev`

**Acceptance Criteria**:
- [ ] Every line has `==` version pin
- [ ] `pip install -r requirements.txt` succeeds in a clean venv
