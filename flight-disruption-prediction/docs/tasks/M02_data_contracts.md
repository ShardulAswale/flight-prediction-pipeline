# Milestone 2 — Data Contracts & Validation Wiring

> **Status**: `[x]` Complete
> **Priority**: 🟡 HIGH
> **Depends on**: M1 (T2 for main.py imports)

---

## `[x]` TASK-3: Define schema contracts for all 5 pipeline stages

**Files**: `src/schemas.py` [NEW]
**What**: Create a single source-of-truth module containing schema dictionaries for: (1) Raw ADS-B States, (2) Trajectories, (3) Flight Features, (4) Canonical Schedule, (5) Final ML Dataset. Each schema specifies `dtype`, `max_null_pct`, and `is_utc` per column, matching the contracts in the implementation plan.

**Steps**:
1. Create `src/schemas.py`
2. Define `ADSB_STATES_SCHEMA`, `TRAJECTORIES_SCHEMA`, `FEATURES_SCHEMA`, `CANONICAL_SCHEDULE_SCHEMA` (reuse `ScheduleNormalizer.canonical_schema()`), `ML_DATASET_SCHEMA`
3. Each dict entry: `{'dtype': str, 'max_null_pct': float, 'is_utc': bool}`

**Acceptance Criteria**:
- [ ] `from src.schemas import ADSB_STATES_SCHEMA, ML_DATASET_SCHEMA` succeeds
- [ ] Each schema is a `dict[str, dict]` with ≥5 columns
- [ ] `ML_DATASET_SCHEMA` contains all 30 columns from the plan's Stage 5 table
- [ ] `CANONICAL_SCHEDULE_SCHEMA` matches `ScheduleNormalizer.canonical_schema()` output

---

## `[x]` TASK-4: Wire DataValidator into main.py at pipeline boundaries

**Files**: `main.py`, `src/data_validator.py`, `configs/config.yaml`
**What**: Import `DataValidator` and the new schemas into `main.py`. Insert `.validate()` calls after each stage output (features, merge, weather, labeling, final). Add a `validation_mode` key to `config.yaml` defaulting to `warn_only`.

**Steps**:
1. Add `from src.data_validator import DataValidator` and `from src.schemas import *` to `main.py`
2. Instantiate `DataValidator(mode=config.get('validation_mode', 'warn_only'))` at top of `main()`
3. After step 4 (features): `validator.validate(df_adsb_features, FEATURES_SCHEMA, 'trajectory_features')`
4. After merge: `validator.validate(df_merged, ML_DATASET_SCHEMA, 'merged_dataset')` (warn-only, schema is partial at this point)
5. After labeling: `validator.validate(df_final, ML_DATASET_SCHEMA, 'ml_dataset')`
6. Add `validation_mode: warn_only` to `configs/config.yaml`

**Acceptance Criteria**:
- [ ] Running `python main.py` produces `logs/validation_*.json` report files
- [ ] Setting `validation_mode: fail_fast` in config causes pipeline to abort on schema violation
- [ ] No change to pipeline output data when validation is `warn_only`
