# Milestone 5 — Merge Engine Hardening

> **Status**: `[x]` Complete
> **Priority**: 🟡 MEDIUM
> **Depends on**: M3 (T5, T6 for normalized schedule data), M4 (T8 for orchestrator)

---

## `[x]` TASK-10: Add configurable per-region tolerance to merge

**Files**: `src/merge.py`, `configs/config.yaml`
**What**: Currently `DataMerger` accepts a single `tolerance_hours`. Extend to accept a dict `{'EU': 2, 'US': 3}` with a fallback default value. The tolerance is selected based on the `region` column in the schedule dataframe.

**Steps**:
1. Change `DataMerger.__init__` to accept `tolerance_hours: dict | int = 2`
2. If int, convert to `{'default': int}`
3. In `merge_datasets()`, look up tolerance from `schedule_df['region'].iloc[0]` with fallback to `'default'`
4. Add to `config.yaml`:
   ```yaml
   merge:
     tolerance_hours:
       EU: 2
       US: 3
       default: 2
   ```

**Acceptance Criteria**:
- [ ] `DataMerger(tolerance_hours={'EU': 2, 'US': 3}).merge_datasets(adsb, eu_sched)` uses 2h tolerance
- [ ] `DataMerger(tolerance_hours=2)` still works (backwards compatible)
- [ ] Config value is read from `config.yaml` in `main.py`

---

## `[x]` TASK-11: Add deterministic tie-breaking & conflict diagnostics

**Files**: `src/merge.py`
**What**: When both Phase A and Phase B match the same ADS-B flight, always prefer Phase A. Log one-to-many conflicts (one ADS-B segment matching multiple schedules) as warnings with flight details.

**Steps**:
1. In final dedup, sort by `['trajectory_id', 'match_quality']` where Phase A sorts before Phase B (already done by string sort — verify)
2. Before dedup, count and log rows where `trajectory_id` appears >1 time: `"CONFLICT: trajectory {id} matched {n} schedules — keeping Phase A"`
3. Add conflict count to `MergeQA.generate_report()`

**Acceptance Criteria**:
- [ ] Given synthetic data where one trajectory matches 2 schedules, the Phase A match is kept
- [ ] Conflict warning appears in logs with trajectory ID and count
- [ ] `merge_qa_report_*.json` contains `"one_to_many_conflicts"` key with integer count
