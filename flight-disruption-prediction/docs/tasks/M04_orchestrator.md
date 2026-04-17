# Milestone 4 — Orchestrator Fix

> **Status**: `[x]` Complete
> **Priority**: 🟡 MEDIUM
> **Depends on**: M1 (T2 for import fixes)
> **Can run in parallel with**: M3

---

## `[x]` TASK-8: Refactor main.py with --stage CLI flag

**Files**: `main.py`
**What**: Add `argparse` with `--stage` flag accepting: `ingest`, `features`, `merge`, `weather`, `label`, `validate`, `train`, `all`. Each stage is a separate function. `all` runs them sequentially. Add `--dry-run` flag that prints what would run.

**Steps**:
1. Add `argparse.ArgumentParser` at top of `__main__` block
2. Refactor the monolithic `main()` into: `stage_ingest()`, `stage_features()`, `stage_merge()`, `stage_weather()`, `stage_label()`, `stage_validate()`, `stage_train()` (stub)
3. Wire `--stage` to call the right function(s)
4. Add run metadata logging: write `logs/run_{timestamp}.json` with start/end time, stage, row counts

**Acceptance Criteria**:
- [ ] `python main.py --stage features` runs only feature extraction, not ingestion
- [ ] `python main.py --stage all` runs the full pipeline exactly as before
- [ ] `python main.py --dry-run --stage merge` prints "Would run: merge" without executing
- [ ] `logs/run_*.json` file is created with `stage`, `start_time`, `end_time`, `status` keys

---

## `[x]` TASK-9: Add resumable stage support

**Files**: `main.py`
**What**: Before each stage, check if its output file already exists. If so, skip with a log message. Add `--force` flag to override and re-run.

**Steps**:
1. Define an output-file mapping: `{'features': 'trajectory_features.parquet', 'merge': 'ml_dataset_merged.parquet', ...}`
2. At start of each stage function, check `if output_path.exists() and not args.force: skip`
3. Log: `Stage 'features' skipped — output already exists. Use --force to re-run.`

**Acceptance Criteria**:
- [ ] Running `python main.py --stage features` twice: second run logs "skipped" and completes in <2s
- [ ] Running `python main.py --stage features --force` re-executes the stage
- [ ] Output files are identical between forced and non-forced runs
