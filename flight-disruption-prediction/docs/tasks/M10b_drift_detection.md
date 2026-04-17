# Milestone 10b — Drift Detection (Real Data)

> **Status**: `[~]` In Progress (needs >=2 monthly partitions in final dataset)
> **Priority**: 🟠 MEDIUM  
> **Depends on**: M0 (T0B for multi-month data), M8 (T17 for trained models)

---

## `[x]` TASK-30: Create drift detector module (real data)

**Files**: `src/drift_detector.py` [NEW]  
**What**: Create `DriftDetector` class with PSI and KS test methods for detecting feature drift across real month-over-month data from the multi-month ADS-B + schedule pipeline. Uses `ml_dataset.parquet` partitioned by `scheduled_dep` month.

**Steps**:
1. Create `src/drift_detector.py` with class `DriftDetector`
2. `compute_psi(reference, current, bins=10) -> float`: standard PSI formula with epsilon for zero-count bins
3. `compute_ks(reference, current) -> float`: `scipy.stats.ks_2samp` statistic + p-value
4. `monthly_drift_report(df, date_col='scheduled_dep', feature_cols=None) -> DataFrame`:
   - Partition by `date_col.dt.to_period('M')`
   - First month = reference distribution
   - Compute PSI and KS for each subsequent month × each feature
   - Flag features where PSI > 0.2 or KS p-value < 0.05
5. `stability_ranking(drift_df) -> DataFrame`: rank features by average PSI ascending
6. `flag_high_importance_unstable(drift_df, importance_df) -> DataFrame`: cross-reference importance with drift
7. Save reports to `logs/drift_report.json` and `logs/drift_summary.json`
8. Generate `outputs/drift_heatmap.png`: seaborn heatmap (features × month, color = PSI)

**Acceptance Criteria**:
- [ ] Runs on real `ml_dataset.parquet` with ≥3 months of data
- [ ] `compute_psi(ref, curr)` returns float ≥0; ~0 for identical distributions
- [ ] `monthly_drift_report()` returns DataFrame with: `feature`, `month`, `psi`, `ks_stat`, `ks_pvalue`, `drift_alert`
- [ ] `drift_alert` is `True` when PSI > 0.2 or KS p-value < 0.05
- [ ] `stability_ranking()` returns features sorted by average PSI ascending
- [ ] `outputs/drift_heatmap.png` is a valid PNG
- [ ] `logs/drift_report.json` and `logs/drift_summary.json` are valid JSON

---

## `[x]` TASK-31: Wire drift detection into pipeline & orchestrator

**Files**: `main.py`, `src/drift_detector.py`  
**What**: Add a `drift` stage to the CLI orchestrator. After model training, run drift detection on ML dataset and cross-reference with feature importance.

**Steps**:
1. Add `--stage drift` option to `main.py` argparse
2. In `stage_drift()`: load `ml_dataset.parquet`, call `monthly_drift_report()`, `stability_ranking()`, `flag_high_importance_unstable()`
3. Print drift summary to console and save reports
4. If any feature has `drift_alert=True` AND is in top-10 importance, emit WARNING log

**Acceptance Criteria**:
- [ ] `python main.py --stage drift` runs drift analysis and creates report files
- [ ] Console output shows which features are drifting
- [ ] Warning log emitted for high-importance + unstable features
- [ ] `--stage all` includes drift detection after training
