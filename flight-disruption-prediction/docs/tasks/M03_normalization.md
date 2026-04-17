# Milestone 3 — Source Adapters & Normalization

> **Status**: `[x]` Complete
> **Priority**: 🟡 HIGH
> **Depends on**: M2 (T3 for schemas)

---

## `[x]` TASK-5: Wire ScheduleNormalizer into BTS combiner output

**Files**: `src/data_ingestion.py`, `src/normalization.py`
**What**: Currently `BTSCombiner.combine_csvs()` calls `preprocess_bts()` from `utils.py` which does a partial mapping. Wire `ScheduleNormalizer.normalize_bts()` as the final transform step before saving, so the output Parquet conforms to the canonical schedule schema.

**Steps**:
1. In `BTSCombiner.combine_csvs()`, after concatenation and dedup, call `ScheduleNormalizer.normalize_bts(combined_df)`
2. The normalized df is what gets saved to Parquet
3. Ensure `normalize_bts()` handles the columns output by `preprocess_bts()` (the raw column names like `FL_DATE` may already be renamed — verify and adjust mapping if needed)

**Acceptance Criteria**:
- [ ] Output Parquet from `BTSCombiner` contains exactly the columns: `flight_key, callsign, scheduled_dep, scheduled_arr, origin, destination, cancelled, source_dataset`
- [ ] `scheduled_dep` and `scheduled_arr` are `datetime64[ns, UTC]`
- [ ] No rows lost vs. the pre-normalization count (beyond intentional dedup)

---

## `[x]` TASK-6: Wire ScheduleNormalizer into Eurocontrol combiner output

**Files**: `src/data_ingestion.py`, `src/normalization.py`
**What**: Same as TASK-5 but for `EuroCombiner.combine_parquets()`. Replace the inline `preprocess_eurocontrol()` call with `ScheduleNormalizer.normalize_eurocontrol()` so the combined Euro dataset also outputs canonical schema.

**Steps**:
1. In `EuroCombiner.combine_parquets()`, after loading each monthly file, call `ScheduleNormalizer.normalize_eurocontrol(df)` instead of `preprocess_eurocontrol(df)`
2. Adjust `normalize_eurocontrol()` if needed to handle OPDI column names (`flt_id`, `adep`, `ades`, `first_seen`, `last_seen`)
3. Remove `region` assignment from combiner (it's now in `source_dataset`)

**Acceptance Criteria**:
- [ ] Output Parquet from `EuroCombiner` contains exactly the canonical columns
- [ ] `DataValidator(mode='fail_fast').validate(df, CANONICAL_SCHEDULE_SCHEMA, 'euro')` passes
- [ ] Row count matches sum of monthly file row counts (minus dedup)

---

## `[x]` TASK-7: Expand IATA-to-ICAO lookup table

**Files**: `src/normalization.py`
**What**: The current `IATA_TO_ICAO` dict has only 30 US airports. Expand to cover the top 100 US airports and add the top 50 European airports for cross-referencing.

**Steps**:
1. Research and add the additional airport codes to `IATA_TO_ICAO` in `ScheduleNormalizer`
2. Add European airports (e.g., `'LHR': 'EGLL'`, `'CDG': 'LFPG'`, etc.)
3. Add a fallback log warning when a code is not found in the lookup

**Acceptance Criteria**:
- [ ] `len(ScheduleNormalizer.IATA_TO_ICAO) >= 130`
- [ ] All top 100 US airports (by 2024 enplanements) are present
- [ ] European airports: LHR, CDG, FRA, AMS, MAD, BCN, FCO, IST, MUC, ZRH are present
- [ ] Unknown codes produce a log warning (not a crash)
