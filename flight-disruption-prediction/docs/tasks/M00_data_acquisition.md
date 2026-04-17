# Milestone 0 — Data Acquisition (CRITICAL PATH)

> **Status**: `[~]` In Progress (blocked on OpenSky historical 403; METAR partial)
> **Priority**: 🔴 CRITICAL — Pipeline produces 0 merged rows without this
> **Depends on**: Nothing (start immediately)

Without this milestone, the merge produces 0 rows and the entire ML pipeline is non-functional. This must be started immediately — OpenSky API rate limits mean ADS-B backfill takes days.

---

## `[x]` TASK-0A: Build ADS-B backfill CLI with date-range support

**Files**: `src/opensky_client.py`, `src/data_ingestion.py`
**What**: Add a `--backfill-from YYYY-MM-DD --backfill-to YYYY-MM-DD` CLI to the OpenSky client that fetches historical ADS-B state vectors day-by-day, partitions output into `data/raw/opensky/year=YYYY/month=MM/states_YYYYMMDD.parquet`, and uses the ManifestManager to track progress (skipping already-downloaded days on resume).

**Steps**:
1. Add `backfill_date_range(start_date, end_date)` method to OpenSky client
2. For each day in range: call OpenSky `/states/all` with `begin` and `end` epoch params (UTC midnight boundaries)
3. Partition output: `data/raw/opensky/year={YYYY}/month={MM}/states_{YYYYMMDD}.parquet`
4. Wire ManifestManager to log each day: `(source='opensky', year, month, status, row_count, file_path)`
5. On resume: skip days where manifest shows `status='SUCCESS'`
6. Add CLI: `python src/data_ingestion.py --source opensky_backfill --backfill-from 2025-01-01 --backfill-to 2025-11-30`
7. Respect OpenSky rate limits: max 1 request per 5 seconds for authenticated users, with exponential backoff

**Acceptance Criteria**:
- [ ] `python src/data_ingestion.py --source opensky_backfill --backfill-from 2025-06-01 --backfill-to 2025-06-03` creates 3 daily Parquet files
- [ ] Files are partitioned at `data/raw/opensky/year=2025/month=06/states_20250601.parquet`
- [ ] Running the same command again skips all 3 days (manifest check), completing in <5 seconds
- [ ] Rate limiting: no more than 12 requests per minute
- [ ] Manifest CSV has entries for each day with status and row count

---

## `[~]` TASK-0B: Fetch overlapping ADS-B data for schedule period

**Files**: `data/raw/opensky/` [OUTPUT], `data/raw/manifest.csv` [OUTPUT]
**What**: Execute the backfill CLI (from TASK-0A) to fetch ADS-B data for January–November 2025, matching the existing BTS and Eurocontrol schedule coverage. This is a long-running data acquisition task (rate-limited, estimated 3–7 days). Focus on the European bbox first (where Eurocontrol schedules are densest).

**Steps**:
1. Run: `python src/data_ingestion.py --source opensky_backfill --backfill-from 2025-01-01 --backfill-to 2025-11-30`
2. Monitor progress via `data/raw/manifest.csv` — check for FAILED entries and retry
3. If OpenSky historical access is restricted (free tier limits), prioritise 3 representative months: March, June, September 2025
4. Validate: load a sample day and check that `icao24`, `callsign`, `timestamp`, `latitude`, `longitude` columns are present

**Acceptance Criteria**:
- [ ] At minimum, 3 months of ADS-B data are downloaded (≥90 daily Parquet files)
- [ ] Each daily file has >0 rows with valid `icao24` and `timestamp` values
- [ ] Manifest shows ≥90 entries with `status='SUCCESS'`
- [ ] Total ADS-B data overlaps with both BTS (US flights) and Eurocontrol (EU flights) schedules
- [ ] `data/raw/opensky/` directory size is >100 MB

---

## `[~]` TASK-0C: Rebuild trajectories from multi-month ADS-B data

**Files**: `src/trajectory_builder.py`, `main.py`
**What**: Update the trajectory builder to process the new partitioned multi-month ADS-B data. Load daily Parquet files incrementally, build flight segments, and save a combined `trajectories.parquet`.

**Steps**:
1. In `main.py` step 3, replace single-file load with a glob: `data/raw/opensky/**/states_*.parquet`
2. Load daily files incrementally (concat in batches to manage memory)
3. Pass combined df to `TrajectoryBuilder.build_flight_segments()`
4. Save to `data/processed/trajectories.parquet`
5. Log: total pings loaded, trajectories reconstructed, time span covered

**Acceptance Criteria**:
- [ ] `trajectories.parquet` contains trajectories spanning ≥3 months
- [ ] `trajectory_features.parquet` has >100 unique trajectories (vs current 1,844 from 1 week)
- [ ] Merge step now produces >0 matched flights (the core data gap is resolved)
- [ ] Memory usage stays under 8 GB during trajectory build (incremental loading)

---

## `[~]` TASK-0D: Acquire and integrate real METAR weather data

**Files**: `src/weather.py`, `scripts/fetch_weather.py` [NEW]
**What**: The current `metar_sample.csv` is a stub. Create a script to download real METAR weather data from Iowa Environmental Mesonet (IEM) for the airports and date range covered by the schedule data.

**Steps**:
1. Create `scripts/fetch_weather.py` that calls IEM ASOS API: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py`
2. Parameters: station list (top 50 ICAO codes from our data), date range (2025-01-01 to 2025-11-30), variables (tmpf, dwpf, sknt, vsby, p01i)
3. Map IEM columns to our schema: `wind_speed`, `visibility`, `temperature`, `precipitation`, `airport_code`, `timestamp`
4. Save to `data/raw/metar_2025.parquet` (partitioned by month if large)
5. Update `config.yaml` to point `metar_data_file` at the new file

**Acceptance Criteria**:
- [ ] `data/raw/metar_2025.parquet` exists with >100K rows
- [ ] Columns match weather schema: `airport_code`, `timestamp`, `wind_speed`, `visibility`, `temperature`, `precipitation`
- [ ] Timestamps are UTC and cover Jan–Nov 2025
- [ ] At least 30 unique airport codes are represented
- [ ] `WeatherIntegrator.add_weather_features()` produces non-null weather matches for >50% of flights
