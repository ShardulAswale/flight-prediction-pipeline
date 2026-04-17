"""
Schema contracts for all pipeline stages.
Single source-of-truth for column definitions, dtypes, null thresholds, and UTC enforcement.
"""

# ── Stage 1: Raw ADS-B State Vectors ─────────────────────────────────────
ADSB_STATES_SCHEMA = {
    'icao24':          {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'callsign':        {'dtype': 'object',   'max_null_pct': 0.10, 'is_utc': False},
    'timestamp':       {'dtype': 'int64',    'max_null_pct': 0.0,  'is_utc': False},
    'latitude':        {'dtype': 'float64',  'max_null_pct': 0.05, 'is_utc': False},
    'longitude':       {'dtype': 'float64',  'max_null_pct': 0.05, 'is_utc': False},
    'baro_altitude':   {'dtype': 'float64',  'max_null_pct': 0.20, 'is_utc': False},
    'velocity':        {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'true_track':      {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'on_ground':       {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
}

# ── Stage 2: Reconstructed Trajectories ──────────────────────────────────
TRAJECTORIES_SCHEMA = {
    'trajectory_id':   {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'icao24':          {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'timestamp':       {'dtype': 'int64',    'max_null_pct': 0.0,  'is_utc': False},
    'latitude':        {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'longitude':       {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'altitude':        {'dtype': 'float64',  'max_null_pct': 0.20, 'is_utc': False},
    'velocity':        {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'heading':         {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'trajectory_quality_status': {'dtype': 'object',  'max_null_pct': 0.0,  'is_utc': False},
    'trajectory_quality_score':  {'dtype': 'float64', 'max_null_pct': 0.0,  'is_utc': False},
    'is_full_flight':           {'dtype': 'bool',    'max_null_pct': 0.0,  'is_utc': False},
    'starts_groundish':         {'dtype': 'bool',    'max_null_pct': 0.0,  'is_utc': False},
    'ends_groundish':           {'dtype': 'bool',    'max_null_pct': 0.0,  'is_utc': False},
    'has_altitude_spike':       {'dtype': 'bool',    'max_null_pct': 0.0,  'is_utc': False},
    'is_mappable':              {'dtype': 'bool',    'max_null_pct': 0.0,  'is_utc': False},
}

# ── Stage 3: Flight-Level Features ───────────────────────────────────────
FEATURES_SCHEMA = {
    'trajectory_id':         {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'icao24':                {'dtype': 'object',   'max_null_pct': 0.05, 'is_utc': False},
    'callsign':              {'dtype': 'object',   'max_null_pct': 0.10, 'is_utc': False},
    'timestamp':             {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'start_time_utc':        {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0,  'is_utc': True},
    'end_time_utc':          {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0,  'is_utc': True},
    'dep_anchor_confidence': {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'arr_anchor_confidence': {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'dep_anchor_is_partial': {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'arr_anchor_is_partial': {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'trajectory_quality_status': {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'trajectory_quality_score':  {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'is_full_flight':            {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'starts_groundish':          {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'ends_groundish':            {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'has_altitude_spike':        {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'is_mappable':               {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'middle_gap_count':          {'dtype': 'int64',    'max_null_pct': 0.0,  'is_utc': False},
    'gap_fraction_of_flight':    {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'route_coverage_fraction':   {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'max_inter_ping_seconds':    {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'median_inter_ping_seconds': {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'ping_interval_cv':          {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'flight_duration':       {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'trajectory_length':     {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'mean_altitude':         {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'altitude_variance':     {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'mean_speed':            {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'max_speed':             {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'speed_std':             {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'vertical_rate_std':     {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'heading_variability':   {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'holding_pattern_count': {'dtype': 'int64',    'max_null_pct': 0.0,  'is_utc': False},
    'altitude_change_count': {'dtype': 'int64',    'max_null_pct': 0.0,  'is_utc': False},
    'unstable_descent_flag': {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
}

# ── Stage 4: Canonical Schedule ──────────────────────────────────────────
CANONICAL_SCHEDULE_SCHEMA = {
    'flight_key':      {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'callsign':        {'dtype': 'object',              'max_null_pct': 0.05, 'is_utc': False},
    'scheduled_dep':   {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0,  'is_utc': True},
    'scheduled_arr':   {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0,  'is_utc': True},
    'origin':          {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'destination':     {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'cancelled':       {'dtype': 'int64',               'max_null_pct': 0.0,  'is_utc': False},
    'source_dataset':  {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
}

# ── Stage 5: Final ML Dataset ────────────────────────────────────────────
ML_DATASET_SCHEMA = {
    # Identifiers & Scheduling
    'flight_key':              {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'region':                  {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'source_dataset':          {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'scheduled_dep':           {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0,  'is_utc': True},
    'scheduled_arr':           {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0,  'is_utc': True},
    'origin':                  {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'destination':             {'dtype': 'object',              'max_null_pct': 0.0,  'is_utc': False},
    'cancelled':               {'dtype': 'int64',               'max_null_pct': 0.0,  'is_utc': False},
    # ADS-B Behavioural Features
    'flight_duration':         {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'trajectory_length':       {'dtype': 'float64',  'max_null_pct': 0.10, 'is_utc': False},
    'mean_speed':              {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'max_speed':               {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'speed_std':               {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'mean_altitude':           {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'altitude_variance':       {'dtype': 'float64',  'max_null_pct': 0.15, 'is_utc': False},
    'vertical_rate_std':       {'dtype': 'float64',  'max_null_pct': 0.20, 'is_utc': False},
    'heading_variability':     {'dtype': 'float64',  'max_null_pct': 0.20, 'is_utc': False},
    'holding_pattern_count':   {'dtype': 'int64',    'max_null_pct': 0.05, 'is_utc': False},
    'altitude_change_count':   {'dtype': 'int64',    'max_null_pct': 0.05, 'is_utc': False},
    'unstable_descent_flag':   {'dtype': 'bool',     'max_null_pct': 0.05, 'is_utc': False},
    # Trajectory completeness / route coverage
    'trajectory_quality_score': {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'is_full_flight':           {'dtype': 'bool',    'max_null_pct': 0.10, 'is_utc': False},
    'dep_anchor_confidence':    {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'arr_anchor_confidence':    {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'route_coverage_fraction':  {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'middle_gap_count':         {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'gap_fraction_of_flight':   {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'max_inter_ping_seconds':   {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'ping_interval_cv':         {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    # Temporal Features
    'dep_hour':                 {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'dep_day_of_week':          {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'dep_month':                {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'is_weekend':               {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'is_peak_hour':             {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    # Airport / route context
    'origin_flight_count':      {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'dest_flight_count':        {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'origin_delay_rate':        {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'dest_delay_rate':          {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'origin_encoded':           {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'dest_encoded':             {'dtype': 'float64', 'max_null_pct': 0.10, 'is_utc': False},
    'route_gc_distance_km':     {'dtype': 'float64', 'max_null_pct': 0.30, 'is_utc': False},
    # Congestion Features
    'origin_flights_1hr':       {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    'dest_flights_1hr':         {'dtype': 'int64',   'max_null_pct': 0.10, 'is_utc': False},
    # Weather Features
    'wind_speed':              {'dtype': 'float64',  'max_null_pct': 0.50, 'is_utc': False},
    'visibility':              {'dtype': 'float64',  'max_null_pct': 0.50, 'is_utc': False},
    'temperature':             {'dtype': 'float64',  'max_null_pct': 0.50, 'is_utc': False},
    'precipitation':           {'dtype': 'float64',  'max_null_pct': 0.50, 'is_utc': False},
    'weather_severity':        {'dtype': 'float64',  'max_null_pct': 0.50, 'is_utc': False},
    'weather_confidence':      {'dtype': 'object',   'max_null_pct': 0.50, 'is_utc': False},
    # Destination Weather
    'wind_speed_dest':         {'dtype': 'float64',  'max_null_pct': 0.80, 'is_utc': False},
    'visibility_dest':         {'dtype': 'float64',  'max_null_pct': 0.80, 'is_utc': False},
    'temperature_dest':        {'dtype': 'float64',  'max_null_pct': 0.80, 'is_utc': False},
    'precipitation_dest':      {'dtype': 'float64',  'max_null_pct': 0.80, 'is_utc': False},
    'weather_severity_dest':   {'dtype': 'float64',  'max_null_pct': 0.80, 'is_utc': False},
    # Route Deviation
    'lateral_deviation_mean_km': {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    'lateral_deviation_max_km':  {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    'lateral_deviation_std_km':  {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    'route_stretch_ratio':       {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    'approach_deviation_km':     {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    'altitude_deviation_mean_m': {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    'altitude_deviation_max_m':  {'dtype': 'float64', 'max_null_pct': 0.95, 'is_utc': False},
    # Quality
    'match_quality':           {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'label_source':            {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'match_score_minutes':     {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'match_anchor':            {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
    'match_confidence':        {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    'training_eligible':       {'dtype': 'bool',     'max_null_pct': 0.0,  'is_utc': False},
    'feature_quality_score':   {'dtype': 'float64',  'max_null_pct': 0.0,  'is_utc': False},
    # Target
    'delay_minutes':           {'dtype': 'float64',  'max_null_pct': 0.30, 'is_utc': False},
    'label':                   {'dtype': 'object',   'max_null_pct': 0.0,  'is_utc': False},
}

# ── Column Order for Final Export (T16) ──────────────────────────────────
ML_DATASET_COLUMN_ORDER = [
    # Identifiers
    'flight_key', 'region', 'source_dataset',
    # Schedule
    'scheduled_dep', 'scheduled_arr', 'origin', 'destination', 'cancelled',
    # ADS-B Features
    'flight_duration', 'trajectory_length',
    'mean_speed', 'max_speed', 'speed_std',
    'mean_altitude', 'altitude_variance', 'vertical_rate_std',
    'heading_variability',
    'holding_pattern_count', 'altitude_change_count', 'unstable_descent_flag',
    # Trajectory completeness / coverage
    'trajectory_quality_score', 'is_full_flight',
    'dep_anchor_confidence', 'arr_anchor_confidence',
    'route_coverage_fraction', 'middle_gap_count',
    'gap_fraction_of_flight', 'max_inter_ping_seconds', 'ping_interval_cv',
    # Temporal Features (Task 3.2)
    'dep_hour', 'dep_day_of_week', 'dep_month', 'is_weekend', 'is_peak_hour',
    # Airport Features (Task 3.3)
    'origin_flight_count', 'dest_flight_count',
    'origin_delay_rate', 'dest_delay_rate', 'origin_encoded', 'dest_encoded', 'route_gc_distance_km',
    # Congestion Features (Task 3.4)
    'origin_flights_1hr', 'dest_flights_1hr',
    # Route Deviation Features (Task 4.1/4.2)
    'lateral_deviation_mean_km', 'lateral_deviation_max_km', 'lateral_deviation_std_km',
    'route_stretch_ratio', 'approach_deviation_km',
    'altitude_deviation_mean_m', 'altitude_deviation_max_m',
    # Origin Weather
    'wind_speed', 'visibility', 'temperature', 'precipitation',
    'weather_severity', 'weather_confidence',
    # Destination Weather (Task 2.3)
    'wind_speed_dest', 'visibility_dest', 'temperature_dest', 'precipitation_dest',
    'weather_severity_dest',
    # Quality
    'match_quality', 'label_source', 'match_score_minutes', 'match_anchor', 'match_confidence',
    'training_eligible', 'feature_quality_score',
    # Target
    'delay_minutes', 'label',
]

# ── Feature columns for ML training (excludes identifiers and targets) ───
ML_FEATURE_COLUMNS = [
    # ADS-B behavioural
    'flight_duration', 'trajectory_length',
    'mean_speed', 'max_speed', 'speed_std',
    'mean_altitude', 'altitude_variance', 'vertical_rate_std',
    'heading_variability',
    'holding_pattern_count', 'altitude_change_count', 'unstable_descent_flag',
    # Completeness / coverage
    'trajectory_quality_score', 'is_full_flight',
    'dep_anchor_confidence', 'arr_anchor_confidence',
    'route_coverage_fraction', 'middle_gap_count',
    'gap_fraction_of_flight', 'max_inter_ping_seconds', 'ping_interval_cv',
    # Temporal
    'dep_hour', 'dep_day_of_week', 'dep_month', 'is_weekend', 'is_peak_hour',
    # Airport
    'origin_flight_count', 'dest_flight_count',
    'origin_delay_rate', 'dest_delay_rate', 'origin_encoded', 'dest_encoded', 'route_gc_distance_km',
    # Congestion
    'origin_flights_1hr', 'dest_flights_1hr',
    # Route deviation
    'lateral_deviation_mean_km', 'lateral_deviation_max_km', 'lateral_deviation_std_km',
    'route_stretch_ratio', 'approach_deviation_km',
    'altitude_deviation_mean_m', 'altitude_deviation_max_m',
    # Origin weather
    'wind_speed', 'visibility', 'temperature', 'precipitation',
    'weather_severity',
    # Destination weather
    'wind_speed_dest', 'visibility_dest', 'temperature_dest', 'precipitation_dest',
    'weather_severity_dest',
]

# ── Feature categories (for grouped importance analysis, Task 5.4) ───────
ML_FEATURE_CATEGORIES = {
    'trajectory_shape': [
        'flight_duration', 'trajectory_length', 'mean_speed', 'max_speed',
        'speed_std', 'mean_altitude', 'altitude_variance', 'vertical_rate_std',
        'heading_variability', 'holding_pattern_count', 'altitude_change_count',
        'unstable_descent_flag',
    ],
    'trajectory_completeness': [
        'trajectory_quality_score', 'is_full_flight',
        'dep_anchor_confidence', 'arr_anchor_confidence',
        'route_coverage_fraction', 'middle_gap_count',
        'gap_fraction_of_flight', 'max_inter_ping_seconds', 'ping_interval_cv',
    ],
    'temporal': ['dep_hour', 'dep_day_of_week', 'dep_month', 'is_weekend', 'is_peak_hour'],
    'airport': [
        'origin_flight_count', 'dest_flight_count',
        'origin_delay_rate', 'dest_delay_rate', 'origin_encoded', 'dest_encoded', 'route_gc_distance_km',
    ],
    'congestion': ['origin_flights_1hr', 'dest_flights_1hr'],
    'route_deviation': [
        'lateral_deviation_mean_km', 'lateral_deviation_max_km',
        'lateral_deviation_std_km', 'route_stretch_ratio',
        'approach_deviation_km', 'altitude_deviation_mean_m',
        'altitude_deviation_max_m',
    ],
    'weather_origin': ['wind_speed', 'visibility', 'temperature', 'precipitation', 'weather_severity'],
    'weather_dest': [
        'wind_speed_dest', 'visibility_dest', 'temperature_dest',
        'precipitation_dest', 'weather_severity_dest',
    ],
}

# ── Identifier / provenance columns to drop before training ──────────────
ML_DROP_COLUMNS = [
    'flight_key', 'region', 'source_dataset',
    'scheduled_dep', 'scheduled_arr', 'origin', 'destination',
    'cancelled', 'match_quality', 'weather_confidence',
    'label_source', 'match_score_minutes', 'match_anchor', 'match_confidence',
    'training_eligible', 'feature_quality_score',
]
