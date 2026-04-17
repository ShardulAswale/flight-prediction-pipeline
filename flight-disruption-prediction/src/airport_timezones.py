from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd


AIRPORT_TIMEZONE_OVERRIDES = {
    "ABQ": "America/Denver",
    "ALB": "America/New_York",
    "ANC": "America/Anchorage",
    "ATL": "America/New_York",
    "AUS": "America/Chicago",
    "BDL": "America/New_York",
    "BHM": "America/Chicago",
    "BIL": "America/Denver",
    "BIS": "America/Chicago",
    "BLI": "America/Los_Angeles",
    "BNA": "America/Chicago",
    "BOI": "America/Denver",
    "BOS": "America/New_York",
    "BRO": "America/Chicago",
    "BUF": "America/New_York",
    "BUR": "America/Los_Angeles",
    "BWI": "America/New_York",
    "BZN": "America/Denver",
    "CAE": "America/New_York",
    "CHA": "America/New_York",
    "CHS": "America/New_York",
    "CID": "America/Chicago",
    "CLE": "America/New_York",
    "CLT": "America/New_York",
    "CMH": "America/New_York",
    "COS": "America/Denver",
    "CVG": "America/New_York",
    "DAB": "America/New_York",
    "DAL": "America/Chicago",
    "DAY": "America/New_York",
    "DCA": "America/New_York",
    "DEN": "America/Denver",
    "DFW": "America/Chicago",
    "DSM": "America/Chicago",
    "DTW": "America/New_York",
    "ECP": "America/Chicago",
    "ELP": "America/Denver",
    "EUG": "America/Los_Angeles",
    "EWR": "America/New_York",
    "FAT": "America/Los_Angeles",
    "FLL": "America/New_York",
    "GEG": "America/Los_Angeles",
    "GJT": "America/Denver",
    "GRR": "America/New_York",
    "GSO": "America/New_York",
    "GSP": "America/New_York",
    "HNL": "Pacific/Honolulu",
    "HOU": "America/Chicago",
    "HSV": "America/Chicago",
    "IAD": "America/New_York",
    "IAH": "America/Chicago",
    "ICT": "America/Chicago",
    "IND": "America/Indiana/Indianapolis",
    "JAC": "America/Denver",
    "JAX": "America/New_York",
    "JFK": "America/New_York",
    "KOA": "Pacific/Honolulu",
    "LAS": "America/Los_Angeles",
    "LAX": "America/Los_Angeles",
    "LEX": "America/New_York",
    "LIH": "Pacific/Honolulu",
    "LIT": "America/Chicago",
    "LGA": "America/New_York",
    "MCI": "America/Chicago",
    "MCO": "America/New_York",
    "MDT": "America/New_York",
    "MDW": "America/Chicago",
    "MEM": "America/Chicago",
    "MHT": "America/New_York",
    "MIA": "America/New_York",
    "MKE": "America/Chicago",
    "MSN": "America/Chicago",
    "MSP": "America/Chicago",
    "MSY": "America/Chicago",
    "MYR": "America/New_York",
    "OAK": "America/Los_Angeles",
    "OGG": "Pacific/Honolulu",
    "OKC": "America/Chicago",
    "OMA": "America/Chicago",
    "ONT": "America/Los_Angeles",
    "ORD": "America/Chicago",
    "ORF": "America/New_York",
    "PBI": "America/New_York",
    "PDX": "America/Los_Angeles",
    "PHL": "America/New_York",
    "PHX": "America/Phoenix",
    "PIE": "America/New_York",
    "PIT": "America/New_York",
    "PNS": "America/Chicago",
    "PSP": "America/Los_Angeles",
    "PVD": "America/New_York",
    "PWM": "America/New_York",
    "RAP": "America/Denver",
    "RDU": "America/New_York",
    "RIC": "America/New_York",
    "RNO": "America/Los_Angeles",
    "ROC": "America/New_York",
    "RSW": "America/New_York",
    "SAN": "America/Los_Angeles",
    "SAT": "America/Chicago",
    "SAV": "America/New_York",
    "SDF": "America/Chicago",
    "SEA": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
    "SJC": "America/Los_Angeles",
    "SJU": "America/Puerto_Rico",
    "SLC": "America/Denver",
    "SMF": "America/Los_Angeles",
    "SNA": "America/Los_Angeles",
    "STL": "America/Chicago",
    "STT": "America/St_Thomas",
    "SYR": "America/New_York",
    "TPA": "America/New_York",
    "TUL": "America/Chicago",
    "TUS": "America/Phoenix",
    "TYS": "America/New_York",
    "VPS": "America/Chicago",
    "XNA": "America/Chicago",
}


STATE_TIMEZONE_MAP = {
    "AK": "America/Anchorage",
    "AL": "America/Chicago",
    "AR": "America/Chicago",
    "AZ": "America/Phoenix",
    "CA": "America/Los_Angeles",
    "CO": "America/Denver",
    "CT": "America/New_York",
    "DC": "America/New_York",
    "DE": "America/New_York",
    "FL": "America/New_York",
    "GA": "America/New_York",
    "HI": "Pacific/Honolulu",
    "IA": "America/Chicago",
    "ID": "America/Denver",
    "IL": "America/Chicago",
    "IN": "America/Indiana/Indianapolis",
    "KS": "America/Chicago",
    "KY": "America/New_York",
    "LA": "America/Chicago",
    "MA": "America/New_York",
    "MD": "America/New_York",
    "ME": "America/New_York",
    "MI": "America/New_York",
    "MN": "America/Chicago",
    "MO": "America/Chicago",
    "MS": "America/Chicago",
    "MT": "America/Denver",
    "NC": "America/New_York",
    "ND": "America/Chicago",
    "NE": "America/Chicago",
    "NH": "America/New_York",
    "NJ": "America/New_York",
    "NM": "America/Denver",
    "NV": "America/Los_Angeles",
    "NY": "America/New_York",
    "OH": "America/New_York",
    "OK": "America/Chicago",
    "OR": "America/Los_Angeles",
    "PA": "America/New_York",
    "PR": "America/Puerto_Rico",
    "RI": "America/New_York",
    "SC": "America/New_York",
    "SD": "America/Chicago",
    "TN": "America/Chicago",
    "TX": "America/Chicago",
    "UT": "America/Denver",
    "VA": "America/New_York",
    "VI": "America/St_Thomas",
    "VT": "America/New_York",
    "WA": "America/Los_Angeles",
    "WI": "America/Chicago",
    "WV": "America/New_York",
    "WY": "America/Denver",
}


def resolve_airport_timezone(airport_code: object, state_code: object = None) -> str | None:
    airport = str(airport_code).upper().strip() if pd.notna(airport_code) else ""
    state = str(state_code).upper().strip() if pd.notna(state_code) else ""

    if airport in AIRPORT_TIMEZONE_OVERRIDES:
        return AIRPORT_TIMEZONE_OVERRIDES[airport]
    if state in STATE_TIMEZONE_MAP:
        return STATE_TIMEZONE_MAP[state]
    return None


def localize_local_times(
    local_naive: pd.Series,
    airport_codes: pd.Series,
    state_codes: pd.Series | None = None,
) -> pd.Series:
    """Localize naive airport-local timestamps into UTC using airport/state timezone hints."""
    if state_codes is None:
        state_codes = pd.Series(index=local_naive.index, dtype="object")

    tz_names = [
        resolve_airport_timezone(airport_code, state_code)
        for airport_code, state_code in zip(airport_codes, state_codes)
    ]
    tz_series = pd.Series(tz_names, index=local_naive.index, dtype="object")
    utc_series = pd.Series(pd.NaT, index=local_naive.index, dtype="datetime64[ns, UTC]")

    for tz_name in sorted(tz_series.dropna().unique()):
        idx = tz_series[tz_series == tz_name].index
        if len(idx) == 0:
            continue
        localized = (
            local_naive.loc[idx]
            .dt.tz_localize(ZoneInfo(tz_name), ambiguous="NaT", nonexistent="shift_forward")
            .dt.tz_convert("UTC")
        )
        utc_series.loc[idx] = localized

    return utc_series
