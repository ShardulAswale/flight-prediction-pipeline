import logging

import numpy as np
import pandas as pd

from src.airport_timezones import localize_local_times

logger = logging.getLogger(__name__)


class ScheduleNormalizer:
    """
    Normalizes disparate external schedule sources (BTS, Eurocontrol) into a strict Canonical Schema.
    """

    IATA_TO_ICAO = {
        'ATL': 'KATL', 'ORD': 'KORD', 'DFW': 'KDFW', 'DEN': 'KDEN', 'JFK': 'KJFK',
        'LAX': 'KLAX', 'SFO': 'KSFO', 'LAS': 'KLAS', 'SEA': 'KSEA', 'CLT': 'KCLT',
        'MCO': 'KMCO', 'MIA': 'KMIA', 'PHX': 'KPHX', 'IAH': 'KIAH', 'BOS': 'KBOS',
        'EWR': 'KEWR', 'MSP': 'KMSP', 'DTW': 'KDTW', 'PHL': 'KPHL', 'LGA': 'KLGA',
        'BWI': 'KBWI', 'SLC': 'KSLC', 'SAN': 'KSAN', 'IAD': 'KIAD', 'DCA': 'KDCA',
        'MDW': 'KMDW', 'TPA': 'KTPA', 'FLL': 'KFLL', 'PDX': 'KPDX', 'HNL': 'PHNL',
        'STL': 'KSTL', 'BNA': 'KBNA', 'AUS': 'KAUS', 'OAK': 'KOAK', 'SMF': 'KSMF',
        'RDU': 'KRDU', 'SJC': 'KSJC', 'SNA': 'KSNA', 'MCI': 'KMCI', 'IND': 'KIND',
        'CLE': 'KCLE', 'PIT': 'KPIT', 'CMH': 'KCMH', 'SAT': 'KSAT', 'RSW': 'KRSW',
        'CVG': 'KCVG', 'MKE': 'KMKE', 'JAX': 'KJAX', 'OGG': 'PHOG', 'PBI': 'KPBI',
        'ABQ': 'KABQ', 'ANC': 'PANC', 'BUF': 'KBUF', 'BUR': 'KBUR', 'ONT': 'KONT',
        'RNO': 'KRNO', 'OMA': 'KOMA', 'ORF': 'KORF', 'RIC': 'KRIC', 'TUS': 'KTUS',
        'ELP': 'KELP', 'SDF': 'KSDF', 'MEM': 'KMEM', 'BHM': 'KBHM', 'BOI': 'KBOI',
        'GEG': 'KGEG', 'LIT': 'KLIT', 'PVD': 'KPVD', 'GRR': 'KGRR', 'OKC': 'KOKC',
        'TUL': 'KTUL', 'PSP': 'KPSP', 'KOA': 'PHKO', 'LIH': 'PHLI', 'SYR': 'KSYR',
        'ALB': 'KALB', 'ROC': 'KROC', 'CHS': 'KCHS', 'GSP': 'KGSP', 'MSN': 'KMSN',
        'DAY': 'KDAY', 'SAV': 'KSAV', 'MYR': 'KMYR', 'DSM': 'KDSM', 'MSY': 'KMSY',
        'HOU': 'KHOU', 'DAL': 'KDAL', 'SJU': 'TJSJ', 'STT': 'TIST', 'ORH': 'KORH',
        'ISP': 'KISP', 'BDL': 'KBDL', 'GSO': 'KGSO', 'ICT': 'KICT', 'XNA': 'KXNA',
        'FAT': 'KFAT', 'COS': 'KCOS', 'HSV': 'KHSV', 'PNS': 'KPNS', 'GNV': 'KGNV',
        'LHR': 'EGLL', 'CDG': 'LFPG', 'FRA': 'EDDF', 'AMS': 'EHAM', 'MAD': 'LEMD',
        'BCN': 'LEBL', 'FCO': 'LIRF', 'IST': 'LTFM', 'MUC': 'EDDM', 'ZRH': 'LSZH',
        'LGW': 'EGKK', 'BRU': 'EBBR', 'VIE': 'LOWW', 'DUB': 'EIDW', 'CPH': 'EKCH',
        'OSL': 'ENGM', 'ARN': 'ESSA', 'LIS': 'LPPT', 'HEL': 'EFHK', 'ATH': 'LGAV',
        'MXP': 'LIMC', 'WAW': 'EPWA', 'PRG': 'LKPR', 'BUD': 'LHBP', 'EDI': 'EGPH',
        'MAN': 'EGCC', 'STN': 'EGSS', 'HAM': 'EDDH', 'DUS': 'EDDL', 'SVO': 'UUEE',
        'DME': 'UUDD', 'LED': 'ULLI', 'KBP': 'UKBB', 'OTP': 'LROP', 'SOF': 'LBSF',
        'BEG': 'LYBE', 'ZAG': 'LDZA', 'LJU': 'LJLJ', 'TLL': 'EETN', 'RIX': 'EVRA',
        'VNO': 'EYVI', 'GVA': 'LSGG', 'NCE': 'LFMN', 'LYS': 'LFLL', 'TXL': 'EDDT',
        'BER': 'EDDB', 'CGN': 'EDDK', 'PMI': 'LEPA', 'AGP': 'LEMG', 'ALC': 'LEAL',
    }
    _WARNED_UNKNOWN_IATA: set[str] = set()

    @staticmethod
    def canonical_schema() -> dict:
        return {
            'flight_key': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
            'callsign': {'dtype': 'object', 'max_null_pct': 0.05, 'is_utc': False},
            'scheduled_dep': {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0, 'is_utc': True},
            'scheduled_arr': {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0, 'is_utc': True},
            'origin': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
            'destination': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
            'cancelled': {'dtype': 'int64', 'max_null_pct': 0.0, 'is_utc': False},
            'source_dataset': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
        }

    @classmethod
    def _iata_to_icao(cls, code: str) -> str:
        if code in cls.IATA_TO_ICAO:
            return cls.IATA_TO_ICAO[code]
        if isinstance(code, str) and len(code) == 3:
            if code not in cls._WARNED_UNKNOWN_IATA:
                logger.warning(f"Unknown IATA airport code '{code}' - using fallback 'K{code}'")
                cls._WARNED_UNKNOWN_IATA.add(code)
            return f"K{code}"
        logger.warning(f"Cannot convert airport code '{code}' to ICAO")
        return str(code)

    @staticmethod
    def _parse_bts_local_clock(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
        numeric_time = pd.to_numeric(time_series, errors='coerce')
        date_part = pd.to_datetime(date_series, errors='coerce')

        valid_mask = numeric_time.notna() & date_part.notna()
        numeric_time = numeric_time.fillna(0).astype(int)
        rollover_mask = numeric_time.eq(2400)
        numeric_time = numeric_time.mask(rollover_mask, 0)

        hours = numeric_time // 100
        minutes = numeric_time % 100
        bad_clock = (hours > 23) | (minutes > 59)

        parsed = (
            date_part
            + pd.to_timedelta(rollover_mask.astype(int), unit='D')
            + pd.to_timedelta(hours, unit='h')
            + pd.to_timedelta(minutes, unit='m')
        )
        parsed = parsed.mask(~valid_mask | bad_clock)
        return parsed

    @classmethod
    def _local_bts_time_to_utc(
        cls,
        date_series: pd.Series,
        time_series: pd.Series,
        airport_series: pd.Series,
        state_series: pd.Series | None = None,
    ) -> pd.Series:
        local_naive = cls._parse_bts_local_clock(date_series, time_series)
        return localize_local_times(local_naive, airport_series, state_series)

    @classmethod
    def _derive_bts_arrival_utc(
        cls,
        date_series: pd.Series,
        arr_time_series: pd.Series | None,
        airport_series: pd.Series,
        state_series: pd.Series | None,
        dep_utc_series: pd.Series | None,
        elapsed_series: pd.Series | None,
    ) -> pd.Series:
        arrival_utc = pd.Series(pd.NaT, index=date_series.index, dtype='datetime64[ns, UTC]')

        if dep_utc_series is not None and elapsed_series is not None:
            elapsed_minutes = pd.to_numeric(elapsed_series, errors='coerce')
            mask = dep_utc_series.notna() & elapsed_minutes.notna()
            if mask.any():
                arrival_utc.loc[mask] = dep_utc_series.loc[mask] + pd.to_timedelta(elapsed_minutes.loc[mask], unit='m')

        if arr_time_series is not None:
            fallback_utc = cls._local_bts_time_to_utc(
                date_series=date_series,
                time_series=arr_time_series,
                airport_series=airport_series,
                state_series=state_series,
            )
            missing_mask = arrival_utc.isna() & fallback_utc.notna()
            if missing_mask.any():
                arrival_utc.loc[missing_mask] = fallback_utc.loc[missing_mask]

        if dep_utc_series is not None:
            too_early = arrival_utc.notna() & dep_utc_series.notna() & ((arrival_utc - dep_utc_series).dt.total_seconds() < -6 * 3600)
            if too_early.any():
                arrival_utc.loc[too_early] = arrival_utc.loc[too_early] + pd.Timedelta(days=1)

        return arrival_utc

    @classmethod
    def normalize_bts(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Normalizing BTS dataframe ({len(df)} rows)")
        df = df.copy()

        carrier_col = next(
            (col for col in ['OP_UNIQUE_CARRIER', 'IATA_CODE_Reporting_Airline', 'Reporting_Airline'] if col in df.columns),
            None,
        )
        flight_num_col = next(
            (col for col in ['OP_CARRIER_FL_NUM', 'Flight_Number_Reporting_Airline'] if col in df.columns),
            None,
        )
        if carrier_col and flight_num_col:
            df['callsign'] = df[carrier_col].astype('string').fillna('') + df[flight_num_col].astype('string').fillna('')
        elif 'callsign' not in df.columns:
            df['callsign'] = 'UNKNOWN'

        date_col = next((col for col in ['FL_DATE', 'FlightDate'] if col in df.columns), None)
        origin_col = next((col for col in ['ORIGIN', 'Origin', 'origin_airport'] if col in df.columns), None)
        destination_col = next((col for col in ['DEST', 'Dest', 'destination_airport'] if col in df.columns), None)
        origin_state_col = next((col for col in ['OriginState'] if col in df.columns), None)
        destination_state_col = next((col for col in ['DestState'] if col in df.columns), None)

        origin_series = df[origin_col] if origin_col else pd.Series(index=df.index, dtype='object')
        destination_series = df[destination_col] if destination_col else pd.Series(index=df.index, dtype='object')
        origin_state_series = df[origin_state_col] if origin_state_col else pd.Series(index=df.index, dtype='object')
        destination_state_series = df[destination_state_col] if destination_state_col else pd.Series(index=df.index, dtype='object')

        if date_col:
            dep_sched_col = next((col for col in ['CRS_DEP_TIME', 'CRSDepTime'] if col in df.columns), None)
            arr_sched_col = next((col for col in ['CRS_ARR_TIME', 'CRSArrTime'] if col in df.columns), None)
            dep_actual_col = next((col for col in ['DEP_TIME', 'DepTime'] if col in df.columns), None)
            arr_actual_col = next((col for col in ['ARR_TIME', 'ArrTime'] if col in df.columns), None)
            sched_elapsed_col = next((col for col in ['CRSElapsedTime'] if col in df.columns), None)
            actual_elapsed_col = next((col for col in ['ActualElapsedTime'] if col in df.columns), None)

            if dep_sched_col:
                df['scheduled_dep_utc'] = cls._local_bts_time_to_utc(
                    df[date_col], df[dep_sched_col], origin_series, origin_state_series
                )
            if dep_actual_col:
                df['actual_dep_utc'] = cls._local_bts_time_to_utc(
                    df[date_col], df[dep_actual_col], origin_series, origin_state_series
                )
            if arr_sched_col:
                df['scheduled_arr_utc'] = cls._derive_bts_arrival_utc(
                    df[date_col],
                    df[arr_sched_col],
                    destination_series,
                    destination_state_series,
                    df.get('scheduled_dep_utc'),
                    df[sched_elapsed_col] if sched_elapsed_col else None,
                )
            if arr_actual_col:
                df['actual_arr_utc'] = cls._derive_bts_arrival_utc(
                    df[date_col],
                    df[arr_actual_col],
                    destination_series,
                    destination_state_series,
                    df.get('actual_dep_utc'),
                    df[actual_elapsed_col] if actual_elapsed_col else None,
                )
        else:
            for col in ['scheduled_dep_utc', 'scheduled_arr_utc', 'actual_dep_utc', 'actual_arr_utc']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

        if origin_col:
            df['origin'] = df[origin_col].apply(cls._iata_to_icao)
            df['Origin'] = df[origin_col].astype('string')
        if destination_col:
            df['destination'] = df[destination_col].apply(cls._iata_to_icao)
            df['Dest'] = df[destination_col].astype('string')
        if origin_state_col:
            df['origin_state'] = df[origin_state_col].astype('string')
        if destination_state_col:
            df['destination_state'] = df[destination_state_col].astype('string')

        cancelled_col = next((col for col in ['CANCELLED', 'Cancelled', 'cancelled'] if col in df.columns), None)
        if cancelled_col:
            df['cancelled'] = pd.to_numeric(df[cancelled_col], errors='coerce').fillna(0).astype('int64')
        else:
            df['cancelled'] = 0
        df['Cancelled'] = df['cancelled'].astype('int64')

        dep_delay_col = next((col for col in ['DepDelay'] if col in df.columns), None)
        if dep_delay_col:
            df['DepDelay'] = pd.to_numeric(df[dep_delay_col], errors='coerce')
        elif {'actual_dep_utc', 'scheduled_dep_utc'}.issubset(df.columns):
            df['DepDelay'] = (df['actual_dep_utc'] - df['scheduled_dep_utc']).dt.total_seconds().div(60)
        else:
            df['DepDelay'] = np.nan

        dep_delay_minutes_col = next((col for col in ['DepDelayMinutes'] if col in df.columns), None)
        if dep_delay_minutes_col:
            df['DepDelayMinutes'] = pd.to_numeric(df[dep_delay_minutes_col], errors='coerce')

        if date_col:
            df['service_day_local'] = pd.to_datetime(df[date_col], errors='coerce').dt.normalize()
        elif 'scheduled_dep_utc' in df.columns:
            df['service_day_local'] = pd.to_datetime(df['scheduled_dep_utc'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
        else:
            df['service_day_local'] = pd.NaT

        if 'scheduled_dep_utc' in df.columns:
            df['service_day_utc'] = pd.to_datetime(df['scheduled_dep_utc'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
            df['scheduled_dep'] = df['scheduled_dep_utc']
        if 'scheduled_arr_utc' in df.columns:
            df['scheduled_arr'] = df['scheduled_arr_utc']
        if 'actual_dep_utc' in df.columns:
            df['actual_dep'] = df['actual_dep_utc']
        if 'actual_arr_utc' in df.columns:
            df['actual_arr'] = df['actual_arr_utc']

        if 'service_day_local' in df.columns:
            df['flight_key'] = df['callsign'].astype('string') + '_' + df['service_day_local'].dt.strftime('%Y%m%d')

        df['source_dataset'] = 'bts'

        cols_to_keep = list(cls.canonical_schema().keys()) + [
            'scheduled_dep_utc', 'scheduled_arr_utc', 'actual_dep_utc', 'actual_arr_utc',
            'service_day_local', 'service_day_utc',
            'actual_dep', 'actual_arr',
            'origin_state', 'destination_state',
            'Reporting_Airline', 'IATA_CODE_Reporting_Airline', 'Flight_Number_Reporting_Airline',
            'Origin', 'Dest', 'Cancelled', 'DepDelay', 'DepDelayMinutes'
        ]
        valid_cols = [col for col in cols_to_keep if col in df.columns]
        return df[valid_cols]

    @classmethod
    def normalize_eurocontrol(cls, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Normalizing Eurocontrol dataframe ({len(df)} rows)")
        df = df.copy()

        col_map_variants = [
            (['flt_id', 'Callsign', 'callsign'], 'callsign'),
            (['adep', 'ADEP', 'origin', 'origin_airport'], 'origin'),
            (['ades', 'ADES', 'destination', 'destination_airport'], 'destination'),
        ]
        for source_cols, target in col_map_variants:
            if target not in df.columns:
                for source_col in source_cols:
                    if source_col in df.columns:
                        df[target] = df[source_col]
                        break

        if 'ScheduledDep' in df.columns:
            df['scheduled_dep'] = pd.to_datetime(df['ScheduledDep'], utc=True, errors='coerce')
        elif 'scheduled_dep' in df.columns:
            df['scheduled_dep'] = pd.to_datetime(df['scheduled_dep'], utc=True, errors='coerce')
        elif 'first_seen' in df.columns:
            df['scheduled_dep'] = pd.to_datetime(df['first_seen'], utc=True, errors='coerce')
        else:
            df['scheduled_dep'] = pd.NaT

        if 'ScheduledArr' in df.columns:
            df['scheduled_arr'] = pd.to_datetime(df['ScheduledArr'], utc=True, errors='coerce')
        elif 'scheduled_arr' in df.columns:
            df['scheduled_arr'] = pd.to_datetime(df['scheduled_arr'], utc=True, errors='coerce')
        elif 'last_seen' in df.columns:
            df['scheduled_arr'] = pd.to_datetime(df['last_seen'], utc=True, errors='coerce')
        else:
            df['scheduled_arr'] = pd.NaT

        if 'actual_dep' in df.columns:
            df['actual_dep'] = pd.to_datetime(df['actual_dep'], utc=True, errors='coerce')
        else:
            df['actual_dep'] = pd.NaT

        if 'actual_arr' in df.columns:
            df['actual_arr'] = pd.to_datetime(df['actual_arr'], utc=True, errors='coerce')
        else:
            df['actual_arr'] = pd.NaT

        if 'dof' in df.columns:
            df['service_day_local'] = pd.to_datetime(df['dof'], errors='coerce').dt.normalize()
        else:
            df['service_day_local'] = pd.to_datetime(df['scheduled_dep'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()
        df['service_day_utc'] = pd.to_datetime(df['scheduled_dep'], utc=True, errors='coerce').dt.tz_localize(None).dt.normalize()

        if 'callsign' in df.columns and 'service_day_local' in df.columns:
            df['flight_key'] = df['callsign'].astype(str) + '_' + df['service_day_local'].dt.strftime('%Y%m%d')

        if 'cancelled' not in df.columns:
            df['cancelled'] = 0
        df['cancelled'] = pd.to_numeric(df['cancelled'], errors='coerce').fillna(0).astype('int64')
        df['source_dataset'] = 'eurocontrol'

        cols_to_keep = list(cls.canonical_schema().keys()) + [
            'actual_dep', 'actual_arr',
            'service_day_local', 'service_day_utc'
        ]
        valid_cols = [col for col in cols_to_keep if col in df.columns]
        return df[valid_cols]
