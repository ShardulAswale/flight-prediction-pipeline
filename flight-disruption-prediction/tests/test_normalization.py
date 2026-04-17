"""Unit tests for ScheduleNormalizer."""
import pandas as pd

from src.normalization import ScheduleNormalizer


class TestScheduleNormalizerBTS:

    def _make_raw_bts(self, n=4):
        return pd.DataFrame({
            'FlightDate': ['2025-06-15'] * n,
            'Reporting_Airline': ['DL', 'WN', 'AA', 'OH'][:n],
            'IATA_CODE_Reporting_Airline': ['DL', 'WN', 'AA', 'OH'][:n],
            'Flight_Number_Reporting_Airline': [str(100 + i) for i in range(n)],
            'Origin': ['ATL', 'LAX', 'PHX', 'DCA'][:n],
            'OriginState': ['GA', 'CA', 'AZ', 'VA'][:n],
            'Dest': ['JFK', 'SEA', 'LAS', 'ALB'][:n],
            'DestState': ['NY', 'WA', 'NV', 'NY'][:n],
            'Cancelled': [0, 0, 0, 1][:n],
            'CRSDepTime': [800, 815, 2330, 1010][:n],
            'DepTime': [805, 820, 2335, None][:n],
            'CRSArrTime': [1130, 1045, 35, 1152][:n],
            'ArrTime': [1135, 1050, 45, None][:n],
            'CRSElapsedTime': [150, 150, 125, 102][:n],
            'ActualElapsedTime': [155, 150, 130, None][:n],
            'DepDelay': [5, 5, 5, None][:n],
            'DepDelayMinutes': [5, 5, 5, None][:n],
        })

    def test_normalize_bts_keeps_core_columns(self):
        df = self._make_raw_bts()
        result = ScheduleNormalizer.normalize_bts(df)
        expected = set(ScheduleNormalizer.canonical_schema().keys())
        assert expected.issubset(set(result.columns))

    def test_normalize_bts_adds_enriched_columns(self):
        df = self._make_raw_bts()
        result = ScheduleNormalizer.normalize_bts(df)
        for column in [
            'scheduled_dep_utc', 'scheduled_arr_utc', 'actual_dep_utc', 'actual_arr_utc',
            'service_day_local', 'service_day_utc', 'DepDelay', 'Cancelled'
        ]:
            assert column in result.columns

    def test_normalize_bts_iata_to_icao(self):
        df = self._make_raw_bts()
        result = ScheduleNormalizer.normalize_bts(df)
        assert result['origin'].iloc[0] == 'KATL'
        assert result['destination'].iloc[0] == 'KJFK'

    def test_normalize_bts_departure_is_utc(self):
        df = self._make_raw_bts()
        result = ScheduleNormalizer.normalize_bts(df)
        assert str(result['scheduled_dep'].dt.tz) == 'UTC'
        assert str(result['actual_dep'].dt.tz) == 'UTC'

    def test_normalize_bts_overnight_arrival_rolls_forward(self):
        df = self._make_raw_bts(n=3)
        result = ScheduleNormalizer.normalize_bts(df)
        overnight = result.iloc[2]
        assert overnight['scheduled_arr'] > overnight['scheduled_dep']


class TestScheduleNormalizerEuro:

    def _make_raw_euro(self, n=4):
        return pd.DataFrame({
            'flt_id': [f'BAW{100+i}' for i in range(n)],
            'adep': ['EGLL', 'LFPG', 'EDDF', 'EHAM'][:n],
            'ades': ['LFPG', 'EDDF', 'EHAM', 'LEMD'][:n],
            'dof': ['2025-06-15'] * n,
            'first_seen': pd.date_range('2025-06-15 06:00', periods=n, freq='h', tz='UTC'),
            'last_seen': pd.date_range('2025-06-15 08:00', periods=n, freq='h', tz='UTC'),
        })

    def test_normalize_eurocontrol_keeps_core_columns(self):
        df = self._make_raw_euro()
        result = ScheduleNormalizer.normalize_eurocontrol(df)
        expected = set(ScheduleNormalizer.canonical_schema().keys())
        assert expected.issubset(set(result.columns))

    def test_normalize_eurocontrol_actuals_are_missing_when_not_provided(self):
        df = self._make_raw_euro()
        result = ScheduleNormalizer.normalize_eurocontrol(df)
        assert result['actual_dep'].isna().all()
        assert result['actual_arr'].isna().all()


class TestEmptyDataFrame:

    def test_normalize_bts_empty(self):
        df = pd.DataFrame(columns=['FlightDate', 'Reporting_Airline', 'Flight_Number_Reporting_Airline'])
        result = ScheduleNormalizer.normalize_bts(df)
        assert len(result) == 0

    def test_normalize_eurocontrol_empty(self):
        df = pd.DataFrame(columns=['flt_id', 'adep', 'ades', 'first_seen', 'last_seen'])
        result = ScheduleNormalizer.normalize_eurocontrol(df)
        assert len(result) == 0
