"""T26: Unit tests for LabelGenerator."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.labeling import LabelGenerator


class TestLabelGenerator:

    def _test_log_dir(self) -> str:
        p = Path("logs/test_labeling")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def _make_df(self, delays, cancelled=None):
        n = len(delays)
        df = pd.DataFrame({
            'flight_key': [f'FL{i}' for i in range(n)],
            'scheduled_dep': pd.date_range('2025-06-15', periods=n, freq='h', tz='UTC'),
            'actual_dep': pd.date_range('2025-06-15', periods=n, freq='h', tz='UTC') + pd.to_timedelta(delays, unit='m'),
            'scheduled_arr': pd.date_range('2025-06-15 02:00', periods=n, freq='h', tz='UTC'),
            'actual_arr': pd.date_range('2025-06-15 02:00', periods=n, freq='h', tz='UTC'),
        })
        if cancelled is not None:
            df['cancelled'] = cancelled
        else:
            df['cancelled'] = 0
        return df

    def test_label_normal(self):
        """5-minute delay should be Normal."""
        df = self._make_df([5])
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        assert result['label'].iloc[0] == 'Normal'

    def test_label_late(self):
        """20-minute delay should be Late."""
        df = self._make_df([20])
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        assert result['label'].iloc[0] == 'Late'

    def test_label_cancelled(self):
        """Cancelled=1 should override to Cancelled."""
        df = self._make_df([0], cancelled=[1])
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        assert result['label'].iloc[0] == 'Cancelled'

    def test_unverified_euro_rows_are_flagged(self):
        df = self._make_df([20])
        df['label_source'] = ['unverified_euro']
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        assert result['label'].iloc[0] == 'Unverified'
        assert pd.isna(result['delay_minutes'].iloc[0])

    def test_label_nat_times(self):
        """NaT departure times should produce NaN delay, label Normal (not crash)."""
        df = pd.DataFrame({
            'flight_key': ['FL0'],
            'scheduled_dep': [pd.NaT],
            'actual_dep': [pd.NaT],
            'cancelled': [0],
        })
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        assert pd.isna(result['delay_minutes'].iloc[0])
        assert result['label'].iloc[0] == 'Normal'

    def test_label_threshold_boundary(self):
        """Exactly 15 minutes should be Normal (> not >=)."""
        df = self._make_df([15])
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        assert result['label'].iloc[0] == 'Normal'

    def test_standardize_drops_extra_columns(self):
        """standardize_dataset should drop actual_dep and actual_arr (leakage guard)."""
        df = self._make_df([5, 20])
        lg = LabelGenerator(delay_threshold_minutes=15, log_dir=self._test_log_dir())
        result = lg.generate_labels(df)
        standardized = lg.standardize_dataset(result)
        assert 'actual_dep' not in standardized.columns
        assert 'actual_arr' not in standardized.columns
        assert 'delay_minutes' in standardized.columns
