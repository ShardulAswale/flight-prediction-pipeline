"""T24: Unit tests for DataValidator."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_validator import DataValidator


class TestDataValidator:

    def _test_log_dir(self) -> str:
        p = Path("logs/test_data_validator")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    
    def test_validate_passes_valid_data(self):
        """Valid data should pass validation."""
        schema = {
            'col_a': {'dtype': 'float64', 'max_null_pct': 0.1, 'is_utc': False},
            'col_b': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
        }
        df = pd.DataFrame({'col_a': [1.0, 2.0, 3.0], 'col_b': ['x', 'y', 'z']})
        
        validator = DataValidator(mode='fail_fast', log_dir=self._test_log_dir())
        result = validator.validate(df, schema, 'test_valid')
        assert result is True

    def test_validate_fails_missing_column(self):
        """Missing required column should fail in fail_fast mode."""
        schema = {
            'col_a': {'dtype': 'float64', 'max_null_pct': 0.0, 'is_utc': False},
            'col_missing': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
        }
        df = pd.DataFrame({'col_a': [1.0, 2.0]})
        
        validator = DataValidator(mode='fail_fast', log_dir=self._test_log_dir())
        with pytest.raises(ValueError, match="Missing required columns"):
            validator.validate(df, schema, 'test_missing')

    def test_validate_warns_missing_column(self):
        """Missing column in warn_only mode should return False but not raise."""
        schema = {
            'col_missing': {'dtype': 'object', 'max_null_pct': 0.0, 'is_utc': False},
        }
        df = pd.DataFrame({'col_a': [1.0]})
        
        validator = DataValidator(mode='warn_only', log_dir=self._test_log_dir())
        result = validator.validate(df, schema, 'test_warn')
        assert result is False

    def test_validate_fails_null_threshold(self):
        """Exceeding null threshold should fail."""
        schema = {
            'col_a': {'dtype': 'float64', 'max_null_pct': 0.1, 'is_utc': False},
        }
        df = pd.DataFrame({'col_a': [1.0, np.nan, np.nan, np.nan]})  # 75% null
        
        validator = DataValidator(mode='fail_fast', log_dir=self._test_log_dir())
        with pytest.raises(ValueError, match="null threshold"):
            validator.validate(df, schema, 'test_null')

    def test_validate_fails_non_utc_datetime(self):
        """Non-UTC datetime should fail when is_utc=True."""
        schema = {
            'ts': {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0, 'is_utc': True},
        }
        # Timezone-naive datetime
        df = pd.DataFrame({'ts': pd.to_datetime(['2025-01-01', '2025-01-02'])})
        
        validator = DataValidator(mode='fail_fast', log_dir=self._test_log_dir())
        with pytest.raises(ValueError, match="timezone"):
            validator.validate(df, schema, 'test_tz')

    def test_validate_passes_utc_datetime(self):
        """UTC datetime should pass when is_utc=True."""
        schema = {
            'ts': {'dtype': 'datetime64[ns, UTC]', 'max_null_pct': 0.0, 'is_utc': True},
        }
        df = pd.DataFrame({
            'ts': pd.to_datetime(['2025-01-01', '2025-01-02']).tz_localize('UTC')
        })
        
        validator = DataValidator(mode='fail_fast', log_dir=self._test_log_dir())
        result = validator.validate(df, schema, 'test_utc_pass')
        assert result is True
