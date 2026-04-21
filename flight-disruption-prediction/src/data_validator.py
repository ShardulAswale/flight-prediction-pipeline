import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype

logger = logging.getLogger(__name__)


def _dtype_matches(expected_dtype: str | None, actual_series: pd.Series) -> bool:
    """Return True when the actual pandas dtype is compatible with the schema.

    Pandas may report text columns as ``object``, ``str``, ``string``, or
    ``string[python]`` depending on construction and parquet round-tripping.
    Those are semantically equivalent for this pipeline.
    """
    if not expected_dtype:
        return True

    expected = str(expected_dtype).lower()
    actual = str(actual_series.dtype).lower()

    if expected in actual:
        return True
    if expected in {"object", "str", "string"} and (
        is_object_dtype(actual_series) or is_string_dtype(actual_series)
    ):
        return True
    if expected.startswith("datetime") and actual.startswith("datetime"):
        return True
    return expected == actual

class DataValidator:
    """
    Validates DataFrames against a strict canonical schema.
    Supports fail-fast and warn-only modes.
    Generates JSON validation reports with counts and examples of bad rows.
    """
    
    def __init__(self, mode: str = 'fail_fast', log_dir: str = 'logs'):
        if mode not in ['fail_fast', 'warn_only']:
            raise ValueError("mode must be 'fail_fast' or 'warn_only'")
        self.mode = mode
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _save_report(self, report: dict, report_name: str) -> None:
        timestamp = datetime.now().strftime("%Y%md_%H%M%S")
        filepath = self.log_dir / f"{report_name}_{timestamp}.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        logger.info(f"Validation report saved to {filepath}")
        
    def validate(self, df: pd.DataFrame, schema: dict, dataset_name: str) -> bool:
        """
        Validates the DataFrame against a defined schema.
        
        schema format:
        {
            'col_name': {
                'dtype': 'float64',  # expected pandas dtype string
                'max_null_pct': 0.05, # max allowed null percentage (0-1)
                'is_utc': False       # if True, enforces timezone-aware UTC datetime
            }
        }
        """
        logger.info(f"Validating dataset '{dataset_name}' containing {len(df)} rows.")
        
        errors: list[dict] = []
        report = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'mode': self.mode,
            'status': 'PASSED'
        }
        
        is_valid = True
        
        # 1. Missing columns
        missing_cols = [c for c in schema.keys() if c not in df.columns]
        if missing_cols:
            is_valid = False
            msg = f"Missing required columns: {missing_cols}"
            errors.append({'type': 'missing_columns', 'message': msg})
        
        for col, rules in schema.items():
            if col in missing_cols:
                continue
                
            # 2. Check DType
            expected_dtype = rules.get('dtype')
            actual_dtype = str(df[col].dtype)
            if not _dtype_matches(expected_dtype, df[col]):
                is_valid = False
                msg = f"Column '{col}' expected dtype '{expected_dtype}', got '{actual_dtype}'"
                errors.append({'type': 'dtype_mismatch', 'column': col, 'message': msg})
            
            # 3. Check Null Threshold
            max_null_pct = rules.get('max_null_pct', 0.0)
            null_count = int(df[col].isna().sum())
            null_pct = null_count / len(df) if len(df) > 0 else 0
            
            if null_pct > max_null_pct:
                is_valid = False
                bad_samples = df[df[col].isna()].head(5).to_dict(orient='records')
                msg = f"Column '{col}' exceeds null threshold: {null_pct:.1%}/{max_null_pct:.1%}"
                errors.append({
                    'type': 'null_threshold_exceeded',
                    'column': col,
                    'message': msg,
                    'null_count': null_count,
                    'examples': bad_samples
                })
                
            # 4. Check UTC Timestamps
            if rules.get('is_utc', False):
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    is_valid = False
                    msg = f"Column '{col}' must be datetime for UTC enforcement."
                    errors.append({'type': 'invalid_datetime', 'column': col, 'message': msg})
                else:
                    if getattr(df[col].dt, 'tz', None) is None:
                        # Sometimes airports store time naive, check if it's explicitly UTC or naive
                        is_valid = False
                        msg = f"Column '{col}' is missing timezone (must be UTC)."
                        errors.append({'type': 'missing_timezone', 'column': col, 'message': msg})
                    elif str(df[col].dt.tz) != 'UTC':
                        is_valid = False
                        msg = f"Column '{col}' timezone is '{str(df[col].dt.tz)}', expected 'UTC'."
                        errors.append({'type': 'invalid_timezone', 'column': col, 'message': msg})

        report['errors'] = errors

        if not is_valid:
            report['status'] = 'FAILED'
            self._save_report(report, f"validation_error_{dataset_name}")
            
            error_summary = "\\n".join([e['message'] for e in errors])
            log_msg = f"Validation failed for '{dataset_name}'. Errors:\\n{error_summary}"
            
            if self.mode == 'fail_fast':
                logger.error(log_msg)
                raise ValueError(log_msg)
            else:
                logger.warning(log_msg)
        else:
            self._save_report(report, f"validation_success_{dataset_name}")
            logger.info(f"Dataset '{dataset_name}' passed all validation checks.")
            
        return is_valid
