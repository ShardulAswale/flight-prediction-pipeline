"""
Quality Gate Runner — Pre-training data quality checks.
T15: Ensures ML dataset meets minimum quality thresholds before model training.
"""
import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class QualityGateRunner:
    """Runs quality gate checks on the ML dataset before training."""
    
    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def check_missingness(self, df: pd.DataFrame, max_null_pct: float = 0.3) -> dict:
        """Check that no feature column exceeds the null threshold."""
        total = len(df)
        if total == 0:
            return {'passed': False, 'reason': 'Empty dataset', 'details': {}}
        
        violations = {}
        for col in df.columns:
            null_pct = df[col].isna().sum() / total
            if null_pct > max_null_pct:
                violations[col] = round(null_pct, 4)
        
        passed = len(violations) == 0
        return {
            'passed': passed,
            'max_null_pct_threshold': max_null_pct,
            'violations': violations,
            'total_columns_checked': len(df.columns)
        }
    
    def check_duplicates(self, df: pd.DataFrame, key_col: str = 'flight_key') -> dict:
        """Check for duplicate flight keys."""
        if key_col not in df.columns:
            return {'passed': True, 'reason': f'Column {key_col} not found, skipping'}
        
        dup_count = df.duplicated(subset=[key_col]).sum()
        passed = bool(dup_count == 0)
        
        return {
            'passed': passed,
            'duplicate_count': int(dup_count),
            'total_rows': len(df),
            'key_column': key_col
        }
    
    def check_label_balance(self, df: pd.DataFrame, label_col: str = 'label', 
                           min_pct: float = 0.05) -> dict:
        """Check that each label class has at least min_pct representation."""
        if 'training_eligible' in df.columns:
            df = df[df['training_eligible'] == True].copy()

        if label_col not in df.columns:
            return {'passed': False, 'reason': f'Column {label_col} not found'}
        if df.empty:
            return {'passed': False, 'reason': 'No training-eligible labeled rows'}
        
        dist = df[label_col].value_counts(normalize=True)
        violations = {}
        for label, pct in dist.items():
            if pct < min_pct:
                violations[str(label)] = round(pct, 4)
        
        passed = len(violations) == 0
        
        distribution = {str(k): round(v, 4) for k, v in dist.items()}
        
        return {
            'passed': passed,
            'min_pct_threshold': min_pct,
            'distribution': distribution,
            'violations': violations
        }
    
    def compute_feature_quality_score(self, df: pd.DataFrame, 
                                       feature_cols: list = None) -> pd.DataFrame:
        """Compute per-row feature quality score and add training_eligible flag.
        
        Args:
            df: Dataset DataFrame.
            feature_cols: List of feature column names. If None, uses numeric columns.
            
        Returns:
            DataFrame with 'feature_quality_score' and 'training_eligible' columns added.
        """
        df = df.copy()
        
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target columns
            feature_cols = [c for c in feature_cols if c not in ['delay_minutes', 'label']]
        
        # Only score columns that exist and have meaningful dataset-wide coverage.
        existing_features = [c for c in feature_cols if c in df.columns]
        coverage_threshold = 0.10
        sparse_optional = [c for c in existing_features if df[c].notna().mean() < coverage_threshold]
        existing_features = [c for c in existing_features if c not in sparse_optional]
        if sparse_optional:
            logger.info(
                "Ignoring %d sparse optional features (<%.0f%% coverage) when computing feature_quality_score: %s",
                len(sparse_optional),
                coverage_threshold * 100,
                sparse_optional,
            )
        
        if not existing_features:
            df['feature_quality_score'] = 0.0
            df['training_eligible'] = False
            return df
        
        # Per-row: 1 - (null_count / total_feature_cols)
        null_counts = df[existing_features].isna().sum(axis=1)
        df['feature_quality_score'] = 1.0 - (null_counts / len(existing_features))
        df['training_eligible'] = df['feature_quality_score'] > 0.7

        if 'label_source' in df.columns:
            verified_mask = ~df['label_source'].astype('string').isin(['unverified_euro', 'no_verified_label'])
            df['training_eligible'] = df['training_eligible'] & verified_mask
        if 'label' in df.columns:
            df['training_eligible'] = df['training_eligible'] & df['label'].isin(['Normal', 'Late', 'Cancelled'])
        
        return df
    
    def run_all(self, df: pd.DataFrame, feature_cols: list = None) -> dict:
        """Run all quality gates and return a comprehensive report.
        
        Returns:
            dict with gate results and overall pass/fail status.
        """
        logger.info(f"Running quality gates on dataset with {len(df)} rows...")
        
        # Run individual checks
        missingness = self.check_missingness(df)
        duplicates = self.check_duplicates(df)
        label_balance = self.check_label_balance(df)
        
        # Add quality score columns
        df_scored = self.compute_feature_quality_score(df, feature_cols)
        
        # Overall pass
        overall_passed = all([
            missingness['passed'],
            duplicates['passed'],
            label_balance['passed']
        ])
        
        # Build report
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_rows': len(df),
            'missingness_passed': missingness['passed'],
            'duplicates_passed': duplicates['passed'],
            'label_balance_passed': label_balance['passed'],
            'overall_passed': overall_passed,
            'details': {
                'missingness': missingness,
                'duplicates': duplicates,
                'label_balance': label_balance,
            },
            'feature_quality': {
                'mean_score': round(float(df_scored['feature_quality_score'].mean()), 4),
                'median_score': round(float(df_scored['feature_quality_score'].median()), 4),
                'training_eligible_count': int(df_scored['training_eligible'].sum()),
                'training_eligible_pct': round(
                    float(df_scored['training_eligible'].mean() * 100), 2
                )
            }
        }
        
        # Save report
        try:
            report_path = self.log_dir / 'quality_gates_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4, default=str)
            logger.info(f"Quality gates report saved to {report_path}")
        except Exception as e:
            logger.warning(f"Failed to save quality gates report: {e}")
        
        status = "PASSED" if overall_passed else "FAILED"
        logger.info(f"Quality gates {status}: missingness={missingness['passed']}, "
                     f"duplicates={duplicates['passed']}, label_balance={label_balance['passed']}")
        
        return report
