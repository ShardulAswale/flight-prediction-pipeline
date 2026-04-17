"""T27: Unit tests for QualityGateRunner."""
import pytest
import pandas as pd
import numpy as np
import json
from pathlib import Path
from src.quality_gates import QualityGateRunner


class TestQualityGateRunner:

    def _test_log_dir(self) -> str:
        p = Path("logs/test_quality_gates")
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    def _make_clean_df(self, n=20):
        np.random.seed(42)
        labels = (['Normal'] * 10 + ['Late'] * 7 + ['Cancelled'] * 3)
        labels = (labels * ((n // len(labels)) + 1))[:n]
        return pd.DataFrame({
            'flight_key': [f'FL{i}' for i in range(n)],
            'feature_a': np.random.randn(n),
            'feature_b': np.random.randn(n),
            'label': labels,
        })

    def test_missingness_passes(self):
        df = self._make_clean_df()
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.check_missingness(df, max_null_pct=0.3)
        assert result['passed'] is True

    def test_missingness_fails(self):
        df = self._make_clean_df()
        df.loc[:14, 'feature_a'] = np.nan  # 75% null
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.check_missingness(df, max_null_pct=0.3)
        assert result['passed'] is False
        assert 'feature_a' in result['violations']

    def test_duplicates_passes(self):
        df = self._make_clean_df()
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.check_duplicates(df)
        assert result['passed'] is True

    def test_duplicates_fails(self):
        df = self._make_clean_df()
        df.loc[19, 'flight_key'] = df.loc[0, 'flight_key']
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.check_duplicates(df)
        assert result['passed'] is False
        assert result['duplicate_count'] == 1

    def test_label_balance_passes(self):
        df = self._make_clean_df()
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.check_label_balance(df, min_pct=0.05)
        assert result['passed'] is True

    def test_label_balance_fails(self):
        df = self._make_clean_df(n=100)
        df['label'] = ['Normal'] * 99 + ['Late']  # Late is 1%
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.check_label_balance(df, min_pct=0.05)
        assert result['passed'] is False

    def test_feature_quality_score(self):
        df = self._make_clean_df()
        qg = QualityGateRunner(log_dir=self._test_log_dir())
        result = qg.compute_feature_quality_score(df, feature_cols=['feature_a', 'feature_b'])
        assert 'feature_quality_score' in result.columns
        assert 'training_eligible' in result.columns
        assert result['feature_quality_score'].between(0, 1).all()

    def test_run_all_returns_report(self):
        df = self._make_clean_df()
        log_dir = Path(self._test_log_dir())
        qg = QualityGateRunner(log_dir=str(log_dir))
        report = qg.run_all(df)
        assert 'missingness_passed' in report
        assert 'duplicates_passed' in report
        assert 'label_balance_passed' in report
        assert 'overall_passed' in report
        # Check JSON file was created
        report_file = log_dir / 'quality_gates_report.json'
        assert report_file.exists()
        with open(report_file) as f:
            loaded = json.load(f)
        assert loaded['overall_passed'] == report['overall_passed']
