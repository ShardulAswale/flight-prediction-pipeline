"""
Drift Detection Module
T30: PSI and KS test for feature drift across month-over-month data.
"""
import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects feature drift across monthly data partitions."""
    
    def __init__(self, log_dir: str = 'logs', output_dir: str = 'outputs'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index between two distributions.
        
        PSI < 0.1: No significant shift
        PSI 0.1-0.2: Moderate shift
        PSI > 0.2: Significant shift
        """
        eps = 1e-6
        
        # Remove NaN
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        
        if len(ref) == 0 or len(cur) == 0:
            return 0.0
        
        # Create bins from reference distribution
        breakpoints = np.linspace(np.min(ref), np.max(ref), bins + 1)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf
        
        ref_counts = np.histogram(ref, bins=breakpoints)[0]
        cur_counts = np.histogram(cur, bins=breakpoints)[0]
        
        ref_pct = ref_counts / len(ref) + eps
        cur_pct = cur_counts / len(cur) + eps
        
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
    
    def compute_ks(self, reference: np.ndarray, current: np.ndarray) -> dict:
        """Compute KS test statistic and p-value."""
        from scipy.stats import ks_2samp
        
        ref = reference[~np.isnan(reference)]
        cur = current[~np.isnan(current)]
        
        if len(ref) < 2 or len(cur) < 2:
            return {'ks_stat': 0.0, 'ks_pvalue': 1.0}
        
        stat, pvalue = ks_2samp(ref, cur)
        return {'ks_stat': float(stat), 'ks_pvalue': float(pvalue)}
    
    def monthly_drift_report(self, df: pd.DataFrame, date_col: str = 'scheduled_dep',
                              feature_cols: list = None) -> Optional[pd.DataFrame]:
        """Compute PSI and KS for each feature across monthly partitions.
        
        First month = reference distribution. Subsequent months compared against reference.
        """
        if date_col not in df.columns:
            logger.error(f"Date column '{date_col}' not found in dataset")
            return None
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Partition by month
        df['_month'] = df[date_col].dt.to_period('M')
        months = sorted(df['_month'].unique())
        
        if len(months) < 2:
            logger.warning("Need at least 2 months for drift detection")
            return None
        
        reference_month = months[0]
        ref_data = df[df['_month'] == reference_month]
        
        logger.info(f"Drift detection: reference={reference_month}, comparing {len(months)-1} months")
        
        records = []
        for month in months[1:]:
            cur_data = df[df['_month'] == month]
            
            for feat in feature_cols:
                if feat not in df.columns:
                    continue
                
                ref_vals = ref_data[feat].values.astype(float)
                cur_vals = cur_data[feat].values.astype(float)
                
                psi = self.compute_psi(ref_vals, cur_vals)
                ks = self.compute_ks(ref_vals, cur_vals)
                
                drift_alert = psi > 0.2 or ks['ks_pvalue'] < 0.05
                
                records.append({
                    'feature': feat,
                    'month': str(month),
                    'psi': round(psi, 6),
                    'ks_stat': round(ks['ks_stat'], 6),
                    'ks_pvalue': round(ks['ks_pvalue'], 6),
                    'drift_alert': drift_alert,
                    'ref_month': str(reference_month),
                    'ref_count': len(ref_vals),
                    'cur_count': len(cur_vals),
                })
        
        drift_df = pd.DataFrame(records)
        
        # Save reports
        try:
            report = drift_df.to_dict(orient='records')
            with open(self.log_dir / 'drift_report.json', 'w') as f:
                json.dump(report, f, indent=4, default=str)
            
            # Summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'reference_month': str(reference_month),
                'total_features': len(feature_cols),
                'months_analyzed': len(months) - 1,
                'total_alerts': int(drift_df['drift_alert'].sum()) if not drift_df.empty else 0,
                'features_with_alerts': drift_df[drift_df['drift_alert']]['feature'].nunique() if not drift_df.empty else 0,
            }
            with open(self.log_dir / 'drift_summary.json', 'w') as f:
                json.dump(summary, f, indent=4)
            
            logger.info(f"Drift reports saved. {summary['total_alerts']} alerts across {summary['features_with_alerts']} features.")
        except Exception as e:
            logger.warning(f"Failed to save drift reports: {e}")
        
        # Generate heatmap
        self._generate_heatmap(drift_df)
        
        return drift_df
    
    def stability_ranking(self, drift_df: pd.DataFrame) -> pd.DataFrame:
        """Rank features by average PSI ascending (most stable first)."""
        if drift_df.empty:
            return pd.DataFrame()
        
        ranking = drift_df.groupby('feature')['psi'].mean().sort_values()
        return ranking.reset_index().rename(columns={'psi': 'avg_psi'})
    
    def flag_high_importance_unstable(self, drift_df: pd.DataFrame, 
                                       importance_df: pd.DataFrame) -> pd.DataFrame:
        """Cross-reference drift with feature importance.
        
        Args:
            drift_df: Output of monthly_drift_report
            importance_df: DataFrame with 'feature' and 'importance' columns
        """
        if drift_df.empty:
            return pd.DataFrame()
        
        stability = self.stability_ranking(drift_df)
        
        # Get top-10 importance features
        if isinstance(importance_df, dict):
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} for k, v in importance_df.items()
            ])
        
        if 'feature' not in importance_df.columns:
            return pd.DataFrame()
        
        top_10 = importance_df.nlargest(10, 'importance')['feature'].tolist()
        
        # Find unstable features in top-10
        unstable = stability[stability['feature'].isin(top_10)]
        drifting = drift_df[drift_df['drift_alert'] == True]['feature'].unique()
        
        flagged = unstable[unstable['feature'].isin(drifting)]
        return flagged
    
    def _generate_heatmap(self, drift_df: pd.DataFrame):
        """Generate PSI heatmap (features × month)."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if drift_df.empty:
                return
            
            pivot = drift_df.pivot(index='feature', columns='month', values='psi')
            
            fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.4)))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
                       linewidths=0.5, ax=ax, vmin=0, vmax=0.5,
                       cbar_kws={'label': 'PSI'})
            ax.set_title('Feature Drift Heatmap (PSI by Month)')
            ax.set_ylabel('Feature')
            ax.set_xlabel('Month')
            
            # Add threshold line annotation
            ax.axhline(y=0, color='red', linestyle='--', alpha=0)  # placeholder
            
            plt.tight_layout()
            fig.savefig(self.output_dir / 'drift_heatmap.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Drift heatmap saved to {self.output_dir / 'drift_heatmap.png'}")
        except Exception as e:
            logger.warning(f"Failed to generate drift heatmap: {e}")
