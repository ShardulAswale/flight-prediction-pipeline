"""
T22: Feature Selection & Analysis Script
Supporting analysis code for notebook 06. Can also be run standalone from the project root.

Sections:
1. Missingness heatmap
2. Correlation matrix
3. Mutual information
4. Permutation importance
5. Leakage risk table
"""
import os
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
from src.utils import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config(str(PROJECT_ROOT / 'configs' / 'config.yaml'))
paths = config['paths']
processed_dir = PROJECT_ROOT / paths['processed_data_dir']
output_dir = PROJECT_ROOT / 'outputs' / 'feature_analysis'
output_dir.mkdir(parents=True, exist_ok=True)

primary_dataset_path = processed_dir / paths['ml_dataset_file']
fallback_dataset_path = processed_dir / 'ml_dataset_labeled.parquet'
dataset_path = primary_dataset_path if primary_dataset_path.exists() else fallback_dataset_path

logger.info(f"Loading feature-selection dataset from {dataset_path} ...")
try:
    df = pd.read_parquet(dataset_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
except FileNotFoundError:
    logger.error(
        f"Dataset not found at {dataset_path}. Run notebook 04 or the weather/label/validate pipeline stages first."
    )
    sys.exit(1)

excluded_numeric = {'delay_minutes', 'cancelled'}
numeric_cols = [
    col for col in df.select_dtypes(include=[np.number]).columns.tolist()
    if col not in excluded_numeric
]
target_col = 'delay_minutes' if 'delay_minutes' in df.columns else None

logger.info("Generating missingness heatmap...")
null_pct = df[numeric_cols].isnull().mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12, max(6, len(numeric_cols) * 0.3)))
null_pct.plot(kind='barh', ax=ax, color='#e74c3c', alpha=0.8)
ax.set_xlabel('Null %')
ax.set_title('Feature Missingness')
ax.axvline(x=0.3, color='red', linestyle='--', label='30% threshold')
ax.legend()
plt.tight_layout()
fig.savefig(output_dir / 'missingness_heatmap.png', dpi=150)
plt.close()
logger.info(f"Saved to {output_dir / 'missingness_heatmap.png'}")

logger.info("Generating correlation matrix...")
corr = df[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
fig.savefig(output_dir / 'correlation_matrix.png', dpi=150)
plt.close()
logger.info(f"Saved to {output_dir / 'correlation_matrix.png'}")

high_corr = []
for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.85:
            high_corr.append({
                'feature_a': corr.columns[i],
                'feature_b': corr.columns[j],
                'correlation': round(corr.iloc[i, j], 4),
            })
if high_corr:
    logger.warning(f"Found {len(high_corr)} highly correlated pairs (|r| > 0.85):")
    for pair in high_corr:
        logger.warning(f"  {pair['feature_a']} <-> {pair['feature_b']}: {pair['correlation']}")

logger.info("Computing mutual information scores...")
try:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.preprocessing import LabelEncoder

    if 'label' in df.columns and numeric_cols:
        X_mi = df[numeric_cols].fillna(0)
        le = LabelEncoder()
        y_mi = le.fit_transform(df['label'])

        mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
        mi_df = pd.DataFrame({
            'feature': numeric_cols,
            'mutual_info': mi_scores,
        }).sort_values('mutual_info', ascending=False)

        fig, ax = plt.subplots(figsize=(12, max(6, len(numeric_cols) * 0.3)))
        mi_df.plot(x='feature', y='mutual_info', kind='barh', ax=ax, color='#3498db', legend=False)
        ax.set_xlabel('Mutual Information')
        ax.set_title('Mutual Information with Target Label')
        plt.tight_layout()
        fig.savefig(output_dir / 'mutual_information.png', dpi=150)
        plt.close()

        mi_df.to_csv(output_dir / 'mutual_information.csv', index=False)
        logger.info(f"Top 5 MI features: {mi_df.head()['feature'].tolist()}")
except Exception as exc:
    logger.warning(f"Mutual information computation failed: {exc}")

logger.info("Computing permutation importance...")
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    if 'label' in df.columns and numeric_cols:
        X_pi = df[numeric_cols].copy()
        imputer = SimpleImputer(strategy='median')
        X_pi = pd.DataFrame(imputer.fit_transform(X_pi), columns=numeric_cols)

        le = LabelEncoder()
        y_pi = le.fit_transform(df['label'])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_pi, y_pi, test_size=0.3, random_state=42, stratify=y_pi
        )

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)

        perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1)
        perm_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance_mean': perm.importances_mean,
            'importance_std': perm.importances_std,
        }).sort_values('importance_mean', ascending=False)

        fig, ax = plt.subplots(figsize=(12, max(6, len(numeric_cols) * 0.3)))
        perm_df.plot(
            x='feature',
            y='importance_mean',
            xerr='importance_std',
            kind='barh',
            ax=ax,
            color='#2ecc71',
            legend=False,
            capsize=3,
        )
        ax.set_xlabel('Permutation Importance')
        ax.set_title('Permutation Importance (Random Forest)')
        plt.tight_layout()
        fig.savefig(output_dir / 'permutation_importance.png', dpi=150)
        plt.close()

        perm_df.to_csv(output_dir / 'permutation_importance.csv', index=False)
        logger.info(f"Top 5 permutation-importance features: {perm_df.head()['feature'].tolist()}")
except Exception as exc:
    logger.warning(f"Permutation importance failed: {exc}")

logger.info("Building leakage risk table...")
leakage_risks = []
known_leaky = ['actual_dep', 'actual_arr', 'delay_minutes']
for col in df.columns:
    risk = 'NONE'
    reason = ''

    if col in known_leaky:
        risk = 'HIGH'
        reason = 'Direct target info / post-hoc measurement'
    elif col == 'cancelled':
        risk = 'MODERATE'
        reason = 'May be known in advance but strongly correlates with target'
    elif target_col and col in numeric_cols and target_col in df.columns:
        corr_val = df[col].corr(df[target_col])
        if pd.notna(corr_val):
            if abs(corr_val) > 0.9:
                risk = 'HIGH'
                reason = f'Very high correlation with target (r={corr_val:.3f})'
            elif abs(corr_val) > 0.7:
                risk = 'MODERATE'
                reason = f'High correlation with target (r={corr_val:.3f})'

    if risk != 'NONE':
        leakage_risks.append({'column': col, 'risk_level': risk, 'reason': reason})

leak_df = pd.DataFrame(leakage_risks)
if not leak_df.empty:
    leak_df.to_csv(output_dir / 'leakage_risk_table.csv', index=False)
    logger.info('Leakage Risk Table:')
    for _, row in leak_df.iterrows():
        logger.info(f"  [{row['risk_level']}] {row['column']}: {row['reason']}")

logger.info(f"All outputs saved to {output_dir}/")
logger.info('Feature analysis complete!')
