"""
ML Model Training Pipeline
T17: Data preparation with stratified split, imputation, scaling
T18: Logistic Regression + Random Forest baselines
T19: XGBoost + LightGBM with Optuna HPO
T20: Model comparison report and visuals
T21: SHAP explanations
"""
import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional
import warnings

logger = logging.getLogger(__name__)


class ModelTrainer:
    """End-to-end ML training pipeline for flight disruption prediction."""
    
    def __init__(self, config: dict, use_gpu: Optional[bool] = None):
        self.config = config
        self.seed = config.get('seed', 42)
        training_config = config.get('training', {}) if isinstance(config.get('training', {}), dict) else {}
        self.use_gpu = training_config.get('use_gpu', False) if use_gpu is None else bool(use_gpu)
        self.models_dir = Path('models')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir = Path('outputs')
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path('logs')
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        self.correlation_dropped_features = []
        self.train_sample_weight = None
        self.test_sample_weight = None
        self.target_label_column = 'label'
        self.subtype_weight_config = {}

    def _gpu_status_message(self) -> str:
        return 'enabled' if self.use_gpu else 'disabled'

    @staticmethod
    def _is_gpu_related_error(exc: Exception) -> bool:
        message = str(exc).lower()
        gpu_tokens = [
            'cuda', 'gpu', 'device', 'driver', 'tree_method', 'gpu_hist',
            'boost.compute', 'opencl', 'no visible gpu', 'not compiled with gpu',
        ]
        return any(token in message for token in gpu_tokens)

    def _xgboost_runtime_params(self) -> dict:
        if not self.use_gpu:
            return {'tree_method': 'hist'}
        return {'tree_method': 'hist', 'device': 'cuda'}

    def _lightgbm_runtime_params(self) -> dict:
        if not self.use_gpu:
            return {'device_type': 'cpu'}
        return {'device_type': 'gpu'}

    def _catboost_runtime_params(self) -> dict:
        if not self.use_gpu:
            return {'task_type': 'CPU'}
        return {'task_type': 'GPU'}

    def _fit_with_gpu_fallback(self, builder, fit_args: tuple, fit_kwargs: Optional[dict] = None, model_name: str = 'model'):
        fit_kwargs = fit_kwargs or {}
        model = builder(self.use_gpu)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(*fit_args, **fit_kwargs)
            return model
        except Exception as exc:
            if self.use_gpu and self._is_gpu_related_error(exc):
                logger.warning('%s GPU training failed (%s). Falling back to CPU.', model_name, exc)
                model = builder(False)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    model.fit(*fit_args, **fit_kwargs)
                return model
            raise

    @staticmethod
    def _safe_cv_splits(y_train: pd.Series, preferred_splits: int = 5) -> int:
        """Choose a CV fold count that respects the least-populated class."""
        counts = pd.Series(y_train).value_counts()
        if counts.empty:
            return 2
        return max(2, min(preferred_splits, int(counts.min())))

    @staticmethod
    def _high_correlation_drop_columns(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return []

        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop: set[str] = set()

        for col in upper.columns:
            for row, corr_value in upper[col].dropna().items():
                if corr_value <= threshold or row in to_drop or col in to_drop:
                    continue

                row_null = float(numeric[row].isna().mean())
                col_null = float(numeric[col].isna().mean())
                row_var = float(numeric[row].var(skipna=True) or 0.0)
                col_var = float(numeric[col].var(skipna=True) or 0.0)

                if row_null != col_null:
                    drop_candidate = row if row_null > col_null else col
                elif row_var != col_var:
                    drop_candidate = row if row_var < col_var else col
                else:
                    drop_candidate = max(row, col)
                to_drop.add(drop_candidate)

        return sorted(to_drop)

    @staticmethod
    def _balanced_sample_weight(y: pd.Series) -> np.ndarray:
        from sklearn.utils.class_weight import compute_sample_weight
        return compute_sample_weight(class_weight='balanced', y=np.asarray(y))

    def _get_training_config(self) -> dict:
        return self.config.get('training', {}) if isinstance(self.config.get('training', {}), dict) else {}

    def _target_mode(self) -> str:
        return str(self._get_training_config().get('target_mode', 'binary_disrupted')).strip().lower()

    def _subtype_weight_config(self) -> dict:
        subtype_cfg = self._get_training_config().get('subtype_sample_weights', {})
        if not isinstance(subtype_cfg, dict):
            subtype_cfg = {}
        return {
            'enabled': bool(subtype_cfg.get('enabled', False)),
            'normal': float(subtype_cfg.get('normal', 1.0)),
            'late': float(subtype_cfg.get('late', 1.0)),
            'cancelled': float(subtype_cfg.get('cancelled', 1.0)),
        }

    def _ensure_training_label_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if 'label_original' not in out.columns:
            out['label_original'] = out.get('label', 'Normal')
        if 'label_binary' not in out.columns:
            out['label_binary'] = np.where(
                out['label_original'].isin(['Late', 'Cancelled']),
                'Disrupted',
                out['label_original'],
            )
        if 'disruption_subtype' not in out.columns:
            out['disruption_subtype'] = np.where(
                out['label_original'] == 'Cancelled',
                'Cancelled',
                np.where(
                    out['label_original'] == 'Late',
                    'Late',
                    out['label_original'],
                ),
            )
        return out

    def _prepare_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self._ensure_training_label_columns(df)
        target_mode = self._target_mode()

        if target_mode == 'binary_disrupted':
            self.target_label_column = 'label_binary'
            return out

        self.target_label_column = 'label'
        cancelled_count = (out['label'] == 'Cancelled').sum()
        cancelled_merge_threshold = int(self._get_training_config().get('cancelled_merge_threshold', 200))
        if cancelled_count > 0 and cancelled_count < cancelled_merge_threshold:
            logger.info(
                "Merging %d Cancelled rows into Late (threshold=%d)",
                cancelled_count,
                cancelled_merge_threshold,
            )
            out.loc[out['label'] == 'Cancelled', 'label'] = 'Late'
        return out

    def _subtype_multiplier(self, subtype: pd.Series) -> pd.Series:
        subtype_cfg = self._subtype_weight_config()
        self.subtype_weight_config = subtype_cfg
        if not subtype_cfg.get('enabled', False):
            return pd.Series(1.0, index=subtype.index, dtype='float64')

        multiplier_map = {
            'Normal': subtype_cfg['normal'],
            'Late': subtype_cfg['late'],
            'Cancelled': subtype_cfg['cancelled'],
            'Disrupted': subtype_cfg['late'],
            'Unverified': 1.0,
        }
        return subtype.map(multiplier_map).fillna(1.0).astype('float64')

    def _compose_training_sample_weight(self, y: pd.Series, subtype_multiplier: Optional[pd.Series] = None) -> pd.Series:
        balanced = pd.Series(self._balanced_sample_weight(y), index=y.index, dtype='float64')
        if subtype_multiplier is None:
            return balanced
        return balanced.mul(subtype_multiplier.reindex(y.index).fillna(1.0).astype('float64'))

    @staticmethod
    def _prediction_to_label_vector(pred) -> np.ndarray:
        """Normalize model predictions/probabilities into 1-D class labels.

        Some GPU/runtime combinations return probability matrices from
        ``predict`` for soft-probability objectives. Scikit-learn metrics need
        class labels, so make that conversion in one place.
        """
        arr = np.asarray(pred)
        if arr.ndim == 2:
            if arr.shape[1] == 1:
                arr = arr.ravel()
            else:
                return np.argmax(arr, axis=1)
        elif arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim > 1:
                return np.argmax(arr, axis=-1).ravel()

        if np.issubdtype(arr.dtype, np.floating):
            finite = arr[np.isfinite(arr)]
            if finite.size and finite.min() >= 0.0 and finite.max() <= 1.0:
                return (arr >= 0.5).astype(int)
        return arr
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare features/targets and persist training sample weights on the trainer instance."""
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        training_config = self._get_training_config()

        logger.info("Preparing data from %d rows...", len(df))

        if 'training_eligible' in df.columns:
            df_eligible = df[df['training_eligible'] == True].copy()
            logger.info("Filtered to %d training-eligible rows", len(df_eligible))
        else:
            df_eligible = df.copy()
            logger.info("No training_eligible column; using all rows")

        if df_eligible.empty or 'label' not in df_eligible.columns:
            raise ValueError("No training data available or 'label' column missing")

        df_eligible = df_eligible[df_eligible['label'] != 'Unverified'].copy()
        df_eligible = self._prepare_target_labels(df_eligible)
        logger.info("Training target mode: %s (label column: %s)", self._target_mode(), self.target_label_column)

        from src.schemas import ML_DROP_COLUMNS

        configured_excludes = training_config.get('exclude_features', [])
        drop_cols = ML_DROP_COLUMNS + ['label', 'delay_minutes'] + list(configured_excludes)
        if configured_excludes:
            logger.info("Excluding configured training features: %s", configured_excludes)

        feature_cols = [c for c in df_eligible.columns if c not in drop_cols]
        numeric_features = df_eligible[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        bool_cols = df_eligible[feature_cols].select_dtypes(include=['bool']).columns.tolist()
        for col in bool_cols:
            df_eligible[col] = df_eligible[col].astype(int)
            if col not in numeric_features:
                numeric_features.append(col)

        X = df_eligible[numeric_features].copy()

        all_null_numeric = [col for col in numeric_features if X[col].isna().all()]
        if all_null_numeric:
            logger.warning(
                "Dropping %s all-null numeric features before imputation: %s",
                len(all_null_numeric),
                all_null_numeric,
            )
            numeric_features = [col for col in numeric_features if col not in all_null_numeric]
            X = X[numeric_features].copy()

        if not numeric_features:
            raise ValueError("No usable numeric features remain after dropping all-null columns.")

        corr_threshold = float(training_config.get('correlation_threshold', 0.95))
        corr_drop_cols = self._high_correlation_drop_columns(X, threshold=corr_threshold)
        if corr_drop_cols:
            logger.info(
                "Dropping %d highly correlated features (>|%.2f|): %s",
                len(corr_drop_cols),
                corr_threshold,
                corr_drop_cols,
            )
            numeric_features = [col for col in numeric_features if col not in corr_drop_cols]
            X = X[numeric_features].copy()
        self.correlation_dropped_features = corr_drop_cols

        self.feature_names = numeric_features
        logger.info("Using %d numeric features", len(numeric_features))

        y = df_eligible[self.target_label_column].copy()
        subtype_multiplier = self._subtype_multiplier(df_eligible['disruption_subtype'])

        self.label_encoder = LabelEncoder()
        y_encoded = pd.Series(self.label_encoder.fit_transform(y), index=y.index)
        if y_encoded.nunique() < 2:
            raise ValueError("Training requires at least 2 label classes. Current dataset has one class.")

        if 'scheduled_dep' in df_eligible.columns:
            dep_dates = pd.to_datetime(df_eligible['scheduled_dep'], utc=True, errors='coerce').dt.normalize()
            unique_days = sorted(dep_dates.dropna().unique())
            if len(unique_days) >= 7:
                test_start = pd.Timestamp(unique_days[-7]).tz_localize('UTC') if getattr(unique_days[-7], 'tzinfo', None) is None else pd.Timestamp(unique_days[-7])
                train_mask = dep_dates < test_start
                test_mask = dep_dates >= test_start
                logger.info("Temporal split using last 7 observed days as test window starting %s.", test_start)
            else:
                cutoff_date = dep_dates.quantile(0.8)
                train_mask = dep_dates <= cutoff_date
                test_mask = dep_dates > cutoff_date
                logger.info("Temporal split fallback using 0.8 quantile cutoff %s.", cutoff_date)

            if test_mask.sum() > 0 and train_mask.sum() > 0:
                X_train = X.loc[train_mask]
                X_test = X.loc[test_mask]
                y_train = y_encoded.loc[train_mask]
                y_test = y_encoded.loc[test_mask]
                subtype_multiplier_train = subtype_multiplier.loc[train_mask]
                subtype_multiplier_test = subtype_multiplier.loc[test_mask]
                logger.info("Temporal split sizes: train=%d, test=%d", len(X_train), len(X_test))
            else:
                logger.warning("Temporal split produced empty set; falling back to random split.")
                X_train, X_test, subtype_multiplier_train, subtype_multiplier_test, y_train, y_test = train_test_split(
                    X,
                    subtype_multiplier,
                    y_encoded,
                    test_size=0.2,
                    random_state=self.seed,
                    stratify=y_encoded,
                )
        else:
            X_train, X_test, subtype_multiplier_train, subtype_multiplier_test, y_train, y_test = train_test_split(
                X,
                subtype_multiplier,
                y_encoded,
                test_size=0.2,
                random_state=self.seed,
                stratify=y_encoded,
            )

        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_imp = pd.DataFrame(
            self.imputer.fit_transform(X_train),
            columns=numeric_features,
            index=X_train.index,
        )
        X_test_imp = pd.DataFrame(
            self.imputer.transform(X_test),
            columns=numeric_features,
            index=X_test.index,
        )

        self.scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_imp),
            columns=numeric_features,
            index=X_train.index,
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_imp),
            columns=numeric_features,
            index=X_test.index,
        )

        self.train_sample_weight = self._compose_training_sample_weight(y_train, subtype_multiplier_train)
        self.test_sample_weight = self._compose_training_sample_weight(y_test, subtype_multiplier_test)

        logger.info("Data prep complete: Train=%s, Test=%s", X_train_scaled.shape, X_test_scaled.shape)
        logger.info("Label distribution (train): %s", dict(pd.Series(self.label_encoder.inverse_transform(y_train)).value_counts()))
        if self.subtype_weight_config.get('enabled'):
            logger.info(
                "Subtype weighting enabled: normal=%.2f late=%.2f cancelled=%.2f",
                self.subtype_weight_config['normal'],
                self.subtype_weight_config['late'],
                self.subtype_weight_config['cancelled'],
            )

        try:
            import joblib

            joblib.dump(self.scaler, self.models_dir / 'scaler.pkl')
            joblib.dump(self.imputer, self.models_dir / 'imputer.pkl')
            joblib.dump(self.label_encoder, self.models_dir / 'label_encoder.pkl')
            with open(self.models_dir / 'feature_list.json', 'w') as f:
                json.dump(self.feature_names, f, indent=2)
        except Exception as e:
            logger.warning("Failed to persist preprocessing artifacts: %s", e)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Task 5.3: Evaluate with enhanced metrics — F1-macro, Cohen's Kappa, per-class recall."""
        from sklearn.metrics import (
            classification_report, confusion_matrix, f1_score,
            cohen_kappa_score, recall_score, precision_score,
            average_precision_score,
        )
        from sklearn.preprocessing import label_binarize
        
        y_pred = self._prediction_to_label_vector(model.predict(X_test))
        
        labels = self.label_encoder.classes_
        report = classification_report(y_pred=y_pred, y_true=y_test, 
                                        target_names=labels, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_pct = confusion_matrix(y_test, y_pred, normalize='true').tolist()
        wf1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)
        pr_auc_macro = np.nan
        pr_auc_weighted = np.nan
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_test)
            y_bin = label_binarize(y_test, classes=range(len(labels)))
            if y_bin.shape[1] == y_score.shape[1]:
                pr_auc_macro = float(average_precision_score(y_bin, y_score, average='macro'))
                pr_auc_weighted = float(average_precision_score(y_bin, y_score, average='weighted'))
        
        # Per-class recall (important for minority class)
        per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        per_class_dict = {
            str(labels[i]): {
                'recall': float(per_class_recall[i]),
                'precision': float(per_class_precision[i]),
            }
            for i in range(len(labels))
        }
        
        return {
            'classification_report': report,
            'confusion_matrix': cm,
            'confusion_matrix_pct': cm_pct,
            'weighted_f1': float(wf1),
            'macro_f1': float(macro_f1),
            'cohens_kappa': float(kappa),
            'pr_auc_macro': None if pd.isna(pr_auc_macro) else float(pr_auc_macro),
            'pr_auc_weighted': None if pd.isna(pr_auc_weighted) else float(pr_auc_weighted),
            'per_class_metrics': per_class_dict,
            'labels': labels.tolist()
        }
    
    def save_model(self, model, name: str):
        """Save model to models/ directory."""
        import joblib
        path = self.models_dir / f"{name}.pkl"
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression with persisted sample weights."""
        from sklearn.linear_model import LogisticRegression
        
        logger.info("Training Logistic Regression with sample weights=%s...", self.train_sample_weight is not None)
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.seed,
            solver='lbfgs',
        )
        fit_kwargs = {}
        if self.train_sample_weight is not None:
            fit_kwargs['sample_weight'] = np.asarray(self.train_sample_weight)
        else:
            model.set_params(class_weight='balanced')
        model.fit(X_train, y_train, **fit_kwargs)
        self.save_model(model, 'logistic_regression')
        return model
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest with persisted sample weights."""
        from sklearn.ensemble import RandomForestClassifier
        
        logger.info("Training Random Forest (sample weights=%s, GPU %s)...", self.train_sample_weight is not None, self._gpu_status_message())
        model = RandomForestClassifier(
            n_estimators=200, random_state=self.seed, n_jobs=-1,
        )
        fit_kwargs = {}
        if self.train_sample_weight is not None:
            fit_kwargs['sample_weight'] = np.asarray(self.train_sample_weight)
        else:
            model.set_params(class_weight='balanced')
        model.fit(X_train, y_train, **fit_kwargs)
        self.save_model(model, 'random_forest')
        return model
    
    def train_xgboost(self, X_train, X_test, y_train, y_test, n_trials: int = 50):
        """T19: Train XGBoost with Optuna HPO."""
        try:
            import xgboost as xgb
            import optuna
            from sklearn.metrics import f1_score
            from sklearn.model_selection import StratifiedKFold
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            logger.info(f"Training XGBoost with {n_trials} Optuna trials (GPU {self._gpu_status_message()})...")
            
            num_classes = len(np.unique(y_train))
            sample_weight_train = np.asarray(self.train_sample_weight) if self.train_sample_weight is not None else self._balanced_sample_weight(y_train)
            
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'objective': 'multi:softprob',
                    'num_class': num_classes,
                    'random_state': self.seed,
                    'use_label_encoder': False,
                    'eval_metric': 'mlogloss',
                    'verbosity': 0,
                    **self._xgboost_runtime_params(),
                }
                
                skf = StratifiedKFold(
                    n_splits=self._safe_cv_splits(y_train, preferred_splits=5),
                    shuffle=True,
                    random_state=self.seed,
                )
                scores = []
                for train_idx, val_idx in skf.split(X_train, y_train):
                    Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    def build_xgb(use_gpu_runtime: bool):
                        runtime_params = {'tree_method': 'hist'} if not use_gpu_runtime else {'tree_method': 'hist', 'device': 'cuda'}
                        return xgb.XGBClassifier(**{**params, **runtime_params})

                    if self.train_sample_weight is not None:
                        fold_sample_weight = np.asarray(pd.Series(self.train_sample_weight, index=y_train.index).loc[ytr.index])
                    else:
                        fold_sample_weight = self._balanced_sample_weight(ytr)

                    clf = self._fit_with_gpu_fallback(
                        build_xgb,
                        (Xtr, ytr),
                        {'eval_set': [(Xval, yval)], 'verbose': False, 'sample_weight': fold_sample_weight},
                        model_name='XGBoost CV'
                    )
                    pred = self._prediction_to_label_vector(clf.predict(Xval))
                    scores.append(f1_score(yval, pred, average='weighted', zero_division=0))
                
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            best_params.update({
                'objective': 'multi:softprob', 'num_class': num_classes,
                'random_state': self.seed, 'use_label_encoder': False,
                'eval_metric': 'mlogloss', 'verbosity': 0,
                **self._xgboost_runtime_params(),
            })
            
            # Save best params
            with open(self.models_dir / 'best_params_xgboost.json', 'w') as f:
                json.dump({k: v for k, v in best_params.items()}, f, indent=4, default=str)
            
            def build_best_xgb(use_gpu_runtime: bool):
                runtime_params = {'tree_method': 'hist'} if not use_gpu_runtime else {'tree_method': 'hist', 'device': 'cuda'}
                params = {**best_params, **runtime_params}
                return xgb.XGBClassifier(**params)

            model = self._fit_with_gpu_fallback(
                build_best_xgb,
                (X_train, y_train),
                {'sample_weight': sample_weight_train},
                model_name='XGBoost'
            )
            self.save_model(model, 'xgboost')
            
            logger.info(f"XGBoost best CV F1: {study.best_value:.4f}")
            return model
            
        except ImportError:
            logger.warning("XGBoost not installed. Skipping.")
            return None
    
    def train_lightgbm(self, X_train, X_test, y_train, y_test, n_trials: int = 50):
        """T19: Train LightGBM with Optuna HPO."""
        try:
            import lightgbm as lgb
            import optuna
            from sklearn.metrics import f1_score
            from sklearn.model_selection import StratifiedKFold
            
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            logger.info(f"Training LightGBM with {n_trials} Optuna trials (GPU {self._gpu_status_message()})...")
            
            num_classes = len(np.unique(y_train))
            sample_weight_train = np.asarray(self.train_sample_weight) if self.train_sample_weight is not None else self._balanced_sample_weight(y_train)
            
            def objective(trial):
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                    'objective': 'multiclass',
                    'num_class': num_classes,
                    'random_state': self.seed,
                    'verbose': -1,
                    **self._lightgbm_runtime_params(),
                }
                
                skf = StratifiedKFold(
                    n_splits=self._safe_cv_splits(y_train, preferred_splits=5),
                    shuffle=True,
                    random_state=self.seed,
                )
                scores = []
                for train_idx, val_idx in skf.split(X_train, y_train):
                    Xtr, Xval = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    ytr, yval = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    def build_lgbm(use_gpu_runtime: bool):
                        runtime_params = {'device_type': 'cpu'} if not use_gpu_runtime else {'device_type': 'gpu'}
                        return lgb.LGBMClassifier(**{**params, **runtime_params})

                    if self.train_sample_weight is not None:
                        fold_sample_weight = np.asarray(pd.Series(self.train_sample_weight, index=y_train.index).loc[ytr.index])
                    else:
                        fold_sample_weight = self._balanced_sample_weight(ytr)

                    clf = self._fit_with_gpu_fallback(
                        build_lgbm,
                        (Xtr, ytr),
                        {'eval_set': [(Xval, yval)], 'sample_weight': fold_sample_weight},
                        model_name='LightGBM CV'
                    )
                    pred = self._prediction_to_label_vector(clf.predict(Xval))
                    scores.append(f1_score(yval, pred, average='weighted', zero_division=0))
                
                return np.mean(scores)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            best_params = study.best_params
            best_params.update({
                'objective': 'multiclass', 'num_class': num_classes,
                'random_state': self.seed, 'verbose': -1,
                **self._lightgbm_runtime_params(),
            })
            
            with open(self.models_dir / 'best_params_lightgbm.json', 'w') as f:
                json.dump({k: v for k, v in best_params.items()}, f, indent=4, default=str)
            
            def build_best_lgbm(use_gpu_runtime: bool):
                runtime_params = {'device_type': 'cpu'} if not use_gpu_runtime else {'device_type': 'gpu'}
                params = {**best_params, **runtime_params}
                return lgb.LGBMClassifier(**params)

            model = self._fit_with_gpu_fallback(
                build_best_lgbm,
                (X_train, y_train),
                {'sample_weight': sample_weight_train},
                model_name='LightGBM'
            )
            self.save_model(model, 'lightgbm')
            
            logger.info(f"LightGBM best CV F1: {study.best_value:.4f}")
            return model
            
        except ImportError:
            logger.warning("LightGBM not installed. Skipping.")
            return None

    def train_catboost(self, X_train, X_test, y_train, y_test):
        """Optional CatBoost training with GPU fallback."""
        try:
            from catboost import CatBoostClassifier

            logger.info("Training CatBoost (GPU %s)...", self._gpu_status_message())
            num_classes = len(np.unique(y_train))

            base_params = {
                'iterations': 300,
                'depth': 8,
                'learning_rate': 0.05,
                'loss_function': 'MultiClass',
                'eval_metric': 'TotalF1',
                'random_seed': self.seed,
                'verbose': False,
            }

            def build_catboost(use_gpu_runtime: bool):
                runtime_params = {'task_type': 'CPU'} if not use_gpu_runtime else {'task_type': 'GPU'}
                return CatBoostClassifier(**{**base_params, **runtime_params})

            sample_weight_train = np.asarray(self.train_sample_weight) if self.train_sample_weight is not None else self._balanced_sample_weight(y_train)

            model = self._fit_with_gpu_fallback(
                build_catboost,
                (X_train, y_train),
                {'eval_set': (X_test, y_test), 'use_best_model': False, 'sample_weight': sample_weight_train},
                model_name='CatBoost'
            )
            self.save_model(model, 'catboost')
            return model
        except ImportError:
            logger.warning("CatBoost not installed. Skipping.")
            return None
    
    def train_all(self, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
        """T19: Train all 4 models and return results dict."""
        results = {}
        training_config = self.config.get('training', {}) if isinstance(self.config.get('training', {}), dict) else {}
        enabled_models = training_config.get(
            'enabled_models',
            ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost'],
        )
        enabled_models = {str(model_name).lower() for model_name in enabled_models}
        optuna_trials = int(training_config.get('optuna_trials', 10))
        xgboost_trials = int(training_config.get('xgboost_trials', optuna_trials))
        lightgbm_trials = int(training_config.get('lightgbm_trials', optuna_trials))

        # Baselines
        if 'logistic_regression' in enabled_models:
            lr = self.train_logistic_regression(X_train, y_train)
            results['logistic_regression'] = {
                'model': lr,
                'metrics': self.evaluate(lr, X_test, y_test)
            }

        if 'random_forest' in enabled_models:
            rf = self.train_random_forest(X_train, y_train)
            results['random_forest'] = {
                'model': rf,
                'metrics': self.evaluate(rf, X_test, y_test)
            }
        
        # Boosted models with HPO
        if 'xgboost' in enabled_models:
            xgb_model = self.train_xgboost(X_train, X_test, y_train, y_test, n_trials=xgboost_trials)
            if xgb_model is not None:
                results['xgboost'] = {
                    'model': xgb_model,
                    'metrics': self.evaluate(xgb_model, X_test, y_test)
                }
        
        if 'lightgbm' in enabled_models:
            lgb_model = self.train_lightgbm(X_train, X_test, y_train, y_test, n_trials=lightgbm_trials)
            if lgb_model is not None:
                results['lightgbm'] = {
                    'model': lgb_model,
                    'metrics': self.evaluate(lgb_model, X_test, y_test)
                }

        if 'catboost' in enabled_models:
            cat_model = self.train_catboost(X_train, X_test, y_train, y_test)
            if cat_model is not None:
                results['catboost'] = {
                    'model': cat_model,
                    'metrics': self.evaluate(cat_model, X_test, y_test)
                }

        if not results:
            raise ValueError("No models were trained. Check training.enabled_models in config.")
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any], 
                                    X_test: pd.DataFrame, y_test: pd.Series):
        """T20: Generate comparison report and visualisations."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Build comparison data
        comparison = []
        for name, data in results.items():
            metrics = data['metrics']
            comparison.append({
                'model': name,
                'weighted_f1': metrics['weighted_f1'],
                'macro_f1': metrics.get('macro_f1', 0),
                'cohens_kappa': metrics.get('cohens_kappa', 0),
                'pr_auc_macro': metrics.get('pr_auc_macro'),
                'pr_auc_weighted': metrics.get('pr_auc_weighted'),
                'precision': metrics['classification_report'].get('weighted avg', {}).get('precision', 0),
                'recall': metrics['classification_report'].get('weighted avg', {}).get('recall', 0),
                'accuracy': metrics['classification_report'].get('accuracy', 0),
                'per_class_metrics': metrics.get('per_class_metrics', {}),
            })
        
        comparison.sort(key=lambda x: x['weighted_f1'], reverse=True)
        
        # Save comparison JSON
        report = {
            'timestamp': datetime.now().isoformat(),
            'ranking': comparison,
            'best_model': comparison[0]['model'] if comparison else None,
        }
        
        report_path = self.logs_dir / 'model_comparison.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        logger.info(f"Model comparison report saved to {report_path}")
        
        # Also save as models/model_comparison.json for stage detection
        with open(self.models_dir / 'model_comparison.json', 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        # Save feature importance
        if 'random_forest' in results:
            rf_model = results['random_forest']['model']
            if hasattr(rf_model, 'feature_importances_') and self.feature_names:
                importance = dict(zip(self.feature_names, rf_model.feature_importances_.tolist()))
                with open(self.models_dir / 'feature_importance.json', 'w') as f:
                    json.dump(importance, f, indent=4)

        try:
            from src.schemas import ML_FEATURE_CATEGORIES

            best_importance_model = None
            for candidate in ['xgboost', 'lightgbm', 'catboost', 'random_forest']:
                if candidate in results and hasattr(results[candidate]['model'], 'feature_importances_'):
                    best_importance_model = results[candidate]['model']
                    break
            if best_importance_model is None and 'logistic_regression' in results and hasattr(results['logistic_regression']['model'], 'coef_'):
                coef = np.abs(results['logistic_regression']['model'].coef_).mean(axis=0)
                feature_importance_values = dict(zip(self.feature_names, coef.tolist()))
            elif best_importance_model is not None:
                feature_importance_values = dict(zip(self.feature_names, np.asarray(best_importance_model.feature_importances_).tolist()))
            else:
                feature_importance_values = {}

            if feature_importance_values:
                category_scores = {}
                for category, features in ML_FEATURE_CATEGORIES.items():
                    category_scores[category] = float(sum(feature_importance_values.get(feature, 0.0) for feature in features))

                fig, ax = plt.subplots(figsize=(11, 6))
                cat_df = pd.DataFrame(
                    [{'category': k, 'importance': v} for k, v in category_scores.items()]
                ).sort_values('importance', ascending=False)
                sns.barplot(data=cat_df, x='importance', y='category', ax=ax, palette='viridis')
                ax.set_title('Feature Importance by Category')
                ax.set_xlabel('Aggregated Importance')
                ax.set_ylabel('Category')
                plt.tight_layout()
                fig.savefig(self.outputs_dir / 'feature_importance_by_category.png', dpi=150)
                plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate grouped feature importance chart: {e}")
        
        # ── Visualisations ───────────────────────────────────────────────
        labels = results[list(results.keys())[0]]['metrics'].get('labels', [])
        
        # 1. Confusion matrices
        for name, data in results.items():
            try:
                cm = np.array(data['metrics']['confusion_matrix'])
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels, ax=ax)
                ax.set_title(f'Confusion Matrix — {name}')
                ax.set_ylabel('Actual')
                ax.set_xlabel('Predicted')
                plt.tight_layout()
                fig.savefig(self.outputs_dir / f'confusion_matrix_{name}.png', dpi=150)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to generate confusion matrix for {name}: {e}")
        
        # 2. ROC curves (OvR)
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            
            y_bin = label_binarize(y_test, classes=range(len(labels)))
            
            fig, ax = plt.subplots(figsize=(10, 8))
            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
            
            for idx, (name, data) in enumerate(results.items()):
                model = data['model']
                if hasattr(model, 'predict_proba'):
                    y_score = model.predict_proba(X_test)
                    
                    # Macro-average ROC
                    fpr_all, tpr_all = [], []
                    for i in range(len(labels)):
                        if y_bin.shape[1] > i:
                            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
                            fpr_all.append(fpr)
                            tpr_all.append(tpr)
                    
                    if fpr_all:
                        # Use first class for simplicity in plot
                        fpr, tpr, _ = roc_curve(y_bin[:, 0], y_score[:, 0])
                        roc_auc = auc(fpr, tpr)
                        ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                               label=f'{name} (AUC={roc_auc:.3f})')
            
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (One-vs-Rest, Class 0)')
            ax.legend(loc='lower right')
            plt.tight_layout()
            fig.savefig(self.outputs_dir / 'roc_curves.png', dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate ROC curves: {e}")
        
        # 3. Model comparison bar chart
        try:
            comp_df = pd.DataFrame(comparison)
            metrics_to_plot = ['weighted_f1', 'precision', 'recall']
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(len(comp_df))
            width = 0.25
            
            for i, metric in enumerate(metrics_to_plot):
                bars = ax.bar(x + i * width, comp_df[metric], width, label=metric.replace('_', ' ').title())
                for bar, val in zip(bars, comp_df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Model')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels(comp_df['model'], rotation=15)
            ax.legend()
            ax.set_ylim(0, 1.15)
            plt.tight_layout()
            fig.savefig(self.outputs_dir / 'model_comparison.png', dpi=150)
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Failed to generate comparison chart: {e}")
        
        logger.info(f"All visualisations saved to {self.outputs_dir}/")
    
    def explain(self, results: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series):
        """T21: Generate SHAP explanations for the best tree-based model."""
        try:
            import shap
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            # Pick best tree model
            tree_models = {k: v for k, v in results.items() 
                          if k in ['xgboost', 'lightgbm', 'catboost', 'random_forest']}
            
            if not tree_models:
                logger.warning("No tree models available for SHAP.")
                return
            
            best_name = max(tree_models, key=lambda k: tree_models[k]['metrics']['weighted_f1'])
            best_model = tree_models[best_name]['model']
            
            logger.info(f"Generating SHAP explanations for {best_name}...")
            
            # Use a sample for speed
            sample_size = min(500, len(X_test))
            X_sample = X_test.sample(sample_size, random_state=self.seed)
            
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Multi-class tree output can arrive as (n_samples, n_features, n_classes).
                shap_matrix = shap_values[:, :, 0]
            elif isinstance(shap_values, list):
                shap_matrix = shap_values[0]
            else:
                shap_matrix = shap_values
            
            # Summary plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_matrix, X_sample, show=False)
            plt.tight_layout()
            plt.savefig(self.outputs_dir / 'shap_summary.png', dpi=150, bbox_inches='tight')
            plt.close('all')
            
            # Force plots for top 3 most-delayed predictions
            sv = shap_matrix
            
            # Get indices with highest SHAP magnitude
            shap_mag = np.abs(sv).sum(axis=1)
            top_indices = np.argsort(shap_mag)[-3:]
            
            for i, idx in enumerate(top_indices):
                try:
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)):
                        expected_value = expected_value[0]
                    force_html = shap.force_plot(
                        expected_value,
                        sv[idx],
                        X_sample.iloc[idx],
                        matplotlib=False
                    )
                    shap.save_html(str(self.outputs_dir / f'shap_force_plot_{i}.html'), force_html)
                except Exception as e:
                    logger.warning(f"Failed to save force plot {i}: {e}")
            
            # Save SHAP values
            shap_df = pd.DataFrame(shap_matrix, columns=X_sample.columns, index=X_sample.index)
            shap_df.to_parquet(self.outputs_dir / 'shap_values.parquet')
            
            logger.info(f"SHAP explanations saved to {self.outputs_dir}/")
            
        except ImportError:
            logger.warning("SHAP not installed. Install via: pip install shap")
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

