# Milestone 8 — ML Training Pipeline

> **Status**: `[~]` In Progress (single-class labels prevent training)
> **Priority**: 🟠 HIGH
> **Depends on**: M7 (T15, T16 for quality-gated dataset)

---

## `[x]` TASK-17: Create model training module (scaffold + data prep)

**Files**: `src/model_training.py` [NEW], `requirements.txt`
**What**: Create `ModelTrainer` class with `prepare_data()` that loads the ML dataset, applies stratified train/test split, handles imputation and scaling. Does NOT yet train models — that's TASK-18/19.

**Steps**:
1. Create `src/model_training.py` with class `ModelTrainer(config: dict)`
2. `prepare_data(df) -> (X_train, X_test, y_train, y_test)`:
   - Filter to `training_eligible == True`
   - Drop identifier/provenance columns
   - Stratified 80/20 split on `label`, seed from config
   - `SimpleImputer(strategy='median')` for numerics
   - `StandardScaler` stored for later use
3. Add `scikit-learn`, `xgboost`, `lightgbm`, `optuna`, `shap` to `requirements.txt`

**Acceptance Criteria**:
- [ ] `ModelTrainer({'seed': 42}).prepare_data(df)` returns 4 objects with correct shapes
- [ ] `y_test.value_counts(normalize=True)` approximately matches `y_train.value_counts(normalize=True)` (stratified)
- [ ] No target leakage: `X_train` columns do not include `label`, `delay_minutes`, `actual_dep`, `actual_arr`
- [ ] `requirements.txt` lists pinned versions for all new deps

---

## `[~]` TASK-18: Train baseline models (LR + Random Forest)

**Files**: `src/model_training.py`
**What**: Add `train_logistic_regression()` and `train_random_forest()` methods. Each returns a fitted sklearn Pipeline. Add `evaluate(model, X_test, y_test) -> dict` that returns classification report + confusion matrix.

**Steps**:
1. `train_logistic_regression()`: Pipeline with StandardScaler + LogisticRegression(max_iter=1000, multi_class='multinomial')
2. `train_random_forest()`: Pipeline with RandomForestClassifier(n_estimators=200, random_state=seed)
3. `evaluate()`: returns `{'classification_report': dict, 'confusion_matrix': list[list], 'weighted_f1': float}`
4. `save_model(model, path)`: pickle to `models/{name}.pkl`

**Acceptance Criteria**:
- [ ] Both models train without error on the prepared dataset
- [ ] `evaluate()` returns dict with `weighted_f1` as a float between 0 and 1
- [ ] Models are saved to `models/logistic_regression.pkl` and `models/random_forest.pkl`
- [ ] Saved models can be loaded with `joblib.load()` and produce predictions

---

## `[~]` TASK-19: Train boosted models (XGBoost + LightGBM) with Optuna

**Files**: `src/model_training.py`
**What**: Add `train_xgboost()` and `train_lightgbm()` methods with Optuna hyperparameter tuning (5-fold stratified CV, 50 trials).

**Steps**:
1. `train_xgboost(n_trials=50)`: Optuna objective tunes `max_depth`, `learning_rate`, `n_estimators`, `subsample`. Metric: weighted F1.
2. `train_lightgbm(n_trials=50)`: Same approach for LightGBM params.
3. `train_all() -> dict[str, Pipeline]`: trains all 4 models, returns dict keyed by name
4. Save best hyperparameters to `models/best_params_{name}.json`

**Acceptance Criteria**:
- [ ] `train_xgboost()` completes 50 Optuna trials and returns a fitted model
- [ ] `models/best_params_xgboost.json` contains the best hyperparameters
- [ ] `train_all()` returns 4 models, each with `weighted_f1 > 0` on test set
- [ ] Total training time < 30 minutes on the dataset

---

## `[~]` TASK-20: Generate model comparison report and visuals

**Files**: `src/model_training.py`, `main.py`
**What**: After training all models, generate a comparison JSON report and matplotlib visuals: confusion matrix heatmaps, ROC curves (one-vs-rest), and a model comparison bar chart.

**Steps**:
1. Add `generate_comparison_report(results: dict) -> dict` that ranks models by weighted F1
2. Save to `logs/model_comparison.json`
3. Generate and save to `outputs/`:
   - `confusion_matrix_{model_name}.png` (seaborn heatmap)
   - `roc_curves.png` (all models, OvR)
   - `model_comparison.png` (grouped bar chart: F1, Precision, Recall per model)
4. Wire into `main.py` `stage_train()`

**Acceptance Criteria**:
- [ ] `logs/model_comparison.json` ranks models by `weighted_f1` descending
- [ ] `outputs/confusion_matrix_xgboost.png` exists and is a valid PNG
- [ ] `outputs/roc_curves.png` shows curves for all 4 models
- [ ] `outputs/model_comparison.png` has grouped bars with visible labels
