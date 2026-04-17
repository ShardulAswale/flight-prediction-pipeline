# Milestone 9 — Explainability & Feature Analysis

> **Status**: `[~]` In Progress (depends on successful model training)
> **Priority**: 🟠 MEDIUM
> **Depends on**: M8 (T20 for trained models)

---

## `[~]` TASK-21: Generate SHAP explanations for best model

**Files**: `src/model_training.py`
**What**: Add `explain(model, X_test) -> shap.Explanation` method. Generate SHAP summary plot and save. Generate SHAP force plot for top 3 most-delayed predictions.

**Steps**:
1. Add `explain()` using `shap.TreeExplainer` for tree models, `shap.LinearExplainer` for LR
2. Save `outputs/shap_summary.png` (beeswarm plot)
3. Save `outputs/shap_force_plot_{i}.html` for top 3 delayed predictions
4. Save SHAP values to `outputs/shap_values.parquet`

**Acceptance Criteria**:
- [ ] `outputs/shap_summary.png` exists and shows feature importance
- [ ] `outputs/shap_force_plot_0.html` is a valid HTML file openable in browser
- [ ] `outputs/shap_values.parquet` has same number of rows as `X_test`

---

## `[x]` TASK-22: Create feature selection notebook

**Files**: `notebooks/06_feature_selection.ipynb` [NEW]
**What**: Jupyter notebook that computes: (1) missingness heatmap, (2) Pearson + Spearman correlation matrix, (3) mutual information ranking vs label, (4) permutation importance from best model, (5) leakage risk checklist.

**Steps**:
1. Create notebook with 5 sections, each generating a titled visual
2. Section 1: `seaborn.heatmap(df.isnull())`
3. Section 2: `df.corr(method='pearson')` and `df.corr(method='spearman')` side by side
4. Section 3: `mutual_info_classif(X, y)` bar chart
5. Section 4: `permutation_importance(best_model, X_test, y_test)` bar chart
6. Section 5: Markdown table listing each feature with leakage risk assessment

**Acceptance Criteria**:
- [ ] Notebook runs end-to-end without errors (`jupyter nbconvert --execute`)
- [ ] All 5 sections produce visible output (plots or tables)
- [ ] Mutual information ranking identifies top 5 features
- [ ] Leakage risk table covers all feature columns
