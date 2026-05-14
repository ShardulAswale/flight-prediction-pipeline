"""
Patch notebook 10: replace the broken prediction cell with a proper
single-flight SHAP TreeExplainer cell.
"""
import json
from pathlib import Path

nb_path = Path(__file__).parent / "10_threshold_hyperparameter_experiments.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

# Find and remove the broken cell (id "b3d14619")
cells = nb["cells"]
cells = [c for c in cells if c.get("id") != "b3d14619"]

# New markdown header cell
header_cell = {
    "cell_type": "markdown",
    "id": "shap_flight_header",
    "metadata": {},
    "source": [
        "## Single-Flight Prediction & SHAP Explanation\n",
        "\n",
        "Pick one flight from the **test set**, show its route (origin \u2192 destination),\n",
        "run the XGBoost prediction, and draw a **SHAP waterfall plot** to explain\n",
        "which features drove the model\u2019s decision for that specific flight."
    ]
}

# New code cell
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "shap_flight_prediction",
    "metadata": {},
    "outputs": [],
    "source": [
        "# ============================================================\n",
        "# Single-flight prediction + SHAP TreeExplainer waterfall plot\n",
        "# ============================================================\n",
        "import joblib\n",
        "import shap\n",
        "import xgboost as xgb\n",
        "from IPython.display import display\n",
        "\n",
        "# ------------------------------------------------------------------\n",
        "# 1) Load the saved XGBoost model (trained earlier in the pipeline)\n",
        "# ------------------------------------------------------------------\n",
        "xgb_model_path = PROJECT_ROOT / 'models' / 'xgboost.pkl'\n",
        "xgb_model = joblib.load(xgb_model_path)\n",
        "print(f'Loaded XGBoost model from {xgb_model_path}')\n",
        "\n",
        "# ------------------------------------------------------------------\n",
        "# 2) Pick a test-set flight \u2014 preferring a disrupted one for interest\n",
        "# ------------------------------------------------------------------\n",
        "disrupted_mask = y_test != normal_idx\n",
        "disrupted_indices_test = X_test.index[disrupted_mask]\n",
        "\n",
        "rng = np.random.default_rng(42)\n",
        "if len(disrupted_indices_test) > 0:\n",
        "    chosen_idx = rng.choice(disrupted_indices_test)\n",
        "else:\n",
        "    chosen_idx = rng.choice(X_test.index)\n",
        "\n",
        "# Look up flight metadata from the original DataFrame\n",
        "flight_meta = df.loc[chosen_idx, ['flight_key', 'origin', 'destination',\n",
        "                                   'scheduled_dep', 'delay_minutes',\n",
        "                                   'label', 'disruption_subtype']]\n",
        "origin = flight_meta['origin']\n",
        "destination = flight_meta['destination']\n",
        "\n",
        "print()\n",
        "print('=' * 80)\n",
        "print(f'  SELECTED FLIGHT: {flight_meta[\"flight_key\"]}')\n",
        "print(f'  Route          : {origin} \\u2192 {destination}')\n",
        "print(f'  Scheduled dep  : {flight_meta[\"scheduled_dep\"]}')\n",
        "print(f'  Actual delay   : {flight_meta[\"delay_minutes\"]} min')\n",
        "print(f'  True label     : {flight_meta[\"label\"]} ({flight_meta[\"disruption_subtype\"]})')\n",
        "print('=' * 80)\n",
        "\n",
        "# ------------------------------------------------------------------\n",
        "# 3) Predict with XGBoost\n",
        "# ------------------------------------------------------------------\n",
        "x_single = X_test.loc[[chosen_idx]]\n",
        "\n",
        "pred_encoded = xgb_model.predict(x_single)[0]\n",
        "pred_label = trainer.label_encoder.inverse_transform([int(pred_encoded)])[0]\n",
        "\n",
        "pred_proba = xgb_model.predict_proba(x_single)[0]\n",
        "proba_df = pd.DataFrame({\n",
        "    'Class': trainer.label_encoder.classes_,\n",
        "    'Probability': pred_proba,\n",
        "}).sort_values('Probability', ascending=False)\n",
        "\n",
        "true_encoded = y_test.loc[chosen_idx]\n",
        "true_label = trainer.label_encoder.inverse_transform([int(true_encoded)])[0]\n",
        "\n",
        "print(f'\\nPredicted class : {pred_label}')\n",
        "print(f'True class      : {true_label}')\n",
        "print(f'Match           : {\"\\u2705\" if pred_label == true_label else \"\\u274c\"}')\n",
        "print('\\nClass probabilities:')\n",
        "display(proba_df.style.format({'Probability': '{:.4f}'}).hide(axis='index'))\n",
        "\n",
        "# ------------------------------------------------------------------\n",
        "# 4) SHAP TreeExplainer \u2014 waterfall plot for the predicted class\n",
        "# ------------------------------------------------------------------\n",
        "explainer = shap.TreeExplainer(xgb_model)\n",
        "shap_values = explainer(x_single)\n",
        "\n",
        "# For multi-class: show SHAP for the predicted class\n",
        "predicted_class_idx = int(pred_encoded)\n",
        "\n",
        "if shap_values.values.ndim == 3:\n",
        "    # shape: (1, n_features, n_classes) \u2014 pick predicted class\n",
        "    shap_explanation = shap.Explanation(\n",
        "        values=shap_values.values[0, :, predicted_class_idx],\n",
        "        base_values=shap_values.base_values[0, predicted_class_idx],\n",
        "        data=shap_values.data[0],\n",
        "        feature_names=shap_values.feature_names,\n",
        "    )\n",
        "else:\n",
        "    shap_explanation = shap_values[0]\n",
        "\n",
        "print(f'\\nSHAP waterfall plot for predicted class \"{pred_label}\"')\n",
        "print(f'Route: {origin} \\u2192 {destination}\\n')\n",
        "\n",
        "shap.plots.waterfall(shap_explanation, max_display=15, show=False)\n",
        "fig = plt.gcf()\n",
        "fig.suptitle(\n",
        "    f'SHAP Explanation \\u2014 {flight_meta[\"flight_key\"]}\\n'\n",
        "    f'{origin} \\u2192 {destination}  |  Predicted: {pred_label}  |  True: {true_label}',\n",
        "    fontsize=11,\n",
        "    y=1.02,\n",
        ")\n",
        "fig.set_size_inches(10, 8)\n",
        "plt.tight_layout()\n",
        "\n",
        "shap_plot_path = OUTPUT_DIR / 'shap_single_flight_waterfall.png'\n",
        "fig.savefig(shap_plot_path, dpi=180, bbox_inches='tight')\n",
        "plt.show()\n",
        "print(f'Saved: {shap_plot_path}')\n",
        "\n",
        "# ------------------------------------------------------------------\n",
        "# 5) SHAP force plot (inline HTML)\n",
        "# ------------------------------------------------------------------\n",
        "shap.initjs()\n",
        "\n",
        "if shap_values.values.ndim == 3:\n",
        "    force = shap.force_plot(\n",
        "        explainer.expected_value[predicted_class_idx],\n",
        "        shap_values.values[0, :, predicted_class_idx],\n",
        "        x_single.iloc[0],\n",
        "        feature_names=list(X_test.columns),\n",
        "    )\n",
        "else:\n",
        "    force = shap.force_plot(\n",
        "        explainer.expected_value,\n",
        "        shap_values.values[0],\n",
        "        x_single.iloc[0],\n",
        "        feature_names=list(X_test.columns),\n",
        "    )\n",
        "\n",
        "display(force)"
    ]
}

cells.append(header_cell)
cells.append(code_cell)
nb["cells"] = cells

nb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
print(f"Patched {nb_path.name}: removed broken cell, added SHAP flight prediction cells.")
