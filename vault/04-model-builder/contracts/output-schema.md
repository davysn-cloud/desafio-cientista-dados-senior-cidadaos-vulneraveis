# Model Builder - Output Contract

## Model Artifacts
| File | Format | Description |
|------|--------|-------------|
| `results/models/logistic_baseline.joblib` | joblib | Trained LogisticRegression |
| `results/models/random_forest.joblib` | joblib | Trained RandomForest |
| `results/models/xgboost_model.joblib` | joblib | Trained/Tuned XGBoost |
| `results/models/lgbm_model.joblib` | joblib | Trained LightGBM |
| `results/models/best_model.joblib` | joblib | Copy of best performing model |

## Predictions
- `results/models/test_predictions.parquet`
- Columns: `y_true` (int), `y_pred` (int), `y_proba` (float 0-1)
- y_proba = probability of class 1 (resolved in 7 days) from best model
- Rows match X_test row count

## Documentation
- `vault/04-model-builder/outputs/model-comparison.md`
- `vault/04-model-builder/outputs/best-model-report.md`
- `vault/04-model-builder/outputs/shap-analysis.md`

## Notebook
- `notebooks/02_modelagem_resolucao.ipynb`
