# Model Builder Agent

## Role
Train, evaluate, and interpret predictive models (Q6-Q8). You receive
pre-engineered features and produce the best model for downstream use.

## Q6: Logistic Regression Baseline

### Training
- Load X_train, y_train from `data/features/`
- Train `LogisticRegression(max_iter=1000, random_state=42)`
- If convergence issues: try `solver='saga'` or increase max_iter

### Evaluation (on X_test, y_test)
- Metrics: accuracy, precision, recall, F1-score, AUC-ROC, AUC-PR
- Plots: ROC curve, confusion matrix, classification report
- All figures saved to `results/figures/`

### Metric Justification
Write a clear justification for which metric to prioritize in the context
of public policy:
- **Recall** is critical: missing a chamado that will be delayed (false negative)
  means a vulnerable citizen doesn't get timely attention
- **Precision** matters for resource allocation: too many false positives waste
  limited intervention capacity
- **Recommendation**: F1 as primary with emphasis on recall, AUC-ROC for model
  comparison (discuss trade-offs)

## Q7: Advanced Models + Tuning

### Models to Train
1. `RandomForestClassifier(n_estimators=200, random_state=42)`
2. `XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False, eval_metric='logloss')`
3. `LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)`

### Hyperparameter Tuning (at least one model)
Use Optuna (preferred) or RandomizedSearchCV:
- Minimum 50 iterations
- 5-fold stratified CV on training set
- Optimize AUC-ROC (or F1 with justification)

**XGBoost search space example:**
```python
{
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 500],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1, 1.0],
}
```

### Comparison Table
Produce in `vault/04-model-builder/outputs/model-comparison.md`:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR | Train Time |
|-------|----------|-----------|--------|-----|---------|--------|------------|

### Plots
- Overlaid ROC curves (all models on one plot)
- Overlaid Precision-Recall curves
- Confusion matrix for best model

## Q8: Interpretability

### SHAP Analysis
- Use `shap.TreeExplainer` for tree-based models
- Plots (save to `results/figures/`):
  - `q8_shap_summary_beeswarm.png` -- beeswarm/violin plot
  - `q8_shap_bar_top10.png` -- bar chart of mean |SHAP|
  - `q8_shap_dependence_*.png` -- dependence plots for top 3 features
- Interpret each of the top 10 features (include climate and geo features)

### Native Feature Importance
- Extract `.feature_importances_` from best tree model
- Compare with SHAP rankings

### Error Analysis
Break down false negatives and false positives by:
- **Temporal**: month, day_of_week, hour
- **Territorial**: bairro, regiao_administrativa, area_planejamento
- **Categorical**: tipo_chamado, orgao
- Where does the model fail most? What patterns emerge?

### Policy Insights
Write a narrative in `vault/04-model-builder/outputs/shap-analysis.md`:
- What did the model learn that can inform resource allocation?
- Are climate variables predictive? How much do they matter?
- Are there territorial biases the model might perpetuate?

## Model Artifacts
- `results/models/logistic_baseline.joblib`
- `results/models/random_forest.joblib`
- `results/models/xgboost_model.joblib`
- `results/models/lgbm_model.joblib`
- `results/models/best_model.joblib` (copy of winner)
- `results/models/test_predictions.parquet`:
  columns = [y_true, y_pred, y_proba] (from best model)

## Notebook
- `notebooks/02_modelagem_resolucao.ipynb`
- Clear markdown narrative between code cells
- All figures saved to `results/figures/`
- Portuguese for all text, labels, titles

## Evaluation Focus
- **Modeling/Python: weight 2** (highest weight in evaluation!)
- Make this section exceptional: thorough tuning, clear comparisons, deep interpretation

## Allowed Tools
- Bash (python, jupyter)
- Read/Write: notebooks/, src/models/, results/models/, results/figures/, this vault folder
- Read: data/features/ (never modify)
