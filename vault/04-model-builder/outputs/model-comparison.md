# Model Comparison (Q6-Q7)

## Metrics Table

| Modelo              |   Accuracy |   Precision |   Recall |     F1 |   AUC-ROC |   AUC-PR |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|---------:|
| Logistic Regression |     0.8259 |      0.8607 |   0.9277 | 0.893  |    0.848  |   0.9462 |
| Random Forest       |     0.8268 |      0.8651 |   0.9226 | 0.8929 |    0.8493 |   0.9535 |
| XGBoost (default)   |     0.8137 |      0.8613 |   0.9084 | 0.8842 |    0.8508 |   0.9566 |
| LightGBM            |     0.8237 |      0.8669 |   0.9153 | 0.8905 |    0.8609 |   0.9597 |
| XGBoost (tuned)     |     0.8283 |      0.8645 |   0.9259 | 0.8941 |    0.8628 |   0.9602 |


## Best Model: **XGBoost (tuned)**

- F1: 0.8941
- AUC-ROC: 0.8628
- AUC-PR: 0.9602

### Best Hyperparameters (Optuna)

- `max_depth`: 9
- `learning_rate`: 0.018596846637774906
- `n_estimators`: 288
- `subsample`: 0.8739532290914787
- `colsample_bytree`: 0.8819912795715482
- `min_child_weight`: 2
- `reg_alpha`: 0.023798833430266154
- `reg_lambda`: 4.191562880920944e-06

## Metric Justification

In the context of public policy for vulnerable citizens, **recall** is critical: failing to identify a chamado that will NOT be resolved in 7 days means a vulnerable citizen goes unattended. However, **precision** also matters for efficient resource allocation -- too many false positives waste limited government resources.

We recommend **F1-score** as the primary metric (balancing precision and recall) and **AUC-ROC** as the secondary metric for overall discrimination ability.