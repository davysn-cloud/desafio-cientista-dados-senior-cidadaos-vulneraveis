# Task: Q7 Advanced Models + Tuning

## Steps
1. Train RandomForest, XGBoost, LightGBM with default hyperparams
2. Evaluate all on test set (same metrics as Q6)
3. Select one model for hyperparameter tuning with Optuna (50+ trials, 5-fold CV)
4. Retrain best config on full training set
5. Produce comparison table with all models
6. Plot overlaid ROC curves and PR curves
7. Save all models to results/models/
8. Document best params and CV results
