# Task: Q4 Multidimensional Demand Forecasting

## Objective
Build a regression model to predict daily call volume using climate, temporal, and geo features.

## Steps
1. Create daily aggregated dataset:
   - Target: daily total chamado count
   - Temporal features: day_of_week, month, is_weekend, is_holiday
   - Climate features: temperature_mean, precipitation_sum, is_extreme_rain
   - Geo features: top bairro volumes or AP-level aggregation
2. Train/test split: train=2023, test=2024
3. Train models: Ridge Regression (baseline), Random Forest Regressor
4. Evaluate: RMSE, MAE, R2 on test set
5. Feature importance analysis
6. Plot actual vs predicted time series

## Visualizations
- Time series: actual vs predicted (2024)
- Feature importance bar chart
- Residual analysis plot

## Output
- Figures in results/figures/q4_*
- Findings in vault/02-eda-analyst/outputs/q4-findings.md
