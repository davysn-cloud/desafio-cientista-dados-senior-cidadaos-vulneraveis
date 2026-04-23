# Task: Q5 Feature Engineering

## Objective
Create a 50K sample dataset with rich features for binary classification:
predict if a chamado will be resolved within 7 days.

## Steps
1. Load raw chamados, filter to 2023-2024
2. Compute target: resolved_in_7_days = (data_fim - data_inicio) <= 7 days
3. Remove chamados without data_fim (or justify keeping them as class 0)
4. Sample 50K: stratified by year and target
5. Merge weather data by date
6. Merge holiday data by date
7. Merge geographic data by bairro
8. Engineer all feature categories (temporal, climate, geo, categorical, contextual)
9. Apply encoding (target encoding with CV on train)
10. Normalize continuous features
11. Save train/test parquets and artifacts
12. Document everything in feature-catalog.md and feature-report.md

## Quality Checks
- [ ] No leakage (data_fim not in features)
- [ ] Train/test have identical columns
- [ ] Class distribution documented
- [ ] All features documented in catalog
