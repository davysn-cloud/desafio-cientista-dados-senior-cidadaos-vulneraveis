# Quality Gates

## Phase 1 Gate: Data Extraction

- [ ] `data/raw/chamados_2023_2024.parquet` exists and has >100K rows
- [ ] `data/raw/weather_rio_2023_2024.csv` exists and has exactly 731 rows
- [ ] `data/raw/holidays_br_2023_2024.csv` exists and has >10 entries per year
- [ ] `data/raw/bairros.parquet` exists and is non-empty
- [ ] `data/raw/areas_planejamento.parquet` exists and is non-empty
- [ ] `data/raw/regioes_admin.parquet` exists and is non-empty
- [ ] `data/raw/subprefeituras.parquet` exists and is non-empty
- [ ] `vault/01-data-engineer/outputs/data-catalog.md` is complete
- [ ] `vault/01-data-engineer/outputs/extraction-report.md` documents any issues

**Validation script:**
```python
import pandas as pd
df = pd.read_parquet('data/raw/chamados_2023_2024.parquet')
assert len(df) > 100_000, f"Expected >100K rows, got {len(df)}"
assert 'data_inicio' in df.columns
assert 'data_fim' in df.columns
weather = pd.read_csv('data/raw/weather_rio_2023_2024.csv')
assert len(weather) == 731, f"Expected 731 weather rows, got {len(weather)}"
```

## Phase 2 Gate: EDA + Feature Engineering

### EDA (Agent 2)
- [ ] `notebooks/01_analise_apis_clima.ipynb` exists and runs without errors
- [ ] At least 8 figures saved in `results/figures/`
- [ ] Each question (Q1-Q4) has findings documented in vault outputs
- [ ] Q4 regression model reports RMSE, MAE, R2

### Features (Agent 3)
- [ ] `data/features/X_train.parquet` exists
- [ ] `data/features/X_test.parquet` exists
- [ ] `data/features/y_train.parquet` exists
- [ ] `data/features/y_test.parquet` exists
- [ ] X_train and X_test have identical columns
- [ ] No column named `data_fim`, `resolved`, or `target` in X files (leakage check)
- [ ] `vault/03-feature-engineer/outputs/feature-catalog.md` documents all features
- [ ] Total sample ~50K (train + test)
- [ ] Train data from 2023, test data from 2024

**Leakage validation script:**
```python
import pandas as pd
X_train = pd.read_parquet('data/features/X_train.parquet')
X_test = pd.read_parquet('data/features/X_test.parquet')
assert list(X_train.columns) == list(X_test.columns), "Column mismatch"
forbidden = {'data_fim', 'resolved', 'resolved_in_7_days', 'target', 'y'}
leak = forbidden & set(X_train.columns)
assert len(leak) == 0, f"Leakage detected: {leak}"
```

## Phase 3 Gate: Modeling

- [ ] `notebooks/02_modelagem_resolucao.ipynb` exists and runs without errors
- [ ] `vault/04-model-builder/outputs/model-comparison.md` shows >= 4 models
- [ ] Best model AUC-ROC > 0.60
- [ ] `results/models/best_model.joblib` exists
- [ ] `results/models/test_predictions.parquet` exists with columns: y_true, y_pred, y_proba
- [ ] SHAP analysis covers top 10 features in `vault/04-model-builder/outputs/shap-analysis.md`

**Validation script:**
```python
import pandas as pd
preds = pd.read_parquet('results/models/test_predictions.parquet')
assert 'y_proba' in preds.columns, "Missing y_proba column"
assert 'y_true' in preds.columns, "Missing y_true column"
assert preds['y_proba'].between(0, 1).all(), "y_proba out of range"
```

## Phase 4 Gate: Prioritization

- [ ] `notebooks/03_sistema_priorizacao.ipynb` exists and runs without errors
- [ ] `vault/05-prioritization-designer/outputs/score-formula.md` has explicit formula
- [ ] `vault/05-prioritization-designer/outputs/simulation-results.md` has comparison metrics
- [ ] Score-based strategy shows lift > 1.0 vs random
- [ ] Lift curve figure exists in `results/figures/`

## Phase 5 Gate: Final Deliverables

- [ ] `README.md` has reproduction instructions
- [ ] `requirements.txt` lists all dependencies
- [ ] All 3 notebooks run end-to-end without errors
- [ ] All text/labels/titles in Portuguese
- [ ] No data files >100MB committed to git
- [ ] Figures are readable and have proper labels
