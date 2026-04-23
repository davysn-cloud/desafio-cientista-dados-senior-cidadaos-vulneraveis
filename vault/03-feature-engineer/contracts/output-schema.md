# Feature Engineer - Output Contract

## X_train / X_test (Parquet)
- Identical column set and order
- No column named: data_fim, resolved, resolved_in_7_days, target, y
- All values numeric (encoded categoricals)
- No infinite values
- Missing values allowed only if model handles them (document which)

## y_train / y_test (Parquet)
- Single column: `resolved_in_7_days` (int: 0 or 1)
- Same number of rows as corresponding X file

## Row Counts
- X_train.shape[0] == y_train.shape[0]
- X_test.shape[0] == y_test.shape[0]
- Total ~50,000
- Train: 2023 data, Test: 2024 data

## Artifacts
- `results/models/feature_scaler.joblib`: fitted StandardScaler
- `results/models/target_encoders.joblib`: fitted target encoders

## Validation
```python
import pandas as pd
X_train = pd.read_parquet('data/features/X_train.parquet')
X_test = pd.read_parquet('data/features/X_test.parquet')
y_train = pd.read_parquet('data/features/y_train.parquet')
y_test = pd.read_parquet('data/features/y_test.parquet')

assert list(X_train.columns) == list(X_test.columns)
assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)
assert y_train.columns.tolist() == ['resolved_in_7_days']
assert set(y_train['resolved_in_7_days'].unique()).issubset({0, 1})
forbidden = {'data_fim', 'resolved', 'resolved_in_7_days', 'target', 'y'}
assert len(forbidden & set(X_train.columns)) == 0
print(f"Train: {len(X_train)}, Test: {len(X_test)}, Features: {X_train.shape[1]}")
```
