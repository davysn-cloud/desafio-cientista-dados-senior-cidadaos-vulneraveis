# Model Builder - Input Requirements

## Required Files (from Feature Engineer)

| File | Format | Validation |
|------|--------|------------|
| `data/features/X_train.parquet` | Parquet | All numeric, no target leakage |
| `data/features/X_test.parquet` | Parquet | Same columns as X_train |
| `data/features/y_train.parquet` | Parquet | Single column: resolved_in_7_days |
| `data/features/y_test.parquet` | Parquet | Single column: resolved_in_7_days |

## Pre-check
```python
import pandas as pd
X_train = pd.read_parquet('data/features/X_train.parquet')
X_test = pd.read_parquet('data/features/X_test.parquet')
y_train = pd.read_parquet('data/features/y_train.parquet')
y_test = pd.read_parquet('data/features/y_test.parquet')

assert list(X_train.columns) == list(X_test.columns), "Column mismatch!"
assert len(X_train) == len(y_train), "Row mismatch train!"
assert len(X_test) == len(y_test), "Row mismatch test!"
print(f"Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Class balance (train): {y_train['resolved_in_7_days'].value_counts(normalize=True).to_dict()}")
```
