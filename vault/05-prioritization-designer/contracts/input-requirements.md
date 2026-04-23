# Prioritization Designer - Input Requirements

## Required Files

| File | Source Agent | Key Columns |
|------|-------------|-------------|
| `results/models/test_predictions.parquet` | Model Builder | y_true, y_pred, y_proba |
| `data/features/X_test.parquet` | Feature Engineer | All feature columns |
| `data/raw/bairros.parquet` | Data Engineer | nome, geometria |

## Pre-check
```python
import pandas as pd

preds = pd.read_parquet('results/models/test_predictions.parquet')
assert 'y_proba' in preds.columns, "Missing y_proba!"
assert 'y_true' in preds.columns, "Missing y_true!"
assert preds['y_proba'].between(0, 1).all(), "y_proba out of [0,1] range"

X_test = pd.read_parquet('data/features/X_test.parquet')
assert len(preds) == len(X_test), "Row count mismatch between predictions and features"

print(f"Test set: {len(preds)} chamados")
print(f"Delayed rate: {(preds['y_true'] == 0).mean():.2%}")
print(f"Top 20% threshold: {len(preds) // 5} chamados")
```
