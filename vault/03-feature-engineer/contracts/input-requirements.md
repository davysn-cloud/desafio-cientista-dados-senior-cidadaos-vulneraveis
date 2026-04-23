# Feature Engineer - Input Requirements

## Required Files

| File | Format | Key Columns |
|------|--------|-------------|
| `data/raw/chamados_2023_2024.parquet` | Parquet | data_inicio, data_fim, tipo, subtipo, bairro, latitude, longitude, orgao, status, data_particao |
| `data/raw/weather_rio_2023_2024.csv` | CSV | time, temperature_2m_max, temperature_2m_min, temperature_2m_mean, precipitation_sum |
| `data/raw/holidays_br_2023_2024.csv` | CSV | date, localName |
| `data/raw/bairros.parquet` | Parquet | nome, id_bairro |
| `data/raw/regioes_admin.parquet` | Parquet | nome |
| `data/raw/subprefeituras.parquet` | Parquet | nome |

## Pre-check
```python
import pandas as pd
df = pd.read_parquet('data/raw/chamados_2023_2024.parquet')
assert 'data_inicio' in df.columns
assert 'data_fim' in df.columns
assert 'data_particao' in df.columns
print(f"Chamados: {len(df)} rows, {df.columns.tolist()}")
```
