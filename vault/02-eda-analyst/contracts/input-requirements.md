# EDA Analyst - Input Requirements

## Required Files (from Data Engineer)

| File | Format | Min Rows | Key Columns Needed |
|------|--------|----------|--------------------|
| `data/raw/chamados_2023_2024.parquet` | Parquet | 100K | data_inicio, tipo, subtipo, bairro, latitude, longitude |
| `data/raw/weather_rio_2023_2024.csv` | CSV | 731 | time, temperature_2m_max, temperature_2m_min, precipitation_sum |
| `data/raw/holidays_br_2023_2024.csv` | CSV | 20 | date, localName |
| `data/raw/bairros.parquet` | Parquet | 100 | nome, geometria (optional) |
| `data/raw/areas_planejamento.parquet` | Parquet | 10 | nome |
| `data/raw/regioes_admin.parquet` | Parquet | 20 | nome |

## Pre-check Script
```python
import os
required = [
    'data/raw/chamados_2023_2024.parquet',
    'data/raw/weather_rio_2023_2024.csv',
    'data/raw/holidays_br_2023_2024.csv',
    'data/raw/bairros.parquet',
]
for f in required:
    assert os.path.exists(f), f"Missing: {f}"
print("All input files present.")
```
