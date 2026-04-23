# BigQuery Cost-Saving Tips

## Golden Rule
**ALWAYS use partition filter**: `WHERE data_particao >= '2023-01-01'`

Without it, BigQuery scans all 14M+ rows and burns your free quota fast.

## Query Development Workflow
1. Start with `LIMIT 100` to inspect schema and data quality
2. Test aggregations on a single month first
3. Only run full query (2023-2024) when you're confident in the logic
4. Save result to Parquet/CSV immediately after successful query

## Caching Strategy
```python
import os
import pandas as pd
import basedosdados as bd

CACHE_PATH = 'data/raw/chamados_2023_2024.parquet'

if os.path.exists(CACHE_PATH):
    print(f"Loading cached data from {CACHE_PATH}")
    df = pd.read_parquet(CACHE_PATH)
else:
    print("Querying BigQuery...")
    query = """
    SELECT *
    FROM `datario.adm_central_atendimento_1746.chamado`
    WHERE data_particao >= '2023-01-01'
      AND data_particao <= '2024-12-31'
    """
    df = bd.read_sql(query, billing_project_id="YOUR_PROJECT")
    df.to_parquet(CACHE_PATH, index=False)
    print(f"Saved {len(df)} rows to {CACHE_PATH}")
```

## Useful Exploratory Queries

### Check schema
```sql
SELECT column_name, data_type
FROM `datario.adm_central_atendimento_1746.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'chamado'
```

### Count by year
```sql
SELECT EXTRACT(YEAR FROM data_particao) as ano, COUNT(*) as total
FROM `datario.adm_central_atendimento_1746.chamado`
WHERE data_particao >= '2023-01-01'
GROUP BY 1 ORDER BY 1
```

### Top tipos
```sql
SELECT tipo, COUNT(*) as total
FROM `datario.adm_central_atendimento_1746.chamado`
WHERE data_particao >= '2023-01-01'
GROUP BY 1 ORDER BY 2 DESC
LIMIT 20
```

## Free Tier Limits
- Google Cloud free tier: 1 TB of BigQuery queries per month
- basedosdados may have its own billing project
- Monitor usage at console.cloud.google.com
