# Task: Extract BigQuery Data

## Objective
Extract the main chamados table and all auxiliary tables from BigQuery.

## Steps
1. Ensure `basedosdados` is installed and GCP credentials are configured
2. Test main query with LIMIT 100 to verify schema
3. Run full extraction with partition filter: `data_particao >= '2023-01-01' AND data_particao <= '2024-12-31'`
4. Extract all 4 auxiliary tables (bairro, area_planejamento, regiao_administrativa, subprefeitura)
5. Save all files to `data/raw/` in Parquet format
6. Verify row counts and column names

## Acceptance Criteria
- [ ] chamados_2023_2024.parquet has >100K rows
- [ ] All 4 auxiliary parquet files exist and are non-empty
- [ ] Column names match output contract
