# Task: Extract Auxiliary Tables

## Objective
Extract geographic reference tables from BigQuery.

## Tables
1. `datario.dados_mestres.bairro` -> `data/raw/bairros.parquet`
2. `datario.dados_mestres.area_planejamento` -> `data/raw/areas_planejamento.parquet`
3. `datario.dados_mestres.regiao_administrativa` -> `data/raw/regioes_admin.parquet`
4. `datario.dados_mestres.subprefeitura` -> `data/raw/subprefeituras.parquet`

## Steps
1. Query each table (no filters needed - these are small reference tables)
2. Save as Parquet
3. Document schema in data-catalog.md
