# Extraction Report

> Agent: 01-data-engineer
> Date: 2026-04-20
> Status: COMPLETE

## Summary

All 7 data files extracted successfully. No errors or retries needed.

| # | File | Rows | Status |
|---|------|------|--------|
| 1 | chamados_2023_2024.parquet | 2,792,446 | OK |
| 2 | bairros.parquet | 166 | OK |
| 3 | areas_planejamento.parquet | 5 | OK |
| 4 | regioes_admin.parquet | 33 | OK |
| 5 | subprefeituras.parquet | 11 | OK |
| 6 | weather_rio_2023_2024.csv | 731 | OK |
| 7 | holidays_br_2023_2024.csv | 29 | OK |

## Issues & Notes

### 1. Chamados: High null rate in coordinates
- `latitude` and `longitude` are null in ~1,542,702 rows (~55%)
- Downstream agents should handle missing geolocation gracefully
- Geospatial analysis (Q2) will be limited to the ~45% with valid coordinates

### 2. Chamados: Diagnostic fields mostly null
- `data_alvo_diagnostico`: ~97% null (2,710,574 of 2,792,446)
- `data_real_diagnostico`: ~97% null (2,703,175)
- `justificativa_status`: ~94% null (2,619,599)
- These fields may not be useful for modeling

### 3. Auxiliary table: subprefeituras
- Only 11 rows (fewer than expected ~33)
- Column is named `geometria` instead of `geometry` (inconsistent with other tables)
- Downstream agents should account for this naming difference

### 4. Auxiliary table: areas_planejamento
- Only 5 rows (spec expected ~16). This is correct -- Rio has 5 APs (1 through 5)

### 5. Holidays encoding
- Holiday names contain Portuguese characters (e.g., "Confraternização Universal")
- UTF-8 encoding preserved in CSV

## Billing Project
- BigQuery project: `desafio-rio-494000`
- All queries filtered with `data_particao >= '2023-01-01'`

## Extraction Scripts
- `src/data/extract_bigquery.py` -- BigQuery extraction with caching
- `src/data/extract_weather.py` -- Open-Meteo API extraction
- `src/data/extract_holidays.py` -- Nager.Date API extraction
