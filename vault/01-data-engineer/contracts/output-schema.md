# Data Engineer - Output Contract

## File: data/raw/chamados_2023_2024.parquet
- **Format**: Apache Parquet
- **Expected rows**: 500K - 2M (filtered 2023-2024)
- **Required columns**: id_chamado, data_inicio, data_fim, data_particao, tipo,
  subtipo, bairro, latitude, longitude, orgao, status, categoria
- **Partition filter**: data_particao BETWEEN '2023-01-01' AND '2024-12-31'
- **Date format**: datetime64[ns] for data_inicio, data_fim

## File: data/raw/bairros.parquet
- **Format**: Apache Parquet (GeoParquet if geometry included)
- **Expected rows**: ~160
- **Required columns**: nome, id_bairro, subprefeitura, area_planejamento, geometria (optional)

## File: data/raw/areas_planejamento.parquet
- **Format**: Apache Parquet
- **Expected rows**: ~16
- **Required columns**: nome, id_area_planejamento, geometria (optional)

## File: data/raw/regioes_admin.parquet
- **Format**: Apache Parquet
- **Expected rows**: ~33
- **Required columns**: nome, id_regiao_administrativa

## File: data/raw/subprefeituras.parquet
- **Format**: Apache Parquet
- **Expected rows**: ~33
- **Required columns**: nome, id_subprefeitura

## File: data/raw/weather_rio_2023_2024.csv
- **Format**: CSV
- **Expected rows**: 731 (365 + 366 days)
- **Required columns**: time, temperature_2m_max, temperature_2m_min,
  temperature_2m_mean, precipitation_sum, rain_sum, windspeed_10m_max, weathercode
- **Date range**: 2023-01-01 to 2024-12-31

## File: data/raw/holidays_br_2023_2024.csv
- **Format**: CSV
- **Expected rows**: ~25 (holidays for 2 years)
- **Required columns**: date, localName, name, countryCode, fixed, global, types
