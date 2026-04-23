# Data Catalog

> Generated: 2026-04-20

## BigQuery Tables

### chamados_2023_2024.parquet
- **Path**: `data/raw/chamados_2023_2024.parquet`
- **Source**: `datario.adm_central_atendimento_1746.chamado`
- **Rows**: 2,792,446
- **Columns** (34): id_chamado, id_origem_ocorrencia, data_inicio, data_fim, id_bairro, id_territorialidade, id_logradouro, numero_logradouro, id_unidade_organizacional, nome_unidade_organizacional, id_unidade_organizacional_mae, unidade_organizacional_ouvidoria, categoria, id_tipo, tipo, id_subtipo, subtipo, status, longitude, latitude, data_alvo_finalizacao, data_alvo_diagnostico, data_real_diagnostico, tempo_prazo, prazo_unidade, prazo_tipo, dentro_prazo, situacao, tipo_situacao, justificativa_status, reclamacoes, extracted_at, updated_at, data_particao
- **Date range**: 2023-01-01 to 2024-12-31
- **Filter applied**: `data_particao >= '2023-01-01' AND data_particao <= '2024-12-31'`
- **Notable nulls**: latitude/longitude (~55% null), data_alvo_diagnostico (~97% null), justificativa_status (~94% null)
- **File size**: ~145 MB

### bairros.parquet
- **Path**: `data/raw/bairros.parquet`
- **Source**: `datario.dados_mestres.bairro`
- **Rows**: 166
- **Columns** (12): id_bairro, nome, id_area_planejamento, id_regiao_planejamento, nome_regiao_planejamento, id_regiao_administrativa, nome_regiao_administrativa, subprefeitura, area, perimetro, geometry_wkt, geometry

### areas_planejamento.parquet
- **Path**: `data/raw/areas_planejamento.parquet`
- **Source**: `datario.dados_mestres.area_planejamento`
- **Rows**: 5
- **Columns** (6): id_area_planejamento, id_area_planejamento_numerico, area, perimetro, geometry_wkt, geometry

### regioes_admin.parquet
- **Path**: `data/raw/regioes_admin.parquet`
- **Source**: `datario.dados_mestres.regiao_administrativa`
- **Rows**: 33
- **Columns** (10): id_regiao_administrativa, nome, id_area_planejamento, id_area_planejamento_numerico, id_area_planejamento_sms, area_total, area, perimetro, geometry_wkt, geometry

### subprefeituras.parquet
- **Path**: `data/raw/subprefeituras.parquet`
- **Source**: `datario.dados_mestres.subprefeitura`
- **Rows**: 11
- **Columns** (5): subprefeitura, area, perimetro, geometry_wkt, geometria

## API Data

### weather_rio_2023_2024.csv
- **Path**: `data/raw/weather_rio_2023_2024.csv`
- **Source**: Open-Meteo Historical Archive API
- **Location**: Rio de Janeiro (-22.9068, -43.1729)
- **Rows**: 731 (daily, 2023-01-01 to 2024-12-31)
- **Columns** (8): time, temperature_2m_max, temperature_2m_min, temperature_2m_mean, precipitation_sum, rain_sum, windspeed_10m_max, weathercode
- **Timezone**: America/Sao_Paulo

### holidays_br_2023_2024.csv
- **Path**: `data/raw/holidays_br_2023_2024.csv`
- **Source**: Nager.Date Public Holiday API
- **Rows**: 29
- **Columns** (9): date, localName, name, countryCode, fixed, global, counties, launchYear, types
- **Years**: 2023, 2024
