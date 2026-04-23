# Data Dictionary

## Main Table: chamado (1746)

Source: `datario.adm_central_atendimento_1746.chamado`

| Column | Type | Description |
|--------|------|-------------|
| id_chamado | string | Unique identifier for the service call |
| data_inicio | datetime | When the chamado was opened |
| data_fim | datetime | When the chamado was closed/resolved (null if still open) |
| data_particao | date | Partition date (use for filtering!) |
| tipo | string | Main category of the service call |
| subtipo | string | Subcategory of the service call |
| bairro | string | Neighborhood where the issue is located |
| latitude | float | Geographic latitude |
| longitude | float | Geographic longitude |
| orgao | string | Government agency responsible |
| status | string | Current status of the chamado |
| categoria | string | Additional categorization |

## Auxiliary Tables

### bairro
| Column | Type | Description |
|--------|------|-------------|
| nome | string | Neighborhood name |
| id_bairro | string | Unique ID |
| subprefeitura | string | Sub-prefecture |
| area_planejamento | string | Planning area |
| geometria | geometry | GeoJSON boundary (if available) |

### area_planejamento
| Column | Type | Description |
|--------|------|-------------|
| nome | string | Planning area name |
| id_area_planejamento | string | Unique ID |

### regiao_administrativa
| Column | Type | Description |
|--------|------|-------------|
| nome | string | Administrative region name |
| id_regiao_administrativa | string | Unique ID |

### subprefeitura
| Column | Type | Description |
|--------|------|-------------|
| nome | string | Sub-prefecture name |
| id_subprefeitura | string | Unique ID |

## Weather Data (Open-Meteo)

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| time | date | - | Date |
| temperature_2m_max | float | C | Daily maximum temperature |
| temperature_2m_min | float | C | Daily minimum temperature |
| temperature_2m_mean | float | C | Daily mean temperature |
| precipitation_sum | float | mm | Total daily precipitation |
| rain_sum | float | mm | Total daily rain |
| windspeed_10m_max | float | km/h | Maximum wind speed |
| weathercode | int | WMO | WMO weather interpretation code |

### WMO Weather Codes
| Code | Description |
|------|-------------|
| 0 | Clear sky |
| 1-3 | Mainly clear to overcast |
| 45-48 | Fog |
| 51-57 | Drizzle |
| 61-67 | Rain |
| 71-77 | Snow (unlikely for Rio) |
| 80-82 | Rain showers |
| 85-86 | Snow showers |
| 95-99 | Thunderstorm |

## Holidays Data (Nager.Date API)

| Column | Type | Description |
|--------|------|-------------|
| date | date | Holiday date |
| localName | string | Name in Portuguese |
| name | string | Name in English |
| countryCode | string | BR |
| fixed | boolean | Whether date is fixed yearly |
| global | boolean | Whether holiday is national |
| types | string | Holiday type (Public, Bank, etc.) |
