# Feature Catalog (Q5)

**Total features:** 34
**Train samples:** 24048 | **Test samples:** 25952

## Feature List

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `hour_of_day` | int | Hora do dia do chamado (0-23) |
| 2 | `day_of_week` | int | Dia da semana (0=seg, 6=dom) |
| 3 | `day_of_month` | int | Dia do mês (1-31) |
| 4 | `month` | int | Mês (1-12) |
| 5 | `quarter` | int | Trimestre (1-4) |
| 6 | `is_weekend` | binary | 1 se fim de semana |
| 7 | `is_holiday` | binary | 1 se feriado nacional |
| 8 | `is_business_hours` | binary | 1 se horário comercial (seg-sex 8h-18h) |
| 9 | `days_since_last_holiday` | int | Dias desde o último feriado |
| 10 | `days_until_next_holiday` | int | Dias até o próximo feriado |
| 11 | `temp_max` | float | Temperatura máxima do dia (°C) |
| 12 | `temp_min` | float | Temperatura mínima do dia (°C) |
| 13 | `temp_mean` | float | Temperatura média do dia (°C) |
| 14 | `precipitation_sum` | float | Precipitação acumulada do dia (mm) |
| 15 | `rain_sum` | float | Chuva acumulada do dia (mm) |
| 16 | `windspeed_max` | float | Velocidade máxima do vento (km/h) |
| 17 | `is_extreme_rain` | binary | 1 se precipitação > percentil 95 (treino) |
| 18 | `is_extreme_heat` | binary | 1 se temperatura máxima > 35°C |
| 19 | `latitude` | float | Latitude do chamado |
| 20 | `longitude` | float | Longitude do chamado |
| 21 | `coords_missing` | binary | 1 se coords ausente |
| 22 | `bairro_encoded` | float | Bairro target-encoded (CV) |
| 23 | `regiao_admin_encoded` | float | Região administrativa target-encoded |
| 24 | `area_plan_encoded` | float | Área de planejamento target-encoded |
| 25 | `subprefeitura_encoded` | float | Subprefeitura target-encoded |
| 26 | `tipo_encoded` | float | Tipo do chamado target-encoded |
| 27 | `subtipo_encoded` | float | Subtipo do chamado target-encoded |
| 28 | `orgao_encoded` | float | Órgão responsável target-encoded |
| 29 | `hist_resolution_rate_bairro` | float | Taxa histórica de resolução do bairro (treino) |
| 30 | `is_reclamacao` | binary | 1 se categoria contém 'reclamação' |
| 31 | `chamados_same_bairro_last_7d` | int | Chamados no mesmo bairro nos últimos 7 dias |
| 32 | `chamados_same_tipo_last_7d` | int | Chamados do mesmo tipo nos últimos 7 dias |
| 33 | `latitude_missing` | binary | 1 se latitude ausente |
| 34 | `longitude_missing` | binary | 1 se longitude ausente |

## Anti-Leakage Protocol

- **Target encoding:** CV out-of-fold para treino, média global do treino para teste
- **Rolling counts:** lookback estrito (exclui data atual)
- **Thresholds:** percentis calculados apenas no treino
- **Imputação:** mediana calculada apenas no treino
- **Scaler:** StandardScaler ajustado apenas no treino
- **data_fim NUNCA usada como feature**

## Target Variable

- `resolved_in_7_days`: 1 se (data_fim - data_inicio) <= 7 dias
- Train positive rate: 0.767
- Test positive rate: 0.783