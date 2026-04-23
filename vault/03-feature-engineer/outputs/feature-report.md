# Feature Engineering Report (Q5)

## Pipeline Summary

- **Amostra total:** 50,000 chamados
- **Treino (2023):** 24,048 chamados
- **Teste (2024):** 25,952 chamados
- **Total de features:** 34

## Dataset Splits

| Split | N | Positive | Negative | Positive Rate |
|-------|---|----------|----------|---------------|
| Train | 24,048 | 18,435 | 5,613 | 0.767 |
| Test  | 25,952 | 20,322 | 5,630 | 0.783 |

## Feature Categories

### Temporal (10 features)
- hour_of_day, day_of_week, day_of_month, month, quarter
- is_weekend, is_holiday, is_business_hours
- days_since_last_holiday, days_until_next_holiday

### Climate (8 features)
- temp_max, temp_min, temp_mean, precipitation_sum, rain_sum, windspeed_max
- is_extreme_rain (threshold from train: 13.4 mm)
- is_extreme_heat (threshold: 35°C)

### Geospatial (3 raw + 4 encoded + 1 historical = 8 features)
- latitude, longitude, coords_missing
- bairro_encoded, regiao_admin_encoded, area_plan_encoded, subprefeitura_encoded
- hist_resolution_rate_bairro

### Categorical (3 features)
- tipo_encoded, subtipo_encoded, orgao_encoded

### Contextual (3 features)
- is_reclamacao
- chamados_same_bairro_last_7d, chamados_same_tipo_last_7d

## Feature Statistics (Train)

| Feature | Mean | Std | Min | Max | Missing% |
|---------|------|-----|-----|-----|----------|
| hour_of_day | 0.000 | 1.000 | -2.889 | 2.180 | 0.0% |
| day_of_week | -0.000 | 1.000 | -1.350 | 2.064 | 0.0% |
| day_of_month | -0.000 | 1.000 | -1.686 | 1.712 | 0.0% |
| month | -0.000 | 1.000 | -1.561 | 1.556 | 0.0% |
| quarter | -0.000 | 1.000 | -1.312 | 1.313 | 0.0% |
| is_weekend | 0.128 | 0.335 | 0.000 | 1.000 | 0.0% |
| is_holiday | 0.015 | 0.123 | 0.000 | 1.000 | 0.0% |
| is_business_hours | 0.661 | 0.473 | 0.000 | 1.000 | 0.0% |
| days_since_last_holiday | -0.000 | 1.000 | -1.415 | 2.839 | 0.0% |
| days_until_next_holiday | -0.000 | 1.000 | -1.413 | 2.863 | 0.0% |
| temp_max | -0.000 | 1.000 | -2.694 | 3.510 | 0.0% |
| temp_min | 0.000 | 1.000 | -2.170 | 2.478 | 0.0% |
| temp_mean | -0.000 | 1.000 | -2.598 | 3.001 | 0.0% |
| precipitation_sum | -0.000 | 1.000 | -0.521 | 7.384 | 0.0% |
| rain_sum | -0.000 | 1.000 | -0.521 | 7.384 | 0.0% |
| windspeed_max | 0.000 | 1.000 | -2.253 | 3.812 | 0.0% |
| is_extreme_rain | 0.045 | 0.207 | 0.000 | 1.000 | 0.0% |
| is_extreme_heat | 0.022 | 0.148 | 0.000 | 1.000 | 0.0% |
| latitude | -0.000 | 1.000 | -3.719 | 97.015 | 0.0% |
| longitude | 0.000 | 1.000 | -4.586 | 14.807 | 0.0% |
| coords_missing | 0.556 | 0.497 | 0.000 | 1.000 | 0.0% |
| bairro_encoded | 0.000 | 1.000 | -3.579 | 1.457 | 0.0% |
| regiao_admin_encoded | -0.000 | 1.000 | -3.206 | 1.309 | 0.0% |
| area_plan_encoded | 0.000 | 1.000 | -1.055 | 1.399 | 0.0% |
| subprefeitura_encoded | -0.000 | 1.000 | -1.796 | 1.373 | 0.0% |
| tipo_encoded | -0.000 | 1.000 | -2.798 | 1.224 | 0.0% |
| subtipo_encoded | 0.000 | 1.000 | -3.119 | 1.077 | 0.0% |
| orgao_encoded | -0.000 | 1.000 | -3.313 | 1.150 | 0.0% |
| hist_resolution_rate_bairro | -0.000 | 1.000 | -4.096 | 2.206 | 0.0% |
| is_reclamacao | 0.036 | 0.187 | 0.000 | 1.000 | 0.0% |
| chamados_same_bairro_last_7d | 0.000 | 1.000 | -0.663 | 2.837 | 0.0% |
| chamados_same_tipo_last_7d | -0.000 | 1.000 | -0.295 | 8.784 | 0.0% |
| latitude_missing | 0.556 | 0.497 | 0.000 | 1.000 | 0.0% |
| longitude_missing | 0.556 | 0.497 | 0.000 | 1.000 | 0.0% |

## Output Files

| File | Path | Shape |
|------|------|-------|
| X_train | `data/features/X_train.parquet` | (24048, 34) |
| X_test | `data/features/X_test.parquet` | (25952, 34) |
| y_train | `data/features/y_train.parquet` | (24048, 1) |
| y_test | `data/features/y_test.parquet` | (25952, 1) |
| Scaler | `results/models/feature_scaler.joblib` | — |
| Encoders | `results/models/target_encoders.joblib` | — |

## Methodology Notes

1. **Amostragem:** 50K chamados amostrados via StratifiedShuffleSplit por (ano, target)
2. **Split temporal:** treino=2023, teste=2024 (sem split aleatório)
3. **Target encoding:** suavizado com Laplace smoothing (alpha=10), CV 5-fold no treino
4. **Rolling counts:** searchsorted vetorizado, lookback estrito de 7 dias
5. **Imputação:** mediana do treino aplicada em ambos splits
6. **Scaling:** StandardScaler ajustado no treino, transformado em ambos splits