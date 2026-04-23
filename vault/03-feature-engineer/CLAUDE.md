# Feature Engineer Agent

## Role
Build the definitive feature-engineered dataset (Q5). Your output is the
single input for all modeling work. Quality here determines model ceiling.

## Target Variable
- `resolved_in_7_days`: 1 if (data_fim - data_inicio) <= 7 days, else 0
- Exclude chamados still open (no data_fim) unless you justify imputation
- Document class distribution in feature-report.md

## Sampling Strategy
- Total: 50,000 chamados from 2023-2024
- Split: train = chamados from 2023, test = chamados from 2024
- Stratify by target to maintain class balance across splits
- Document exact counts in feature-report.md

## Feature Categories (ALL required)

### Temporal Features
- `hour_of_day`: hour extracted from data_inicio (0-23)
- `day_of_week`: 0=Monday to 6=Sunday
- `day_of_month`: 1-31
- `month`: 1-12
- `quarter`: 1-4
- `is_weekend`: 1 if Saturday/Sunday
- `is_holiday`: 1 if date matches holidays_br_2023_2024.csv
- `is_business_hours`: 1 if weekday AND 8<=hour<18
- `days_since_last_holiday`: integer
- `days_until_next_holiday`: integer

### Climate Features (join by date from weather data)
- `temperature_max`: daily max temperature
- `temperature_min`: daily min temperature
- `temperature_mean`: daily mean temperature
- `precipitation_sum`: daily total precipitation (mm)
- `rain_sum`: daily total rain (mm)
- `windspeed_max`: daily max wind speed
- `is_extreme_rain`: 1 if precipitation > 95th percentile of training data
- `is_extreme_heat`: 1 if temperature_max > 35C
- `weather_code_category`: categorical grouping of weathercode

### Geospatial Features
- `bairro_encoded`: target-encoded bairro name (leakage-safe)
- `regiao_administrativa`: label or target encoded
- `area_planejamento`: label or target encoded
- `subprefeitura`: label or target encoded
- `latitude`: float (if available in raw data)
- `longitude`: float (if available in raw data)
- `historical_resolution_rate_bairro`: % resolved in 7 days per bairro
  (computed ONLY from training data)

### Categorical Features
- `tipo_encoded`: target-encoded tipo_chamado
- `subtipo_encoded`: target-encoded subtipo
- `orgao_encoded`: target-encoded orgao_responsavel

### Contextual Features
- `is_reclamacao`: 1 if chamado is a complaint (from status or categoria)
- `chamados_same_bairro_last_7d`: count of chamados in same bairro in past 7 days
  (leakage-safe: use only data prior to chamado's data_inicio)
- `chamados_same_tipo_last_7d`: count of same-type chamados in past 7 days

## Processing Standards

### Missing Values
- Document missing rate per column before processing
- Numeric: median imputation (from training set only)
- Categorical: mode imputation or "unknown" category
- Coordinates: impute from bairro centroid if available
- Create `_missing` indicator flags for columns with >5% missing

### Encoding
- Target encoding: use 5-fold CV on training set to prevent leakage
- For test set: use global training set mean for target encoding
- One-hot encoding: only for low-cardinality features (<10 categories)

### Normalization
- StandardScaler for continuous features
- Fit on train, transform both train and test
- Save scaler object to `results/models/feature_scaler.joblib`

## Anti-Leakage Protocol (CRITICAL)
- NEVER use `data_fim` or any derivative as a feature
- NEVER use information from the future relative to data_inicio
- Target encoding: compute means ONLY from training fold
- Rolling/window features: strict lookback from data_inicio
- Historical rates: compute ONLY from training set
- Percentile thresholds (e.g., extreme rain): compute from training set only

## Output Files
- `data/features/X_train.parquet` -- features for training set
- `data/features/X_test.parquet` -- features for test set (same columns, same order)
- `data/features/y_train.parquet` -- single column: resolved_in_7_days
- `data/features/y_test.parquet` -- single column: resolved_in_7_days
- `results/models/feature_scaler.joblib` -- fitted scaler
- `results/models/target_encoders.joblib` -- fitted target encoders

## Documentation
- `vault/03-feature-engineer/outputs/feature-catalog.md`: every feature with
  name, dtype, source, transformation logic, missing_rate
- `vault/03-feature-engineer/outputs/feature-report.md`: distributions,
  correlations with target, class balance, rationale for decisions

## Allowed Tools
- Bash (python scripts)
- Read/Write: data/features/, src/features/, results/models/, this vault folder
- Read: data/raw/ (never modify)
