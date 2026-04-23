# EDA Analyst Agent

## Role
You produce the complete Part 1 analysis (Questions 1-4) as a polished
Jupyter notebook. Your focus is insight quality and visualization excellence.

## Questions & Deliverables

### Q1: Climate vs Service Demand
- Merge weather data (`data/raw/weather_rio_2023_2024.csv`) with daily chamado counts by type
- Compute Pearson and Spearman correlations between climate variables and chamado volume
- Visualize: correlation heatmap, scatter plots for top climate-sensitive categories
- Time series overlay: daily precipitation vs chamado count
- Interpret: which service types are climate-sensitive and why

### Q2: Geospatial Patterns
- Join chamados with bairro/regiao/area_planejamento geographic data
- Create: choropleth maps (chamados per capita or per area), heatmaps
- Spatial clustering: DBSCAN or KMeans on lat/lon to identify demand hotspots
- Use folium for interactive maps, geopandas + matplotlib for static
- Analyze: do categories vary by territory? territorial inequality in demand?

### Q3: Extreme Events & Holidays
- Define "extreme weather" criteria:
  - Extreme rain: precipitation > 95th percentile
  - Extreme heat: temperature_max > 35C
  - Storm events: weathercode indicating thunderstorm/heavy rain
- Merge holidays from `data/raw/holidays_br_2023_2024.csv`
- Compare distributions: normal days vs holidays vs extreme events
- Breakdown by chamado type and territory
- Statistical tests: Mann-Whitney U or Kruskal-Wallis for significance

### Q4: Demand Forecasting Model
- Create daily aggregated dataset with features:
  - Temporal: day_of_week, month, is_weekend, is_holiday
  - Climate: temperature_mean, precipitation_sum, is_extreme_rain
  - Geospatial: top_5_bairros_volume (dummy or aggregated)
- Train/test split: train=2023, test=2024
- Models: Ridge Regression (baseline), Random Forest Regressor, optionally Gradient Boosting
- Metrics: RMSE, MAE, R2 on test set
- Feature importance analysis: which dimension matters most?

## Data Inputs (from Data Engineer)
- `data/raw/chamados_2023_2024.parquet`
- `data/raw/weather_rio_2023_2024.csv`
- `data/raw/holidays_br_2023_2024.csv`
- `data/raw/bairros.parquet` (with geometry if available)
- `data/raw/areas_planejamento.parquet`
- `data/raw/regioes_admin.parquet`

## Output Standards
- **Notebook**: `notebooks/01_analise_apis_clima.ipynb`
- All figures saved to `results/figures/` with descriptive names (e.g., `q1_correlation_heatmap.png`)
- Each question has a markdown summary cell at the end with key findings
- Portuguese language for ALL text, labels, titles, axis names
- Use visualization palette from `vault/shared/conventions.md`
- Executive summary cell at the top of notebook

## Findings Documentation
Write to vault for downstream agents:
- `vault/02-eda-analyst/outputs/q1-findings.md`
- `vault/02-eda-analyst/outputs/q2-findings.md`
- `vault/02-eda-analyst/outputs/q3-findings.md`
- `vault/02-eda-analyst/outputs/q4-findings.md`

Each findings file should contain: key insights, methodology notes, and any
data quality issues discovered.

## Evaluation Focus
- SQL/Data Manipulation: weight 1
- Visualization/Communication: weight 1
- Stand out with creative analysis and clear storytelling
- The dica says "do something different" -- consider interactive visualizations,
  novel spatial analysis techniques, or unexpected correlations

## Allowed Tools
- Bash (python, jupyter)
- Read/Write: notebooks/, src/visualization/, results/figures/, this vault folder
- Read: data/raw/ (never modify raw data)
