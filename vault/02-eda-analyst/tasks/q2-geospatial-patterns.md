# Task: Q2 Geospatial Demand Patterns

## Objective
Analyze spatial distribution of chamados to identify territorial patterns.

## Steps
1. Join chamados with geographic reference tables (bairro, regiao, AP)
2. Aggregate counts by bairro, regiao_administrativa, area_planejamento
3. Create choropleth map (chamados per bairro)
4. Create density heatmap from lat/lon coordinates
5. Apply spatial clustering (DBSCAN or KMeans)
6. Analyze category variation by territory
7. Identify territorial inequalities in demand patterns

## Visualizations
- Choropleth map (folium or plotly)
- Heatmap of call density
- Bar charts by regiao_administrativa
- Category breakdown by territory

## Output
- Figures in results/figures/q2_*
- Interactive map: results/figures/q2_mapa_coropletico.html
- Findings in vault/02-eda-analyst/outputs/q2-findings.md
