# Task: Q1 Climate vs Service Demand Correlation

## Objective
Investigate the relationship between weather conditions and 1746 call volume.

## Steps
1. Load chamados and aggregate to daily counts (total + by tipo)
2. Load weather data and merge by date
3. Compute Pearson and Spearman correlations
4. Create correlation heatmap (weather vars vs top chamado types)
5. Create scatter plots for most correlated pairs
6. Time series overlay: precipitation + chamado volume
7. Write interpretation of climate-sensitive service categories

## Key Questions to Answer
- Which chamado types increase with rain?
- Does heat affect service demand?
- Is the relationship linear or threshold-based?

## Output
- Figures in results/figures/q1_*
- Findings in vault/02-eda-analyst/outputs/q1-findings.md
