# Task: Q3 Extreme Events and Holidays Impact

## Objective
Compare chamado patterns during normal days, holidays, and extreme weather events.

## Steps
1. Define extreme weather criteria:
   - Extreme rain: precipitation > 95th percentile
   - Extreme heat: temperature_max > 35C
   - Storm: weathercode >= 95
2. Classify each day as: normal, holiday, extreme_weather, both
3. Compare daily volume distributions across categories
4. Statistical significance tests (Mann-Whitney U, Kruskal-Wallis)
5. Breakdown by chamado type: which types spike during events?
6. Territorial analysis: which regions are most affected?

## Visualizations
- Box plots: daily volume by day category
- Bar charts: top chamado types during extreme events
- Geographic overlay: extreme event impact by bairro

## Output
- Figures in results/figures/q3_*
- Findings in vault/02-eda-analyst/outputs/q3-findings.md
