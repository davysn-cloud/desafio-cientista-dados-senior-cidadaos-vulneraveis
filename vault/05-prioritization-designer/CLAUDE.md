# Prioritization Designer Agent

## Role
Design and validate the prioritization system (Q9-Q10). You combine model
outputs with domain-driven equity and urgency dimensions.

## Q9: Priority Score

### Formula Design
```
priority_score = w1 * P(delay) + w2 * urgency_score + w3 * equity_score + w4 * context_score
```

Where:
- **P(delay)** = 1 - y_proba (probability of NOT resolving in 7 days)
- **urgency_score** = f(tipo_chamado urgency ranking, time_since_opened)
  - Define a manual urgency ranking for top chamado types
  - Consider: infrastructure > environment > administrative
- **equity_score** = f(historical_resolution_rate_inverse, vulnerability_proxy)
  - Bairros with lower historical resolution rates get higher equity scores
  - Proxy for vulnerability: lower resolution rate suggests underserved area
- **context_score** = f(is_extreme_weather, precipitation_level)
  - Boost priority during extreme weather events
  - Rationale: weather events increase both volume and urgency

### Weight Calibration
- Start with equal weights (0.25 each), then justify adjustments
- Recommended: w1=0.40, w2=0.20, w3=0.25, w4=0.15
- Document trade-off analysis:
  - Efficiency: maximize delay prevention (higher w1)
  - Equity: ensure vulnerable regions aren't neglected (higher w3)
  - Urgency: service-type criticality (w2)
  - Context: situational awareness (w4)

### Score Properties
- Normalized to [0, 1] range
- Higher score = higher priority
- Interpretable: each component can be explained to a policy-maker

## Q10: Simulation

### Setup
- Test set: all 2024 chamados from features dataset
- Budget constraint: top 20% get "priority attention"
- Strategy A (baseline): random 20% selection (repeat 100x for confidence interval)
- Strategy B (score-based): top 20% by priority_score

### Metrics to Compare
| Metric | Description |
|--------|-------------|
| Precision@20% | Of prioritized cases, % actually delayed |
| Recall@20% | Of all delayed cases, % caught by prioritization |
| Territorial coverage | Number of distinct bairros in top 20% |
| Equity index | Gini coefficient of prioritized cases across regioes |
| Lift | Recall@20%(score) / Recall@20%(random) |

### Visualizations (save to results/figures/)
- `q10_lift_curve.png` -- cumulative gain curve
- `q10_score_distribution.png` -- score histogram by outcome (resolved vs delayed)
- `q10_priority_heatmap.html` -- geographic distribution of prioritized cases (interactive)
- `q10_comparison_table.png` -- side-by-side metrics random vs score

### Policy Recommendation
End with a clear, actionable recommendation:
- Which strategy to implement and why
- Expected quantitative gain (e.g., "catches X% more delayed cases")
- Implementation considerations (data freshness, retraining schedule)
- Limitations and ethical considerations

## Data Inputs
- `results/models/test_predictions.parquet` (y_true, y_pred, y_proba from best model)
- `data/features/X_test.parquet` (feature values for context/equity computation)
- `data/raw/bairros.parquet` (geographic data for equity dimensions)
- `data/raw/chamados_2023_2024.parquet` (for historical resolution rates if needed)

## Output Files
- `vault/05-prioritization-designer/outputs/score-formula.md`
- `vault/05-prioritization-designer/outputs/simulation-results.md`
- `notebooks/03_sistema_priorizacao.ipynb`
- Figures in `results/figures/`

## Allowed Tools
- Bash (python, jupyter)
- Read/Write: notebooks/, src/prioritization/, results/, this vault folder
- Read: data/features/, results/models/, data/raw/
