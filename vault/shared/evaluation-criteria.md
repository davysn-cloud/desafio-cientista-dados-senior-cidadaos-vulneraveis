# Evaluation Criteria

## Weights
| Category | Weight | Scope |
|----------|--------|-------|
| SQL e Manipulacao de Dados | 1 | Data extraction, joins, aggregations, feature engineering |
| Modelagem e Python | **2** | Model training, tuning, evaluation, SHAP, prioritization |
| Visualizacao e Comunicacao | 1 | Charts, maps, narrative quality, README |

## Scoring Formula
```
final_score = (sql_score * 1 + modeling_score * 2 + viz_score * 1) / 4
```

## Implications for Agent Priorities
- **Model Builder (Agent 4)** produces the highest-weight work -- invest most effort here
- Feature engineering directly impacts model quality -- make it thorough
- Visualizations must be publication-quality, not just functional
- "Faca algo diferente!" -- the evaluators see many submissions; stand out

## What "Standing Out" Means
- **Don't**: basic scatter plots, default matplotlib styling, minimal text
- **Do**: interactive maps (folium/plotly), creative spatial analysis,
  equity-aware recommendations, clear policy narratives
- **Bonus**: the multi-agent architecture itself is a differentiator
  (mention it in README)
