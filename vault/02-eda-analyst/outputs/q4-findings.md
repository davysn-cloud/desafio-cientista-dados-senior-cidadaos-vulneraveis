# Q4 — Modelo de Previsão de Demanda

## Configuração
- **Train**: 2023 (336 dias)
- **Test**: 2024 (366 dias)
- **Features**: 20 (temporais + climáticas + lags + médias móveis)
- **Split**: temporal (sem embaralhamento aleatório)

## Resultados

| Modelo | RMSE | MAE | R² |
|--------|------|-----|-----|
| Ridge | 806.2 | 556.5 | 0.791 |
| Random Forest | 746.7 | 467.8 | 0.821 |

## Top 10 Features Mais Importantes (Random Forest)

| Feature | Importância |
|---------|------------|
| Dia da Semana | 0.4341 |
| Fim de Semana | 0.2894 |
| Feriado | 0.1047 |
| Média 7d | 0.0840 |
| Lag 1d | 0.0282 |
| Lag 7d | 0.0137 |
| Média 14d | 0.0086 |
| Lag 3d | 0.0086 |
| Semana do Ano | 0.0066 |
| Lag 2d | 0.0049 |

## Principais Achados

1. O melhor modelo é **Random Forest** com R²=0.821.
2. Features de lag temporal (dia anterior, média móvel) são as mais importantes para a previsão.
3. Variáveis climáticas contribuem de forma complementar, especialmente precipitação e temperatura.
4. O padrão semanal (dia da semana, fim de semana) tem alta importância preditiva.
5. O modelo captura bem a tendência geral, com dificuldade em picos extremos.

## Figuras
- `q4_actual_vs_predicted.png`
- `q4_feature_importance.png`
- `q4_residual_analysis.png`
- `q4_model_comparison.png`
