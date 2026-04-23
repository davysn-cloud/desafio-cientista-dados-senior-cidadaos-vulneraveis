# Q9 — Formula do Score de Priorizacao

## Formula

```
priority_score = w1 * P(atraso) + w2 * urgency_score + w3 * equity_score + w4 * context_score
```

## Pesos

| Componente | Peso | Justificativa |
|---|---|---|
| **P(Atraso)** — `w1` | 0.40 | O preditor de atraso do modelo XGBoost (AUC=0.8628) e a informacao mais rica e individualizada. Recebe o maior peso. |
| **Urgencia (Subtipo)** — `w2` | 0.20 | O tipo de servico determina a complexidade intrinseca do chamado. Subtipos com baixa taxa de resolucao historica sao inerentemente mais dificeis. |
| **Equidade Territorial** — `w3` | 0.25 | Bairros historicamente mal atendidos precisam de atencao compensatoria. Peso elevado para garantir justica distributiva. |
| **Contexto Climatico** — `w4` | 0.15 | Eventos extremos (chuva forte, calor extremo) aumentam a urgencia pontual. Peso menor pois e um fator transitorio. |

**Soma dos pesos:** 1.00

## Componentes — Estatisticas Descritivas

| Componente | Media | Mediana | Std | Min | Max |
|---|---|---|---|---|---|
| P(Atraso) | 0.206 | 0.081 | 0.261 | 0.001 | 0.981 |
| Urgencia | 0.232 | 0.162 | 0.231 | 0.000 | 1.000 |
| Equidade | 0.358 | 0.390 | 0.159 | 0.000 | 1.000 |
| Contexto | 0.062 | 0.000 | 0.178 | 0.000 | 1.000 |


## Normalizacao
Cada componente e normalizado para [0, 1] via Min-Max antes da combinacao.
O score final tambem e normalizado para [0, 1].

## Trade-offs

### Preditivo vs. Equitativo
- Pesos maiores em P(Atraso) maximizam a precisao (identifica mais casos que realmente atrasam).
- Pesos maiores em Equidade aumentam a cobertura territorial, garantindo que bairros vulneraveis recebam atencao mesmo quando o modelo nao preve alto risco individual.

### Sensibilidade
- A analise de sensibilidade (`q9_weight_sensitivity.png`) mostra que o recall e robusto a variacoes moderadas dos pesos.
- P(Atraso) e o componente mais influente: aumentar w1 melhora recall mas pode concentrar atencao em poucos bairros.
- Equidade age como contrapeso geografico, garantindo diversidade territorial.

## Visualizacao
- Distribuicao dos componentes: `results/figures/q9_score_components.png`
- Sensibilidade dos pesos: `results/figures/q9_weight_sensitivity.png`
