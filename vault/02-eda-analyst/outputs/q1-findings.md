# Q1 — Clima vs Demanda de Serviços

## Correlações Gerais (Total de Chamados vs Variáveis Climáticas)

| Variável | Pearson r | Spearman ρ |
|----------|-----------|------------|
| Temp Max | 0.162 | 0.221 |
| Temp Min | 0.197 | 0.252 |
| Temp Média | 0.206 | 0.267 |
| Precipitação | -0.055 | -0.028 |
| Chuva | -0.055 | -0.028 |
| Vento Max | -0.021 | -0.009 |

## Tipos Mais Sensíveis à Precipitação

| Tipo | Spearman ρ (precipitação) | p-valor |
|------|--------------------------|---------|
| Estacionamento irregular | -0.259 | 1.24e-12 |
| Remoção Gratuita | -0.089 | 1.62e-02 |
| Iluminação Pública | -0.082 | 2.66e-02 |
| Perturbação do sossego | -0.079 | 3.21e-02 |
| Pavimentação | -0.074 | 4.66e-02 |

## Principais Achados

1. A precipitação apresenta correlação negativa (ρ=-0.028) com o volume total de chamados.
2. A temperatura média mostra correlação de ρ=0.267 com a demanda.
3. Alguns tipos de chamado são significativamente mais sensíveis às condições climáticas.
4. O vento máximo apresenta correlação de ρ=-0.009 com o volume.

## Figuras
- `q1_correlation_heatmap.png`
- `q1_scatter_precipitation_sensitive.png`
- `q1_timeseries_precip_chamados.png`
- `q1_timeseries_temp_chamados.png`
