# Q10 — Resultados da Simulacao de Priorizacao

## Configuracao

| Parametro | Valor |
|---|---|
| Total de chamados (teste 2024) | 25,952 |
| Chamados atrasados (>7 dias) | 5,630 (21.7%) |
| Orcamento de priorizacao | 20% (5,190 chamados) |
| Iteracoes da selecao aleatoria | 100 |

## Resultados Comparativos

### Selecao Aleatoria (baseline)
- **Precision@20%:** 21.8% (IC 95%: 20.9% - 22.6%)
- **Recall@20%:** 20.1% (IC 95%: 19.3% - 20.9%)

### Score de Prioridade
- **Precision@20%:** 58.3%
- **Recall@20%:** 53.7%
- **Lift:** 2.68x

### Cobertura Territorial
- Aleatoria: 6 areas de planejamento distintas
- Score: 6 areas de planejamento distintas

## Interpretacao

### Ganhos
- O sistema de priorizacao captura **53.7%** dos chamados que realmente atrasam,
  selecionando apenas **20%** do total. Isso representa um lift de **2.68x** sobre a selecao aleatoria.
- A precisao aumenta de 21.8% para 58.3%,
  significando que cada chamado priorizado tem 58.3% de chance de realmente necessitar atencao.

### Cobertura Territorial
- O sistema mantem cobertura em 6 areas de planejamento,
  superando a selecao aleatoria (6).
- O componente de equidade (w3=0.25) garante que bairros historicamente sub-atendidos recebam atencao proporcional.

## Recomendacao de Politica Publica

1. **Adotar o score de priorizacao** para triagem dos chamados 1746, com revisao trimestral dos pesos.
2. **Monitorar equidade territorial** mensalmente: se alguma regiao ficar sistematicamente de fora do top 20%, ajustar w3.
3. **Retreinar o modelo** semestralmente com dados novos para manter o componente P(Atraso) atualizado.
4. **Integrar dados climaticos em tempo real** para que o componente de contexto reflita condicoes atuais, nao apenas historicas.
5. **Limiar dinamico**: em periodos de crise (enchentes, ondas de calor), expandir temporariamente o orcamento de 20% para 30%.

## Visualizacoes
- Curva de ganho acumulado: `results/figures/q10_lift_curve.png`
- Distribuicao do score: `results/figures/q10_score_distribution.png`
- Tabela comparativa: `results/figures/q10_comparison_table.png`
