# Q3 — Eventos Extremos e Feriados

## Definições
- **Evento Extremo**: precipitação > P95 (14.0 mm) OU temperatura máxima > 35°C
- **Feriado**: conforme calendário oficial brasileiro 2023-2024

## Contagens
- Dias normais: 642
- Feriados: 27
- Eventos extremos: 60
- Extremo + Feriado: 2

## Testes Estatísticos (Mann-Whitney U)

| Comparação | n | Média Normal | Média Grupo | U | p-valor | Significativo (α=0.05) |
|-----------|---|-------------|-------------|---|---------|----------------------|
| Normal vs Feriado | 27 | 3936 | 1652 | 15150 | 4.42e-11 | Sim |
| Normal vs Evento Extremo | 60 | 3936 | 3627 | 21216 | 1.93e-01 | Não |

## Principais Achados

1. Eventos extremos de clima afetam de forma não significativa o volume de chamados.
2. Feriados reduzem o volume médio de chamados em relação a dias normais.
3. Diferentes tipos de chamado respondem de formas distintas a eventos extremos e feriados.

## Figuras
- `q3_distribution_day_types.png`
- `q3_tipo_by_day_type.png`
- `q3_timeline_events.png`
