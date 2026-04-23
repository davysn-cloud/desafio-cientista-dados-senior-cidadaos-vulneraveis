# Q2 — Padrões Geoespaciais

## Top 5 Bairros por Volume de Chamados

| Bairro | Total de Chamados |
|--------|-------------------|
| Campo Grande | 141,365 |
| Tijuca | 64,997 |
| Bangu | 54,380 |
| Barra da Tijuca | 53,961 |
| Santa Cruz | 51,688 |

## Distribuição por Área de Planejamento

| AP | Total | Densidade (chamados/km²) |
|----|-------|--------------------------|
| AP 3 | 627,448 | 3,083 |
| AP 5 | 497,817 | 870 |
| AP 2 | 363,517 | 3,619 |
| AP 4 | 282,901 | 963 |
| AP 1 | 140,690 | 4,090 |

## Clusters Espaciais (KMeans k=8)

| Cluster | Tamanho (amostra 50k) | Lat Centróide | Lon Centróide |
|---------|----------------------|---------------|---------------|
| 0 | 5,725 | -22.9686 | -43.1990 |
| 1 | 7,320 | -22.8289 | -43.2941 |
| 2 | 3,730 | -22.9369 | -43.6279 |
| 3 | 10,626 | -22.9176 | -43.2188 |
| 4 | 6,010 | -22.8822 | -43.5065 |
| 5 | 3,523 | -22.9353 | -43.3639 |
| 6 | 9,499 | -22.8836 | -43.3164 |
| 7 | 3,567 | -23.0036 | -43.4071 |

## Principais Achados

1. Os 5 bairros com maior volume concentram uma parcela significativa dos chamados, indicando forte desigualdade territorial.
2. A razão entre o bairro mais demandado e o 20° é de 6.6x.
3. A análise por Área de Planejamento revela diferenças expressivas de densidade quando normalizada pela área geográfica.
4. Os clusters espaciais identificam hotspots de demanda concentrados em regiões específicas da cidade.
5. Registros com coordenadas válidas: 1,249,453 (44.7% do total).

## Figuras
- `q2_top20_bairros.png`
- `q2_areas_planejamento.png`
- `q2_top15_regioes_admin.png`
- `q2_tipo_by_ap.png`
- `q2_spatial_clusters.png`
- `q2_density_ap.png`
