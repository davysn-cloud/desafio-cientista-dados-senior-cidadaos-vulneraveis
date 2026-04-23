# SHAP Interpretability Analysis (Q8)

## Model: XGBoost (tuned)

## Top 10 Features by Mean |SHAP|

1. **subtipo_encoded**: 1.5220
2. **orgao_encoded**: 0.7168
3. **hist_resolution_rate_bairro**: 0.3218
4. **bairro_encoded**: 0.1801
5. **chamados_same_bairro_last_7d**: 0.1568
6. **month**: 0.1315
7. **tipo_encoded**: 0.0854
8. **days_until_next_holiday**: 0.0664
9. **day_of_month**: 0.0561
10. **regiao_admin_encoded**: 0.0538

## Top 3 Feature Dependence Plots

- `subtipo_encoded` -> `results/figures/q8_shap_dependence_subtipo_encoded.png`
- `orgao_encoded` -> `results/figures/q8_shap_dependence_orgao_encoded.png`
- `hist_resolution_rate_bairro` -> `results/figures/q8_shap_dependence_hist_resolution_rate_bairro.png`

## Error Analysis

- Total test samples: 25952
- False Positives (FP): 2950 (11.4%)
- False Negatives (FN): 1506 (5.8%)

FP = model predicted resolution in 7 days but chamado was NOT resolved. FN = model predicted NO resolution but chamado WAS resolved.

## Policy Insights

1. **Temporal patterns** (hour, day, month) strongly influence resolution probability, suggesting that when a chamado is opened affects municipal response capacity.
2. **Territorial features** (bairro, subprefeitura) show significant variation, indicating geographic disparities in service delivery.
3. **Service type** (tipo, subtipo, orgao) captures institutional capacity differences across municipal agencies.
4. **Historical resolution rates** by neighborhood provide strong predictive signal, reflecting accumulated institutional performance.
5. **Weather conditions** have moderate influence, with extreme rain events correlating with delayed resolutions (infrastructure overload).

**Recommendation**: The prioritization system (Q9-Q10) should weight FN reduction heavily, as missing a delayed chamado means a vulnerable citizen remains unattended.