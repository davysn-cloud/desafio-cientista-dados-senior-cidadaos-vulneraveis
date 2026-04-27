# Desafio Técnico — Cientista de Dados Sênior
## Programa Pequenos Cariocas (PIC) | Prefeitura do Rio de Janeiro

---

## Sobre a abordagem

Esse repositório é minha resposta ao desafio técnico. Usei os dados públicos do 1746
para responder às 10 questões, organizando o trabalho em três notebooks progressivos:
EDA com APIs externas → modelagem preditiva → sistema de priorização.

A escolha de criar módulos Python em `src/` (em vez de colocar tudo nos notebooks) foi
deliberada: facilita reusar lógica entre notebooks e deixa o código testável de forma
independente.

---

## Resultados Principais

### Parte 1 — Análise Exploratória

- **Clima e demanda (Q1):** precipitação tem correlação positiva com chamados de
  infraestrutura urbana (drenagem, árvores), mas negativa com alguns serviços de
  conservação. Análise de lag (precipitação acumulada em janelas de 1-5 dias) mostra
  que o efeito é predominantemente instantâneo no agregado, mas varia por tipo de serviço.

- **Padrões territoriais (Q2):** demanda muito concentrada — top 10 bairros respondem
  por fatia desproporcional do total. Mais relevante: o *perfil* de chamado muda por
  Área de Planejamento. APs periféricas têm mais chamados de infraestrutura básica.
  Possível viés de sub-notificação em bairros vulneráveis — menor acesso digital
  pode significar menos chamados apesar de problemas iguais ou piores.

- **Feriados e eventos extremos (Q3):** feriados reduzem volume. Eventos extremos
  (precipitação > p95 ou temperatura > 35°C) aumentam só os tipos ligados ao clima —
  para serviços administrativos, efeito próximo de zero. Mann-Whitney U confirma
  significância estatística nas categorias sensíveis.

- **Previsão de demanda (Q4):** Random Forest com R²=0.821 no teste (2024), superando
  Ridge (R²=0.791). Features de lag temporal dominam; clima entra como ajuste fino.

### Parte 2 — Modelagem Preditiva

| Modelo | Accuracy | Precisão | Recall | F1 | AUC-ROC |
|--------|----------|----------|--------|----|---------|
| Regressão Logística | 0.826 | 0.861 | 0.928 | 0.893 | 0.848 |
| Random Forest | 0.827 | 0.865 | 0.923 | 0.893 | 0.849 |
| XGBoost (default) | 0.814 | 0.861 | 0.908 | 0.884 | 0.851 |
| LightGBM | 0.824 | 0.867 | 0.915 | 0.891 | 0.861 |
| **XGBoost (tuned)** | **0.828** | **0.865** | **0.926** | **0.894** | **0.863** |

**Melhor modelo:** XGBoost tunado com Optuna (50 trials, 5-fold CV).
Métrica primária: F1 — equilibra o custo de falso negativo (cidadão sem atendimento)
com o custo de falso positivo (desperdício de recurso de triagem).
Threshold otimizado via curva PR (0.37 vs default 0.50), com ganho marginal em F1.

**Interpretabilidade (Q8):** subtipo e órgão responsável dominam as predições via SHAP.
A taxa histórica de resolução do bairro aparece forte — o que confirma Q2 e levanta
um alerta: o modelo pode perpetuar desigualdades territoriais se não for contrabalanceado.

### Parte 3 — Sistema de Priorização

Score composto: `priority = 0.40·P(atraso) + 0.20·urgência + 0.25·equidade + 0.15·contexto`

O componente de equidade (w=0.25, segundo maior peso) foi uma escolha deliberada:
um ranking puramente por P(atraso) excluiria sistematicamente regiões com baixa
resolução histórica — exatamente as mais vulneráveis.

| Estratégia | Precision@20% | Recall@20% | Lift |
|------------|--------------|-----------|------|
| Seleção aleatória | 21.8% | 20.1% | 1.0x |
| Score de prioridade | 58.3% | 53.7% | **2.68x** |

Com o mesmo orçamento de 20%, o sistema identifica 2.68x mais chamados que realmente
vão atrasar, mantendo cobertura em todas as Áreas de Planejamento.

---

## Como reproduzir

### Requisitos

```
Python 3.10+
pip install -r requirements.txt
```

### Credenciais BigQuery

Os dados já estão extraídos em `data/raw/`. Para re-extrair do zero, configure
as credenciais do Google Cloud conforme a
documentação do pacote `basedosdados` (disponível no PyPI).

### Executar os notebooks

```bash
# na raiz do projeto
jupyter notebook

# ou pela CLI
jupyter nbconvert --to notebook --execute --inplace notebooks/01_analise_apis_clima.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_modelagem_resolucao.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_sistema_priorizacao.ipynb
```

Os notebooks foram desenvolvidos e testados com Python 3.13 + kernel `desafio-pic`.
Se usar outro ambiente, instale as dependências do `requirements.txt` e registre
o kernel: `python -m ipykernel install --user --name desafio-pic`.

---

## Estrutura do Repositório

```
.
├── notebooks/
│   ├── 01_analise_apis_clima.ipynb     # Parte 1: Q1-Q4 (EDA + APIs)
│   ├── 02_modelagem_resolucao.ipynb    # Parte 2: Q5-Q8 (Feature eng + modelos)
│   └── 03_sistema_priorizacao.ipynb    # Parte 3: Q9-Q10 (Score + simulação)
│
├── src/
│   ├── data/           # Extração BigQuery e APIs externas
│   ├── eda/            # Pipeline de análise exploratória
│   ├── features/       # Feature engineering (Q5)
│   ├── models/         # Treino e avaliação dos modelos (Q6-Q8)
│   ├── prioritization/ # Score de priorização e simulação (Q9-Q10)
│   └── visualization/  # Plots reutilizáveis
│
├── data/
│   ├── raw/            # Dados extraídos (gitignored se > 100MB)
│   ├── processed/      # Dados limpos intermediários
│   └── features/       # X_train, X_test, y_train, y_test
│
├── results/
│   ├── figures/        # Todas as visualizações (Q1-Q10)
│   └── models/         # Modelos treinados (.joblib)
│
├── requirements.txt
└── README.md
```

---

## Principais Bibliotecas

| Biblioteca | Uso |
|---|---|
| `pandas`, `pyarrow` | Manipulação de dados e leitura de Parquet |
| `basedosdados`, `google-cloud-bigquery` | Extração do BigQuery |
| `scikit-learn` | Baseline, encoding, métricas |
| `xgboost`, `lightgbm` | Modelos avançados |
| `optuna` | Tuning de hiperparâmetros |
| `shap` | Interpretabilidade |
| `geopandas`, `folium` | Análise e visualização geoespacial |
| `matplotlib`, `seaborn`, `plotly` | Visualizações |
| `requests` | APIs de clima e feriados |

---

## Notas sobre os dados

- Filtro de partição aplicado em todas as queries: `data_particao >= '2023-01-01'`
- Amostra de 50k chamados para modelagem (Q5-Q10): 24.048 treino (2023) + 25.952 teste (2024)
- Feature engineering com protocolo anti-leakage rigoroso: target encoding por CV,
  thresholds calculados só no treino, `data_fim` nunca usada como feature
